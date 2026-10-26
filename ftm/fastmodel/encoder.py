"""Fine-tuned ModernBERT / DeBERTa classifiers (and a TF-IDF floor) for the five small decisions: the E10 component baselines.

Owner: T2 (Bhavya).  Spec: plan.md 7.3 and 8 (E10), tracks.md 4.9 ("fine-tuned ModernBERT/DeBERTa classifiers ... on the small-decision
set. These fit on your 4 GB GPU"), week 10.  Data: `scripts/make_decision_set.py` (hand-off #13); the five decisions and their
labels are T3's `ftm/fastmodel/decisions.py`.

One classifier per decision type, fine-tuned on the TRAIN rows of the decision set and nothing else. The context text goes in as
the input; the decision's question is fixed per classifier so it is not repeated. `EncoderDecisionModel.raw_logps(ex)` returns the
label log-probabilities in `DECISIONS` order, exactly what `LogprobFastModel.raw_logps` returns, so T3's temperature scaling,
conformal cascade and `scripts/score_fast_baseline.py report` take a classifier as the fast tier without change.

Models (pinned, `ENCODERS`): ModernBERT-base (Apache-2.0, 149M), DeBERTa-v3-base and -small (MIT). 4 GB of VRAM: bf16 autocast,
max length 256, batch 16, gradient checkpointing when asked. Needs torch + transformers (not in the base install; CI skips).
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ftm.fastmodel.decisions import DECISIONS, DecisionExample

ENCODERS = {  # name -> (hub id, revision)
    "modernbert-base": ("answerdotai/ModernBERT-base", "8949b909ec900327062f0ebf497f51aef5e6f0c8"),
    "deberta-v3-base": ("microsoft/deberta-v3-base", "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"),
    "deberta-v3-small": ("microsoft/deberta-v3-small", "a36c739020e01763fe789b4b85e2df55d6180012"),
}
FLOOR_LOGP = math.log(1e-6)


@dataclass
class EncoderConfig:
    model: str = "modernbert-base"
    max_len: int = 256
    epochs: int = 3
    batch_size: int = 16
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup: float = 0.1
    seed: int = 0
    grad_checkpointing: bool = False
    max_rows: int | None = None  # per decision, drawn with `seed`; None = all


def _seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pretrained(cfg: EncoderConfig, n_labels: int):
    """(tokenizer, model) from the pinned hub revision, with a fresh classification head of `n_labels`."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    hub, rev = ENCODERS[cfg.model]
    tok = AutoTokenizer.from_pretrained(hub, revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(hub, revision=rev, num_labels=n_labels)
    return tok, model


def _batches(items: Sequence, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def train_decision_encoder(rows: Sequence[DecisionExample], decision: str, cfg: EncoderConfig, tokenizer=None, model=None,
                           device: str | None = None, log: Callable[[str], None] | None = None) -> tuple[object, object, list[dict]]:  # fmt: skip
    """Fine-tune one classifier on the rows of `decision`. Returns (tokenizer, model, per-epoch history). `tokenizer` and
    `model` can be passed in (tests use a tiny random model); otherwise they come from the pinned hub revision."""
    import torch
    from transformers import get_linear_schedule_with_warmup

    d = DECISIONS[decision]
    rows = [r for r in rows if r.decision == decision]
    bad = {r.label for r in rows} - set(d.labels)
    if bad:
        raise ValueError(f"{decision}: labels {sorted(bad, key=str)} are not in {d.labels}")
    if not rows:
        raise ValueError(f"no training rows for {decision!r}")
    if cfg.max_rows and len(rows) > cfg.max_rows:
        rows = random.Random(cfg.seed).sample(rows, cfg.max_rows)
    _seed_all(cfg.seed)
    if tokenizer is None or model is None:
        tokenizer, model = load_pretrained(cfg, len(d.labels))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    if cfg.grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    y_all = [d.labels.index(r.label) for r in rows]
    steps = cfg.epochs * math.ceil(len(rows) / cfg.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup * steps), steps)
    rng = random.Random(cfg.seed)
    history = []
    for epoch in range(cfg.epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        total, correct, n = 0.0, 0, 0
        for idx in _batches(order, cfg.batch_size):
            enc = tokenizer([rows[i].context for i in idx], truncation=True, max_length=cfg.max_len, padding=True,
                            return_tensors="pt").to(device)  # fmt: skip
            y = torch.tensor([y_all[i] for i in idx], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(**enc).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            total += float(loss) * len(idx)
            correct += int((logits.argmax(-1) == y).sum())
            n += len(idx)
        history.append({"epoch": epoch + 1, "loss": total / n, "train_accuracy": correct / n, "rows": n})
        if log:
            log(f"{decision} epoch {epoch + 1}/{cfg.epochs}: loss {history[-1]['loss']:.4f} acc {history[-1]['train_accuracy']:.3f}")
    model.eval()
    return tokenizer, model, history


class EncoderDecisionModel:
    """The fine-tuned classifiers of the decisions, as a fast-model tier (`raw_logps`, `predict_proba`)."""

    def __init__(self, parts: dict[str, tuple[object, object]], max_len: int = 256, batch_size: int = 32,
                 device: str | None = None):  # fmt: skip
        import torch

        self.parts, self.max_len, self.batch_size = parts, max_len, batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        for _, model in parts.values():
            model.to(self.device).eval()

    def logps_batch(self, decision: str, contexts: Sequence[str]) -> np.ndarray:
        """(n, k) label log-probabilities."""
        import torch

        if decision not in self.parts:
            raise KeyError(f"no classifier was trained for {decision!r}; have {sorted(self.parts)}")
        tok, model = self.parts[decision]
        out = []
        order = sorted(range(len(contexts)), key=lambda i: len(contexts[i]))  # similar lengths together: less padding
        with torch.no_grad():
            for idx in _batches(order, self.batch_size):
                enc = tok([contexts[i] for i in idx], truncation=True, max_length=self.max_len, padding=True,
                          return_tensors="pt").to(self.device)  # fmt: skip
                out.append(torch.log_softmax(model(**enc).logits.float(), dim=-1).cpu().numpy())
        flat = np.concatenate(out) if out else np.zeros((0, len(DECISIONS[decision].labels)))
        res = np.empty_like(flat)
        res[order] = flat
        return res

    def raw_logps(self, ex: DecisionExample) -> np.ndarray:
        return self.logps_batch(ex.decision, [ex.context])[0]

    def predict_proba(self, ex: DecisionExample) -> dict[str, float]:
        p = np.exp(self.raw_logps(ex))
        return dict(zip(DECISIONS[ex.decision].labels, (p / p.sum()).tolist(), strict=True))

    def save(self, directory: str | Path, meta: dict | None = None) -> Path:
        d = Path(directory)
        for decision, (tok, model) in self.parts.items():
            model.save_pretrained(d / decision)
            tok.save_pretrained(d / decision)
        (d / "meta.json").write_text(json.dumps({"decisions": sorted(self.parts), "max_len": self.max_len, **(meta or {})}, indent=1))
        return d

    @classmethod
    def load(cls, directory: str | Path, device: str | None = None, batch_size: int = 32) -> EncoderDecisionModel:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        parts = {x: (AutoTokenizer.from_pretrained(d / x), AutoModelForSequenceClassification.from_pretrained(d / x))
                 for x in meta["decisions"]}  # fmt: skip
        return cls(parts, meta.get("max_len", 256), batch_size, device)


@dataclass
class TfidfDecisionModel:
    """The floor: TF-IDF (1-2 grams) + logistic regression per decision. Same interface as the encoders; runs anywhere."""

    max_features: int = 20000
    c: float = 4.0
    seed: int = 0
    parts: dict = field(default_factory=dict)

    def fit(self, rows: Sequence[DecisionExample]) -> TfidfDecisionModel:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        for decision in sorted({r.decision for r in rows}):
            rs = [r for r in rows if r.decision == decision]
            vec = TfidfVectorizer(ngram_range=(1, 2), max_features=self.max_features, sublinear_tf=True)
            X = vec.fit_transform([r.context for r in rs])
            clf = LogisticRegression(C=self.c, max_iter=1000, random_state=self.seed)
            clf.fit(X, [r.label for r in rs])
            self.parts[decision] = (vec, clf)
        return self

    def logps_batch(self, decision: str, contexts: Sequence[str]) -> np.ndarray:
        if decision not in self.parts:
            raise KeyError(f"no model was fitted for {decision!r}; have {sorted(self.parts)}")
        vec, clf = self.parts[decision]
        labels = DECISIONS[decision].labels
        out = np.full((len(contexts), len(labels)), FLOOR_LOGP)  # a label never seen in training stays at the floor
        lp = np.log(np.clip(clf.predict_proba(vec.transform(contexts)), 1e-6, 1.0))
        for j, lab in enumerate(clf.classes_):
            out[:, labels.index(lab)] = lp[:, j]
        return out

    def raw_logps(self, ex: DecisionExample) -> np.ndarray:
        return self.logps_batch(ex.decision, [ex.context])[0]

    def predict_proba(self, ex: DecisionExample) -> dict[str, float]:
        p = np.exp(self.raw_logps(ex))
        return dict(zip(DECISIONS[ex.decision].labels, (p / p.sum()).tolist(), strict=True))


def config_dict(cfg: EncoderConfig) -> dict:
    hub, rev = ENCODERS[cfg.model]
    return {**asdict(cfg), "hub_id": hub, "revision": rev}
