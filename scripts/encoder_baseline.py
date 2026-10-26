"""ModernBERT / DeBERTa classifier baselines and RouteLLM-style routing on the small-decision set (E10, week 10, secondary).

Owner: T2 (Bhavya).  Spec: tracks.md 4.9, plan.md 7.3 and 8.  Run from the repo root. Needs torch + transformers for `train` and
`score`; `tfidf` needs scikit-learn; `route` needs neither.

    # 1. fine-tune one classifier per decision on the TRAIN rows (4 GB GPU: ModernBERT-base, bf16, max length 256)
    python scripts/encoder_baseline.py train --model modernbert-base --out outputs/fast/modernbert
    # 2. score the val rows; the output has T3's row format, so scripts/score_fast_baseline.py report takes it as --fast
    python scripts/encoder_baseline.py score --model-dir outputs/fast/modernbert --out outputs/fast/fast_modernbert.jsonl
    #    (the floor, no GPU: TF-IDF + logistic regression, trained and scored in one go)
    python scripts/encoder_baseline.py tfidf --out outputs/fast/fast_tfidf.jsonl
    # 3. routers against T3's conformal cascade, given the 8B's scores from score_fast_baseline.py
    python scripts/encoder_baseline.py route --fast outputs/fast/fast_modernbert.jsonl --large outputs/fast/large_8b_json.jsonl \\
        --out outputs/fast/route_modernbert.json

`--decisions-train` / `--decisions` default to `data/decisions/hi_medium/decisions_{train,val}.jsonl` (hand-off #13) and `--alerts` to
the val alert set that carries `val_part`. Train rows come from the train split only, val rows are joined to `fit`/`cal`/`eval` by
alert id (`ftm.fastmodel.splits`); the test split is never read. `--cap N` keeps at most N val rows per (decision, part) with the
same seed as `score_fast_baseline.py`, so every tier scores the same rows.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ftm.fastmodel.decisions import DECISIONS, DecisionExample  # noqa: E402
from ftm.fastmodel.encoder import ENCODERS, EncoderConfig, config_dict  # noqa: E402
from ftm.fastmodel.router import route_report  # noqa: E402

DEFAULT_TRAIN = ROOT / "data" / "decisions" / "hi_medium" / "decisions_train.jsonl"
DEFAULT_VAL = ROOT / "data" / "decisions" / "hi_medium" / "decisions_val.jsonl"
DEFAULT_ALERTS = ROOT / "data" / "alerts" / "hi_medium" / "val.parquet"


def _t3():
    """T3's row loader, so a classifier scores exactly the rows the LLM tiers score."""
    spec = importlib.util.spec_from_file_location("score_fast_baseline", ROOT / "scripts" / "score_fast_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_train(path: Path) -> list[DecisionExample]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if {r["split"] for r in rows} != {"train"}:
        raise SystemExit(f"{path}: this command trains on the train split only (found {sorted({r['split'] for r in rows})})")
    return [DecisionExample(r["decision"], r["context"], r["label"], r.get("alert_id"), r["split"]) for r in rows]


def write_scores(rows, model, out: Path) -> dict:
    """{"i": row index, "logps": label log-probabilities} per val row: the format of `score_fast_baseline.py score --tier fast`."""
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["decision"], []).append(r)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0, n = time.perf_counter(), 0
    with out.open("w", encoding="utf-8") as f:
        for decision, rs in sorted(by.items()):
            if decision not in model.parts:
                print(f"skipping {len(rs)} {decision} rows: no classifier for it", file=sys.stderr)
                continue
            lp = model.logps_batch(decision, [r["context"] for r in rs])
            for r, x in zip(rs, lp, strict=True):
                f.write(json.dumps({"i": r["i"], "logps": x.tolist()}) + "\n")
            n += len(rs)
    dt = time.perf_counter() - t0
    return {"rows_scored": n, "seconds": round(dt, 1), "rows_per_s": round(n / dt, 1) if dt else None}


def cmd_train(a) -> int:
    from ftm.fastmodel.encoder import EncoderDecisionModel, train_decision_encoder

    cfg = EncoderConfig(model=a.model, max_len=a.max_len, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, seed=a.seed,
                        grad_checkpointing=a.grad_checkpointing, max_rows=a.max_rows)  # fmt: skip
    rows = read_train(a.decisions_train)
    names = a.only or sorted({r.decision for r in rows})
    parts, history = {}, {}
    for d in names:
        t0 = time.perf_counter()
        tok, model, hist = train_decision_encoder(rows, d, cfg, log=print)
        parts[d], history[d] = (tok, model), {"epochs": hist, "seconds": round(time.perf_counter() - t0, 1)}
    meta = {"config": config_dict(cfg), "history": history, "train_file": str(a.decisions_train), "labels": {d: list(DECISIONS[d].labels) for d in names}}
    EncoderDecisionModel(parts, a.max_len).save(a.out, meta)
    print(json.dumps({"saved": str(a.out), **{d: h["epochs"][-1] for d, h in history.items()}}, indent=1))
    return 0


def cmd_score(a) -> int:
    from ftm.fastmodel.encoder import EncoderDecisionModel

    model = EncoderDecisionModel.load(a.model_dir, batch_size=a.batch_size)
    rows = _t3().load_rows(a.decisions, a.alerts, a.cap, a.seed)
    print(json.dumps({"model_dir": str(a.model_dir), **write_scores(rows, model, a.out)}))
    return 0


def cmd_tfidf(a) -> int:
    from ftm.fastmodel.encoder import TfidfDecisionModel

    model = TfidfDecisionModel(seed=a.seed).fit(read_train(a.decisions_train))
    rows = _t3().load_rows(a.decisions, a.alerts, a.cap, a.seed)
    print(json.dumps({"model": "tfidf-logreg", **write_scores(rows, model, a.out)}))
    return 0


def cmd_route(a) -> int:
    t3 = _t3()
    rows = t3.load_rows(a.decisions, a.alerts, a.cap, a.seed)
    fast = {json.loads(x)["i"]: json.loads(x) for x in a.fast.read_text().splitlines() if x.strip()}
    large = {json.loads(x)["i"]: json.loads(x) for x in a.large.read_text().splitlines() if x.strip()}
    from_logps = a.large_kind == "logprob"
    rows = [r for r in rows if r["i"] in fast and (r["val_part"] == "fit" or r["i"] in large)]  # same rule as T3's report
    exs = [t3.example(r) for r in rows]
    parts = [r["val_part"] for r in rows]
    if not any(p == "eval" for p in parts):
        raise SystemExit("no eval rows: score the fast and the large tier on the same rows first")

    def large_probs(r):
        if r["i"] not in large:
            return None
        v = large[r["i"]]
        if not from_logps:
            return v["probs"]
        from ftm.fastmodel.calibration import softmax_t

        p = softmax_t(np.array(v["logps"])[None, :], 1.0)[0]
        return dict(zip(DECISIONS[exs[rows.index(r)].decision].labels, p.tolist(), strict=True))

    rep = route_report(exs, parts, [np.array(fast[r["i"]]["logps"]) for r in rows], [large_probs(r) for r in rows], a.alpha)
    rep["selection"] = {"fast": str(a.fast), "large": str(a.large), "large_kind": a.large_kind, "cap": a.cap, "seed": a.seed}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    c = rep["conformal"]
    print(json.dumps({"rows": rep["rows"], "fast_only_error": rep["fast_only_error"], "large_only_error": rep["large_only_error"],
                      "conformal_unsettled_cal": c["unsettled_share_cal"], "conformal_error_default_as_answer": c["error_default_as_answer"],
                      "routers_matched": {k: v["matched_to_conformal"] for k, v in rep["routers"].items()}}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, train=False, val=False):
        if train:
            p.add_argument("--decisions-train", type=Path, default=DEFAULT_TRAIN)
        if val:
            p.add_argument("--decisions", type=Path, default=DEFAULT_VAL)
            p.add_argument("--alerts", type=Path, default=DEFAULT_ALERTS)
            p.add_argument("--cap", type=int, default=None, help="max val rows per (decision, part); same rows as score_fast_baseline.py")
        p.add_argument("--seed", type=int, default=0)

    t = sub.add_parser("train")
    t.add_argument("--model", choices=sorted(ENCODERS), default="modernbert-base")
    t.add_argument("--only", nargs="+", choices=sorted(DECISIONS), default=None, help="decisions to train (default: all in the file)")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--max-len", type=int, default=256)
    t.add_argument("--max-rows", type=int, default=None, help="train rows per decision (random draw)")
    t.add_argument("--grad-checkpointing", action="store_true", help="trade speed for VRAM on the 4 GB card")
    t.add_argument("--out", type=Path, required=True)
    common(t, train=True)
    s = sub.add_parser("score")
    s.add_argument("--model-dir", type=Path, required=True)
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--out", type=Path, required=True)
    common(s, val=True)
    f = sub.add_parser("tfidf")
    f.add_argument("--out", type=Path, required=True)
    common(f, train=True, val=True)
    r = sub.add_parser("route")
    r.add_argument("--fast", type=Path, required=True)
    r.add_argument("--large", type=Path, required=True)
    r.add_argument("--large-kind", choices=["json", "logprob"], default="json")
    r.add_argument("--alpha", type=float, default=0.05)
    r.add_argument("--out", type=Path, required=True)
    common(r, val=True)
    a = ap.parse_args(argv)
    return {"train": cmd_train, "score": cmd_score, "tfidf": cmd_tfidf, "route": cmd_route}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
