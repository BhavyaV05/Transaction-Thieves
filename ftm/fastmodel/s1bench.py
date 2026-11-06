"""S1BENCH core: typed decisions over arbitrary option sets, the shared metric set, option-order permutations, selection helpers.

Owner: T3 (Akshat).  Spec: docs/t3_s1bench.md (protocol and selection rule, written before any run).

Everything here is model-agnostic: a *candidate* is a callable `probs(task, item) -> {label: probability}` (System 1 through
`ftm.fastmodel.typed.TypedDecider.decide`, or an open System One model through `ftm.fastmodel.s1bench_adapters`).  The module
adds what the benchmark needs on top of the typed API:

  * `TaskSpec` / `Item`: a typed question (text, question, labels with optional descriptions) and its gold label;
  * `registered(...)`: put a `Decision` for one question into `DECISIONS` for the duration of a call (the typed API looks decisions up by name);
  * `make_decision`: yes/no tasks use the word tokens the adapters were trained on, every other task the letters A-Z (labels shown as
    "A = label"), more than 26 labels use `typed.py`'s two-stage scorer;
  * `ListedTwoStage`: benchmark-only variant of the two-stage scorer whose stage 1 lists every label of every group (the production
    stage 1 shows only "first .. last", which cannot separate the labels in between);
  * the metrics (multiclass Brier, log loss, equal-mass ECE, accuracy, macro-F1, F1/AUROC, option-order bias) and the paired cluster
    bootstrap + selection rule of section 4 of the doc.
"""

from __future__ import annotations

import contextlib
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ftm.fastmodel.decisions import DECISIONS, Decision
from ftm.fastmodel.model import FLOOR_LOGP
from ftm.fastmodel.typed import LETTERS, TypedDecider, _ask, group_split

_YN = ("yes", "no")
PERM_SEEDS = (11, 12, 13)  # option-order permutations (doc section 2.3); the lettered canonical order is permutation 0
ECE_BINS = 15


# ---------------------------------------------------------------------------------------------------------------- tasks
@dataclass(frozen=True)
class Item:
    id: str
    text: str
    label: int  # index into the task's labels
    cluster: str | None = None  # bootstrap cluster (default: the item id)


@dataclass(frozen=True)
class TaskSpec:
    """A single-label typed task: `labels` are what the model reads (names or definitions), `question` the instruction."""

    name: str
    question: str
    labels: tuple[str, ...]
    hint: str = ""

    @property
    def binary_yes_no(self) -> bool:
        return tuple(self.labels) == _YN


def _unique(labels: Sequence[str]) -> tuple[str, ...]:
    """Display strings must be distinct (the model reads them); a repeated one gets an index suffix."""
    seen: dict[str, int] = {}
    out = []
    for lb in labels:
        seen[lb] = seen.get(lb, 0) + 1
        out.append(lb if seen[lb] == 1 else f"{lb} ({seen[lb]})")
    return tuple(out)


def make_decision(name: str, question: str, labels: Sequence[str], *, hint: str = "", lettered: bool | None = None) -> Decision:
    """Typed decision over `labels` (in the order shown).  yes/no in canonical order -> the word tokens; otherwise letters.
    More than 26 labels: letters are placeholders only (two-stage scoring never reads them)."""
    labels = _unique(labels)
    if lettered is None:
        lettered = labels != _YN
    if not lettered:
        return Decision(name, question, labels, _YN, None, hint)
    tokens = tuple(LETTERS[i] if i < len(LETTERS) else f"#{i}" for i in range(len(labels)))
    return Decision(name, question, labels, tokens, None, hint)


@contextlib.contextmanager
def registered(*decisions: Decision):
    """Insert decisions into the global `DECISIONS` for the duration of the block (the typed API looks them up by name)."""
    for d in decisions:
        DECISIONS[d.name] = d
    try:
        yield
    finally:
        for d in decisions:
            DECISIONS.pop(d.name, None)


class ListedTwoStage(TypedDecider):
    """Two-stage scorer whose stage 1 lists the labels of each group (benchmark-only variant, doc section 2.4).  Same maths as
    `TypedDecider._two_stage`: p(label) = p(group) * p(label | group), one extra pass per group."""

    def _two_stage(self, d, text: str) -> np.ndarray:
        from ftm.fastmodel.decisions import DecisionExample, messages

        groups = group_split(len(d.labels), self.cap)
        base = messages(DecisionExample(d.name, text))
        opts = ["; ".join(d.labels[i] for i in idx) for idx in groups]
        glogp = self._read(_ask(base, d.question, [f"{LETTERS[g]} = one of: {o}" for g, o in enumerate(opts)]), LETTERS[: len(groups)])
        glogp = glogp - np.logaddexp.reduce(glogp)
        out = np.full(len(d.labels), FLOOR_LOGP)
        for g, idx in enumerate(groups):
            inner = self._read(_ask(base, d.question, [f"{LETTERS[j]} = {d.labels[i]}" for j, i in enumerate(idx)]), LETTERS[: len(idx)])
            inner = inner - np.logaddexp.reduce(inner)
            for j, i in enumerate(idx):
                out[i] = glogp[g] + inner[j]
        return out


def permutations(k: int, seeds: Sequence[int] = PERM_SEEDS) -> list[list[int]]:
    """Identity first, then one random NON-identity permutation of range(k) per seed (k = 2 -> always the swap)."""
    out = [list(range(k))]
    for s in seeds:
        rng = random.Random(s)
        while True:
            p = list(range(k))
            rng.shuffle(p)
            if p != list(range(k)) or k < 2:
                break
        out.append(p)
    return out


def realign(probs: Sequence[float], perm: Sequence[int]) -> np.ndarray:
    """Probabilities over the SHOWN order (shown position j holds canonical label perm[j]) -> canonical order."""
    out = np.zeros(len(perm))
    for j, c in enumerate(perm):
        out[c] = probs[j]
    return out


# -------------------------------------------------------------------------------------------------------------- metrics
def _arr(P, y):
    P = np.asarray(P, float)
    y = np.asarray(y, int)
    if P.ndim != 2 or len(P) != len(y):
        raise ValueError("P must be (n, k) and y (n,)")
    return P, y


def brier_multiclass(P, y) -> float:
    """Mean over items of sum_k (p_k - 1[y = k])^2."""
    P, y = _arr(P, y)
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((P - onehot) ** 2, axis=1)))


def log_loss(P, y, eps: float = 1e-12) -> float:
    P, y = _arr(P, y)
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], eps, 1.0))))


def ece_equal_mass(P, y, bins: int = ECE_BINS) -> float:
    """Top-label ECE with bins of (nearly) equal item count: sort by confidence, split into `bins` chunks."""
    P, y = _arr(P, y)
    if len(y) == 0:
        return float("nan")
    return ece_from(P.max(axis=1), (P.argmax(axis=1) == y).astype(float), bins)


def ece_from(conf, correct, bins: int = ECE_BINS) -> float:
    """Equal-mass ECE from a confidence and a (possibly soft, in [0, 1]) correctness per item."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    order = np.argsort(conf, kind="stable")
    err = 0.0
    for chunk in np.array_split(order, min(bins, len(conf))):
        if len(chunk):
            err += len(chunk) / len(conf) * abs(correct[chunk].mean() - conf[chunk].mean())
    return float(err)


def accuracy(P, y) -> float:
    P, y = _arr(P, y)
    return float(np.mean(P.argmax(axis=1) == y)) if len(y) else float("nan")


def macro_f1(P, y) -> float:
    P, y = _arr(P, y)
    pred = P.argmax(axis=1)
    f = []
    for k in sorted(set(y.tolist())):  # classes present in the gold labels (sklearn's labels=unique(y) behaviour)
        tp = np.sum((pred == k) & (y == k))
        fp = np.sum((pred == k) & (y != k))
        fn = np.sum((pred != k) & (y == k))
        f.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)) if f else float("nan")


def binary_prf(pred, y) -> dict:
    pred, y = np.asarray(pred, int), np.asarray(y, int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


def auroc(score, y) -> float | None:
    """Rank-based AUROC with average ranks for ties; None when one class is absent."""
    score, y = np.asarray(score, float), np.asarray(y, int)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_s = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tv_distance(p, q) -> float:
    return 0.5 * float(np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())


def core_metrics(P, y, labels: Sequence[str] | None = None) -> dict:
    """The metric set every task reports: accuracy, macro-F1, Brier, log loss, ECE (equal mass), majority baseline, prediction shares."""
    P, y = _arr(P, y)
    k = P.shape[1]
    pred = P.argmax(axis=1)
    counts = np.bincount(y, minlength=k)
    maj = int(counts.argmax())
    out = {
        "n": int(len(y)),
        "accuracy": accuracy(P, y),
        "macro_f1": macro_f1(P, y),
        "brier": brier_multiclass(P, y),
        "log_loss": log_loss(P, y),
        "ece": ece_equal_mass(P, y),
        "majority_label": maj if labels is None else labels[maj],
        "majority_accuracy": float(counts[maj] / len(y)) if len(y) else float("nan"),
        "pred_distinct": int(len(set(pred.tolist()))),
        "pred_majority_share": float(np.mean(pred == maj)) if len(y) else float("nan"),
    }
    prior = counts / max(1, counts.sum())
    out["brier_prior"] = brier_multiclass(np.tile(prior, (len(y), 1)), y) if len(y) else float("nan")
    return out


def order_bias(P_canon, P_perm_realigned: Sequence, y) -> dict:
    """P_canon: lettered identity-order distributions (n, k); P_perm_realigned: one (n, k) array per random permutation, already
    realigned to the canonical label order.  Reports accuracy per arm, the accuracy range, and the mean TV to the canonical arm."""
    arms = [np.asarray(P_canon, float), *[np.asarray(p, float) for p in P_perm_realigned]]
    accs = [accuracy(p, y) for p in arms]
    tvs = [float(np.mean(0.5 * np.abs(arms[0] - p).sum(axis=1))) for p in arms[1:]]
    flips = [float(np.mean(arms[0].argmax(1) != p.argmax(1))) for p in arms[1:]]
    return {"accuracy_by_arm": accs, "accuracy_range": max(accs) - min(accs), "mean_tv": float(np.mean(tvs)) if tvs else 0.0,
            "tv_by_perm": tvs, "argmax_flip_rate": float(np.mean(flips)) if flips else 0.0}  # fmt: skip


def latency_summary(seconds: Sequence[float]) -> dict:
    s = np.asarray(seconds, float)
    if not len(s):
        return {"n": 0, "median_s": None, "p90_s": None, "mean_s": None}
    return {"n": int(len(s)), "median_s": float(np.median(s)), "p90_s": float(np.quantile(s, 0.9)), "mean_s": float(s.mean())}


# ----------------------------------------------------------------------------------------------- paired cluster bootstrap
def paired_cluster_bootstrap(per_row_a: Sequence[float], per_row_b: Sequence[float], clusters: Sequence, *, n_boot: int = 2000, seed: int = 0,
                             alpha: float = 0.05) -> dict:  # fmt: skip
    """Paired difference of the mean of per-row losses (a - b), resampling whole clusters with replacement.  Percentile CI.
    Same fast path as ftm.eval.bootstrap: one multinomial draw of cluster weights, then weights @ sums."""
    a, b = np.asarray(per_row_a, float), np.asarray(per_row_b, float)
    if a.shape != b.shape or len(a) != len(clusters):
        raise ValueError("per-row arrays and clusters must align")
    uniq, inv = np.unique(np.asarray([str(c) for c in clusters]), return_inverse=True)
    d_sum = np.bincount(inv, weights=a - b, minlength=len(uniq))
    cnt = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=n_boot).astype(float)
    num, den = w @ d_sum, w @ cnt
    diffs = num / den
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"diff": float((a - b).mean()), "ci_lo": float(lo), "ci_hi": float(hi), "n_clusters": int(len(uniq)),
            "p_two_sided": float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))}  # fmt: skip


def per_row_brier(P, y) -> np.ndarray:
    P, y = _arr(P, y)
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0
    return np.sum((P - onehot) ** 2, axis=1)


# ------------------------------------------------------------------------------------------------------- collapse checks
def collapse_flags(P, y, P_fit_prior: Sequence[float], fit_majority: int, margin: float = 0.01) -> dict:
    """Doc section 4.1.  `P_fit_prior` / `fit_majority` come from the FIT rows only.
    constant: one predicted label for every row; no_skill: accuracy <= majority-baseline accuracy + margin AND Brier not below the
    prior's; majority_only: >= 99% of predictions are the majority label while the truth's majority share is < 0.9."""
    P, y = _arr(P, y)
    pred = P.argmax(axis=1)
    maj_acc = float(np.mean(y == fit_majority))
    acc = accuracy(P, y)
    prior_b = brier_multiclass(np.tile(np.asarray(P_fit_prior, float), (len(y), 1)), y)
    constant = len(set(pred.tolist())) == 1
    no_skill = acc <= maj_acc + margin and brier_multiclass(P, y) >= prior_b
    majority_only = bool(np.mean(pred == fit_majority) >= 0.99 and maj_acc < 0.9)
    return {"constant": constant, "no_skill": bool(no_skill), "majority_only": majority_only,
            "collapsed": bool(constant or no_skill or majority_only), "accuracy": acc, "majority_accuracy": maj_acc,
            "brier": brier_multiclass(P, y), "brier_prior": prior_b}  # fmt: skip


# ------------------------------------------------------------------------------------------------------ selection rule
@dataclass
class Candidate:
    name: str
    params_m: float  # millions of parameters (ties -> smaller)
    latency_s: float  # median seconds per decision
    brier_by_decision: dict[str, np.ndarray] = field(default_factory=dict)  # per-row Brier, same row order for every candidate
    ece_by_decision: dict[str, float] = field(default_factory=dict)
    collapsed: dict[str, bool] = field(default_factory=dict)


def select_candidate(cands: Sequence[Candidate], clusters: dict[str, Sequence], *, n_boot: int = 2000, seed: int = 0,
                     ece_margin: float = 0.005) -> dict:  # fmt: skip
    """Doc section 4.  Eligible = collapsed on no decision.  Order by macro Brier (equal weight over decisions); S = lowest.  Tied set =
    candidates whose paired cluster-bootstrap 95% CI of (macro Brier - S's) contains 0.  Within the tied set: lowest macro ECE
    (within `ece_margin` is a tie), then lowest latency, then the smaller model.  Fit+cal rows only: the caller passes no eval row."""
    decisions = sorted(set.intersection(*[set(c.brier_by_decision) for c in cands])) if cands else []
    eligible = [c for c in cands if not any(c.collapsed.get(d, False) for d in decisions)]
    report = {"decisions": decisions, "ineligible": [c.name for c in cands if c not in eligible], "ranked": [], "tied": [], "selected": None}
    if not eligible:
        return report

    def macro(c: Candidate) -> float:
        return float(np.mean([c.brier_by_decision[d].mean() for d in decisions]))

    ranked = sorted(eligible, key=macro)
    best = ranked[0]
    report["ranked"] = [{"name": c.name, "macro_brier": macro(c), "macro_ece": float(np.mean([c.ece_by_decision[d] for d in decisions])),
                         "latency_s": c.latency_s, "params_m": c.params_m} for c in ranked]  # fmt: skip

    n_tot = sum(len(clusters[d]) for d in decisions)

    def macro_rows(c: Candidate, d_rows: dict[str, np.ndarray]) -> np.ndarray:
        """Per-row losses scaled so that the pooled row mean equals the equal-weight mean over decisions of the per-decision means."""
        return np.concatenate([d_rows[d] * n_tot / (len(d_rows[d]) * len(decisions)) for d in decisions])

    cl = np.concatenate([np.asarray([str(x) for x in clusters[d]]) for d in decisions])
    tied = [best]
    ci = {}
    for c in ranked[1:]:
        r = paired_cluster_bootstrap(macro_rows(c, c.brier_by_decision), macro_rows(best, best.brier_by_decision), cl, n_boot=n_boot, seed=seed)
        ci[c.name] = r
        if r["ci_lo"] <= 0:
            tied.append(c)
    report["ci_vs_best"] = ci
    report["tied"] = [c.name for c in tied]
    ece = {c.name: float(np.mean([c.ece_by_decision[d] for d in decisions])) for c in tied}
    lo = min(ece.values())
    tied2 = [c for c in tied if ece[c.name] <= lo + ece_margin]
    fastest = min(c.latency_s for c in tied2)
    tied3 = [c for c in tied2 if c.latency_s <= fastest * 1.0000001]
    chosen = min(tied3, key=lambda c: (c.params_m, c.name))
    report["selected"] = chosen.name
    report["steps"] = {"tied_on_brier": report["tied"], "after_ece": [c.name for c in tied2], "after_latency": [c.name for c in tied3]}
    return report


def sigmoid_safe(x: float) -> float:
    return 1 / (1 + math.exp(-max(min(x, 50), -50)))


Prober = Callable[[TaskSpec, Item], dict[str, float]]
