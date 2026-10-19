"""Point metrics for plan.md 9.1-9.2: Brier, gap closed, PR-AUC, balanced accuracy, catch at fixed capacity,
flipped calls, calibration error. numpy only, so CI (base install) can run every metric.

Owner: T4 (Shreyas).  Spec: plan.md 9; tracks.md 6.7.
Confidence intervals for any of these come from ftm.eval.bootstrap (cluster bootstrap), never from row noise.
"""

from __future__ import annotations

import numpy as np

# Below this, Brier(one bank) - Brier(upper limit) is "tiny" and gap closed is not reported (plan.md 9.1).
# Fixed here, before any result is seen; prereg/ may tighten it.
MIN_GAP_DENOMINATOR = 1e-5


def _yp(y, p):
    y, p = np.asarray(y, dtype="f8"), np.asarray(p, dtype="f8")
    if y.shape != p.shape:
        raise ValueError(f"y and p differ in shape: {y.shape} vs {p.shape}")
    if np.isnan(p).any():
        raise ValueError("p contains NaN; drop or impute unscored rows before computing metrics")
    return y, p


def brier(y, p) -> float:
    """Mean squared error of probability p against the 0/1 label y. Lower is better."""
    y, p = _yp(y, p)
    return float(np.mean((p - y) ** 2))


def brier_skill(y, p) -> float:
    """1 - Brier(p) / Brier(constant base rate). 0 = no better than the base rate."""
    y, p = _yp(y, p)
    ref = brier(y, np.full_like(y, y.mean()))
    return float(1 - brier(y, p) / ref) if ref > 0 else float("nan")


def gap_closed(b_one_bank: float, b_system: float, b_upper: float,
               min_denominator: float = MIN_GAP_DENOMINATOR) -> float:  # fmt: skip
    """[B(one bank) - B(system)] / [B(one bank) - B(upper limit)]. 0 = one bank, 1 = see everything.

    NaN when the denominator is below `min_denominator`: then report raw Brier differences instead
    (plan.md 9.1). Report it against both upper limits (GFP see-everything and the best SOTA model).
    """
    den = b_one_bank - b_upper
    if den < min_denominator:
        return float("nan")
    return float((b_one_bank - b_system) / den)


def pr_auc(y, p) -> float:
    """Average precision (step-wise area under the precision-recall curve), as sklearn computes it."""
    y, p = _yp(y, p)
    n_pos = y.sum()
    if n_pos == 0:
        return float("nan")
    o = np.argsort(-p, kind="stable")
    y, p = y[o], p[o]
    last = np.r_[np.flatnonzero(np.diff(p) != 0), len(p) - 1]  # last index of each tied-score block
    tp = np.cumsum(y)[last]
    precision = tp / (last + 1)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def balanced_accuracy(y, p, threshold: float) -> float:
    """Mean of the true positive rate and the true negative rate at `threshold` (p >= threshold = launder)."""
    y, p = _yp(y, p)
    pred = p >= threshold
    pos, neg = y == 1, y == 0
    tpr = pred[pos].mean() if pos.any() else float("nan")
    tnr = (~pred[neg]).mean() if neg.any() else float("nan")
    return float((tpr + tnr) / 2)


def caught_at_capacity(y, p, day, per_day: int) -> dict:
    """Investigators review the `per_day` highest-p alerts of each day. Returns launderers caught and recall.

    Ties at the cut are broken by row order (stable), so the reviewed count is exactly min(per_day, n_day).
    """
    y, p = _yp(y, p)
    day = np.asarray(day)
    if day.shape != y.shape:
        raise ValueError("day must have one entry per alert")
    reviewed = np.zeros(len(y), dtype=bool)
    for d in np.unique(day):
        idx = np.flatnonzero(day == d)
        reviewed[idx[np.argsort(-p[idx], kind="stable")[:per_day]]] = True
    caught = int(y[reviewed].sum())
    n_pos = int(y.sum())
    return {"reviewed": int(reviewed.sum()), "caught": caught, "positives": n_pos,
            "recall": caught / n_pos if n_pos else float("nan")}  # fmt: skip


def flips(y, p_before, p_after, threshold: float) -> dict:
    """Calls changed by the system vs the starting score: right-to-wrong and wrong-to-right counts."""
    y, pb = _yp(y, p_before)
    _, pa = _yp(y, p_after)
    right_b = (pb >= threshold) == (y == 1)
    right_a = (pa >= threshold) == (y == 1)
    return {"right_to_wrong": int((right_b & ~right_a).sum()), "wrong_to_right": int((~right_b & right_a).sum()),
            "n": int(len(y))}  # fmt: skip


def ece(y, p, n_bins: int = 15) -> float:
    """Expected calibration error with equal-mass bins: sum over bins of (bin share) * |mean p - mean y|.

    Equal-mass rather than equal-width bins, because laundering is rare and almost every p is near 0.
    """
    y, p = _yp(y, p)
    if len(y) == 0:
        return float("nan")
    o = np.argsort(p, kind="stable")
    total = 0.0
    for b in np.array_split(o, min(n_bins, len(y))):
        total += len(b) * abs(p[b].mean() - y[b].mean())
    return float(total / len(y))


def end_to_end_probability(probability, p0) -> tuple[np.ndarray, float]:
    """Fill missing verdicts (an agent that never gave a probability, e.g. ReAct without finish) with p0.

    Rule agreed with T3 (PR #7): no verdict = the agent changed nothing. Returns the filled probabilities and the
    no-verdict rate, which is reported next to every end-to-end score so abstaining cannot look calibrated.
    """
    p0 = np.asarray(p0, dtype="f8")
    p = np.array([np.nan if v is None else v for v in probability], dtype="f8")
    if p.shape != p0.shape:
        raise ValueError("probability and p0 need one entry per alert")
    missing = np.isnan(p)
    return np.where(missing, p0, p), float(missing.mean()) if len(p) else float("nan")


def detection_summary(y, p, threshold: float | None = None) -> dict:
    """The detection row of plan.md 9.2 for one system. Threshold-free unless `threshold` is given."""
    out = {"n": int(len(np.asarray(y))), "n_pos": int(np.asarray(y).sum()), "brier": brier(y, p),
           "brier_skill": brier_skill(y, p), "pr_auc": pr_auc(y, p), "ece": ece(y, p)}  # fmt: skip
    if threshold is not None:
        out["balanced_accuracy"] = balanced_accuracy(y, p, threshold)
    return out
