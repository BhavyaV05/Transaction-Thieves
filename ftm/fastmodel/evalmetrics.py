"""Per-decision calibration, reliability and agreement-with-the-large-model metrics for the fast model (C13).

Owner: T3 (Akshat).  Spec: wave 4 (docs/t3_system1.md).  Two views, as in TypeSafe's Jev evaluation plus ours: against the GROUND-TRUTH
label (multiclass Brier, log loss, top-label ECE, reliability bins) and against the LARGE model's distribution (mean total variation,
mean KL(large || fast), top-1 agreement).
"""

from __future__ import annotations

import numpy as np

from ftm.fastmodel.calibration import top_label_ece


def calibration_metrics(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    """probs: (n, k) calibrated distributions, y: (n,) true label indices."""
    p, y = np.asarray(probs, "f8"), np.asarray(y, int)
    n, k = p.shape
    onehot = np.eye(k)[y]
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    rel = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        rel.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "mean_conf": float(conf[m].mean()) if m.any() else None,
                    "accuracy": float((pred[m] == y[m]).mean()) if m.any() else None})  # fmt: skip
    return {"n": int(n), "accuracy": float((pred == y).mean()), "brier": float(((p - onehot) ** 2).sum(1).mean()),
            "log_loss": float(-np.log(np.clip(p[np.arange(n), y], 1e-12, 1)).mean()), "ece": top_label_ece(p, y), "reliability": rel}


def agreement(p_fast: np.ndarray, p_large: np.ndarray) -> dict:
    """How closely the fast distribution follows the large model's (rows paired)."""
    pf, pl = np.clip(np.asarray(p_fast, "f8"), 1e-12, 1), np.clip(np.asarray(p_large, "f8"), 1e-12, 1)
    return {"n": int(len(pf)), "mean_tv": float((0.5 * np.abs(pf - pl).sum(1)).mean()),
            "mean_kl_large_to_fast": float((pl * (np.log(pl) - np.log(pf))).sum(1).mean()),
            "top1_agreement": float((pf.argmax(1) == pl.argmax(1)).mean())}  # fmt: skip
