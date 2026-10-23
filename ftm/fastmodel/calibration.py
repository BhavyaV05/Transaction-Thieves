"""Temperature scaling for the fast model's label distribution (C13). Fit on validation only.

Owner: T3 (Akshat).  Spec: plan.md 6.14 ("constrained single-token outputs + temperature scaling").
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar


def softmax_t(logps: np.ndarray, t: float) -> np.ndarray:
    z = logps / t
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def nll(logps: np.ndarray, y: np.ndarray, t: float) -> float:
    p = softmax_t(logps, t)[np.arange(len(y)), y]
    return float(-np.mean(np.log(np.clip(p, 1e-12, 1.0))))


def fit_temperature(logps: np.ndarray, y: np.ndarray, bounds: tuple[float, float] = (0.05, 20.0)) -> float:
    """T minimising validation NLL. `logps`: (n, k) label log-probabilities, `y`: (n,) true label indices."""
    logps, y = np.asarray(logps, float), np.asarray(y, int)
    res = minimize_scalar(lambda lt: nll(logps, y, math.exp(lt)), bounds=tuple(math.log(b) for b in bounds),
                          method="bounded")  # fmt: skip
    return float(math.exp(res.x))


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error of the top label."""
    probs, y = np.asarray(probs, float), np.asarray(y, int)
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            err += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(err)
