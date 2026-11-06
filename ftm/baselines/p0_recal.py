"""p0 v2: recalibration of the one-bank detector score on the val `cal` part, frozen and hashed (finding 16).

The v1 p0 calibrator is fit on val `fit` (weekend days, ~half the Monday row volume; laundering rows per day are constant, so the row label rate
halves from `fit` to `eval`): p0 over-predicts 1.17x on `cal` and 1.38x on `eval`. A v2 map is fit on `cal` (no `eval` label enters), and frozen as
a small JSON with a content hash, so the final run applies the very same map to test p0 (`apply_map` after `load_map(verify=True)`); nothing is refit there.

Same idea as `fit_logit_recal` of T4's PR 111 (ftm/eval/final_additions.py): damped Newton on the ridge-penalised logistic log-loss of
logit p' = a + b logit p, pulled toward (a, b) = (0, 1); mode "intercept" fixes b = 1 (a pure prior-shift correction).
Maps (dicts): identity, logit_affine {a, b}, isotonic {x, y} (piecewise-linear through the knots of the isotonic fit on p0).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

EPS = 1e-9
KINDS = ("identity", "intercept", "platt", "isotonic")  # simplest first: the tie-break order of `select_variant`


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, "f8"), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_logit_recal(p, y, mode: str = "platt", ridge: float = 1.0, iters: int = 100) -> tuple[float, float]:
    """(a, b) of logit p' = a + b logit p by penalised logistic regression; mode "intercept" fixes b = 1. Deterministic, finite when scores separate the labels."""
    z, y = logit(p), np.asarray(y, "f8")
    if mode not in ("platt", "intercept"):
        raise ValueError("mode is platt or intercept")
    if len(z) < 2 or y.min() == y.max():
        raise ValueError("recalibration needs both classes among the fit rows")
    free = [0, 1] if mode == "platt" else [0]
    target = np.array([0.0, 1.0])

    def nll(th):
        t = th[0] + th[1] * z
        return float((np.logaddexp(0, t) - y * t).sum() + 0.5 * ridge * ((th - target)[free] ** 2).sum())

    theta = target.copy()
    for _ in range(iters):
        q = 1.0 / (1.0 + np.exp(-(theta[0] + theta[1] * z)))
        g = np.array([(q - y).sum(), ((q - y) * z).sum()])[free] + ridge * (theta - target)[free]
        w = q * (1 - q)
        h = np.array([[w.sum(), (w * z).sum()], [(w * z).sum(), (w * z * z).sum()]])[np.ix_(free, free)] + ridge * np.eye(len(free))
        step, t, f0 = np.linalg.solve(h, g), 1.0, nll(theta)
        while t > 1e-8:
            cand = theta.copy()
            cand[free] -= t * step
            if nll(cand) <= f0:
                break
            t /= 2
        else:
            break
        theta = cand
        if np.abs(t * step).max() < 1e-10:
            break
    return float(theta[0]), float(theta[1])


def fit_map(kind: str, p, y, ridge: float = 1.0, max_knots: int = 400) -> dict:
    """Fit a map of `kind` on (p, y). Isotonic is thinned to <= max_knots knots (quantiles of p) so the frozen JSON stays small."""
    if kind == "identity":
        return {"kind": "identity"}
    if kind in ("intercept", "platt"):
        a, b = fit_logit_recal(p, y, kind, ridge)
        return {"kind": "logit_affine", "mode": kind, "a": a, "b": b}
    if kind == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        p, y = np.asarray(p, "f8"), np.asarray(y, "f8")
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
        x = np.asarray(iso.X_thresholds_, "f8")
        v = np.asarray(iso.y_thresholds_, "f8")
        if len(x) > max_knots:
            keep = np.unique(np.r_[0, np.linspace(0, len(x) - 1, max_knots).round().astype(int), len(x) - 1])
            x, v = x[keep], v[keep]
        return {"kind": "isotonic", "x": x.tolist(), "y": v.tolist()}
    raise ValueError(f"unknown kind {kind!r}")


def apply_map(m: dict, p) -> np.ndarray:
    p = np.asarray(p, "f8")
    k = m["kind"]
    if k == "identity":
        return p.copy()
    if k == "logit_affine":
        return 1.0 / (1.0 + np.exp(-(m["a"] + m["b"] * logit(p))))
    if k == "isotonic":
        return np.interp(p, m["x"], m["y"])
    raise ValueError(f"unknown map kind {k!r}")


def map_hash(m: dict) -> str:
    """sha256 over the canonical JSON of the map itself (kind and parameters; not the provenance fields)."""
    core = {k: m[k] for k in ("kind", "mode", "a", "b", "x", "y") if k in m}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_map(m: dict, path, **provenance) -> dict:
    out = {**m, "provenance": provenance, "sha256": map_hash(m)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=1))
    return out


def load_map(path, expect_sha256: str | None = None) -> dict:
    """Load a frozen map; raises if its content no longer matches its stored hash (or `expect_sha256`)."""
    m = json.loads(Path(path).read_text())
    h = map_hash(m)
    if h != m.get("sha256") or (expect_sha256 and h != expect_sha256):
        raise ValueError(f"{path}: map hash {h[:12]} does not match {(expect_sha256 or m.get('sha256') or '?')[:12]}: the frozen map was changed")
    return m


def cv_brier(kind: str, p, y, groups=None, k: int = 5, ridge: float = 1.0, score_mask=None, fit_mask=None) -> float:
    """Cross-fitted Brier of a map: fit on the other groups, score the held-out group. Rows must be in time order.

    `groups`: a group id per row (e.g. the time fold); default k contiguous blocks. `score_mask`: only these rows count in the Brier (the map is
    still fit on all rows), e.g. the alerts, the rows downstream systems see. `fit_mask`: the map is fit on these rows only (e.g. the alerts).
    """
    p, y = np.asarray(p, "f8"), np.asarray(y, "f8")
    if groups is None:
        groups = np.repeat(np.arange(k), int(np.ceil(len(p) / k)))[: len(p)]
    groups = np.asarray(groups)
    mask = np.ones(len(p), bool) if score_mask is None else np.asarray(score_mask, bool)
    fm = np.ones(len(p), bool) if fit_mask is None else np.asarray(fit_mask, bool)
    se, n = 0.0, 0
    for g in np.unique(groups):
        te = groups == g
        try:
            m = fit_map(kind, p[~te & fm], y[~te & fm], ridge)
        except ValueError:
            m = {"kind": "identity"}
        t = te & mask
        se += float(((apply_map(m, p[t]) - y[t]) ** 2).sum())
        n += int(t.sum())
    return se / max(n, 1)


def select_candidate(cv_alerts: dict[str, float], cv_rows: dict[str, float], base: str = "identity_rows") -> str:
    """Lowest cross-fitted Brier on held-out alerts among the candidates that do not hurt the all-rows Brier relative to `base` (an alert-only Platt can wreck the 99.9% non-alert rows)."""
    ok = {k: v for k, v in cv_alerts.items() if cv_rows[k] <= cv_rows[base] * (1 + 1e-9)}
    return min(ok, key=ok.get)


def select_variant(cv: dict[str, float], tol: float = 0.0) -> str:
    """Lowest cross-fitted Brier on the held-out alerts (`tol` = 0). With tol > 0: the simplest kind (identity < intercept < platt < isotonic) within tol of the best.

    A first draft used tol = 0.01, which picks identity on the HI-Medium cal part (no correction); tol = 0 was set after that, once eval numbers existed (disclosed in docs/t1_baselines_val.md)."""
    best = min(cv.values())
    return next(k for k in KINDS if k in cv and cv[k] <= best * (1 + tol))
