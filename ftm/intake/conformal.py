"""C2: Mondrian (class-conditional) split conformal intake, time-weighted calibration, selective risk bound.

Owner: T1 (Abhishek).  Spec: plan.md 6.3, tracks.md 3.6.  Schemas and results: ftm/intake/README.md.

Binary problem, score p = p0 = P(launder | x). Non-conformity of label k is s_k(x) = 1 - P(k | x), so
  launder in set  <=>  p >= t_launder      (t_launder = 1 - q_1, q_1 = calibrated on laundering rows only)
  clean   in set  <=>  p <= t_clean        (t_clean   = q_0 = 1 - (1 - q_0'),  q_0' calibrated on clean rows only)
Per-class coverage 1 - alpha_k holds for exchangeable data (Vovk 2005; Mondrian taxonomy = the label).
Alert = launder is in the set, so alert <=> p >= t_launder; "both labels in set" is the uncertain part
of it (t_launder <= p <= t_clean), "launder only" is p > t_clean. Sets with neither label (empty) happen
when t_launder > t_clean and are not alerts.

Time weighting (drift, ref [39] of plan.md; weighted conformal prediction in the style of Tibshirani et
al. 2019): calibration row i gets weight 0.5 ** (age_days_i / half_life_days), age measured from the last
calibration day; the test point carries weight 1 (age 0). Plain split conformal is the `half_life=None` case.

Selective risk bound (ref [38] of plan.md names selective conformal risk control; this module ships the
simplest correct variant, a fixed-sequence Clopper-Pearson test as in Learn-then-Test, Angelopoulos et
al. 2021): the auto-cleared region is p < lam; lam is the largest value for which the one-sided
(1 - delta) upper confidence bound on P(laundering | cleared) is still <= beta, scanning lam upwards and
stopping at the first failure. Unweighted. The exact procedure of [38] is not implemented.
"""

from __future__ import annotations

import numpy as np


def time_weights(days, half_life_days: float | None) -> np.ndarray:
    """0.5 ** (age / half_life); age = last day - day. None means equal weights."""
    d = np.asarray(days, dtype="f8")
    if half_life_days is None:
        return np.ones(len(d))
    return 0.5 ** ((d.max() - d) / half_life_days)


def weighted_quantile_with_test(scores, weights, level: float) -> float:
    """Smallest s with weighted mass(scores <= s) / (sum(weights) + 1) >= level; +inf if none.

    The +1 is the test point (weight 1, score treated as +inf), the finite-sample correction of split
    conformal. With unit weights this is the ceil((n + 1) * level)-th smallest score.
    """
    s = np.asarray(scores, dtype="f8")
    w = np.asarray(weights, dtype="f8")
    if len(s) == 0:
        return float("inf")
    o = np.argsort(s, kind="stable")
    s, w = s[o], w[o]
    cum = np.cumsum(w) / (w.sum() + 1.0)
    # tolerance: with unit weights cum*(n+1) is an integer up to float error
    k = np.searchsorted(cum, level - 1e-12, side="left")
    return float(s[k]) if k < len(s) else float("inf")


def mondrian_thresholds(p, y, alpha_launder: float, alpha_clean: float, days=None, half_life_days=None):
    """(t_launder, t_clean) from calibration scores `p` and labels `y`; see the module docstring."""
    p, y = np.asarray(p, "f8"), np.asarray(y).astype(int)
    w = time_weights(days if days is not None else np.zeros(len(p)), half_life_days)
    pos, neg = y == 1, y == 0
    q1 = weighted_quantile_with_test(1 - p[pos], w[pos], 1 - alpha_launder)  # non-conformity of launder
    q0 = weighted_quantile_with_test(p[neg], w[neg], 1 - alpha_clean)  # non-conformity of clean = 1-(1-p)
    return 1.0 - q1, q0


def prediction_sets(p, t_launder: float, t_clean: float):
    """(has_launder, has_clean) boolean arrays. alert = has_launder."""
    p = np.asarray(p, "f8")
    return p >= t_launder, p <= t_clean


def achieved_miss_rates(p, y, t_launder: float, t_clean: float) -> dict:
    """Per-class miss rate = share of rows whose true label is NOT in the prediction set, with counts."""
    has_l, has_c = prediction_sets(p, t_launder, t_clean)
    y = np.asarray(y).astype(int)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    m1 = int(((y == 1) & ~has_l).sum())
    m0 = int(((y == 0) & ~has_c).sum())
    return {"launder_miss": m1 / n1 if n1 else float("nan"), "n_launder": n1,
            "clean_miss": m0 / n0 if n0 else float("nan"), "n_clean": n0}  # fmt: skip


def selective_clear_threshold(p, y, beta: float, delta: float = 0.1) -> float:
    """Largest lam such that auto-clearing {p < lam} keeps P(laundering | cleared) <= beta w.p. 1 - delta.

    Fixed-sequence Clopper-Pearson test over lam ascending (valid for iid calibration rows and unknown
    test row exchangeable with them). Returns 0.0 if even the smallest cleared set fails.
    """
    from scipy.stats import beta as beta_dist

    p, y = np.asarray(p, "f8"), np.asarray(y).astype(int)
    o = np.argsort(p, kind="stable")
    ps, ys = p[o], y[o]
    k = np.arange(1, len(ps) + 1)
    pos = np.cumsum(ys)
    ub = np.where(pos >= k, 1.0, beta_dist.ppf(1 - delta, pos + 1, np.maximum(k - pos, 1)))
    last_of_tie = np.r_[ps[1:] != ps[:-1], True]  # a cleared set must contain all rows with the same score
    # The sequence starts at k_min = 4 ln(1/delta)/beta cleared rows: a cleared set with zero laundering rows
    # first passes at ln(1/delta)/beta, and starting 4x later tolerates a few laundering rows among the
    # cleared ones. The start depends only on (beta, delta), not on the labels, so validity is kept.
    k_min = int(np.ceil(4 * np.log(1 / delta) / beta))
    ok = (ub <= beta) | (k < k_min)
    bad = np.flatnonzero(~ok & last_of_tie)
    first_bad = bad[0] if len(bad) else len(ps)
    good = np.flatnonzero(last_of_tie[:first_bad] & (k[:first_bad] >= k_min))
    if len(good) == 0:
        return 0.0
    i = good[-1]
    return float(ps[i + 1]) if i + 1 < len(ps) else float(np.nextafter(ps[i], np.inf))


def fixed_window(p, lo: float = 0.15, hi: float = 0.85):
    """Ablation: alert iff lo <= p <= hi (plan.md 6.3)."""
    p = np.asarray(p, "f8")
    return (p >= lo) & (p <= hi)


def selective_risk_upper_bound(p, y, t_alert: float, delta: float = 0.1) -> dict:
    """Clopper-Pearson (1 - delta) upper bound on P(laundering | auto-cleared = {p < t_alert}) from calibration rows."""
    from scipy.stats import beta as beta_dist

    p, y = np.asarray(p, "f8"), np.asarray(y).astype(int)
    c = p < t_alert
    k, pos = int(c.sum()), int(y[c].sum())
    ub = 1.0 if k == 0 or pos >= k else float(beta_dist.ppf(1 - delta, pos + 1, k - pos))
    return {"n_cleared": k, "n_laundering_cleared": pos, "upper_bound": ub, "delta": delta}
