"""C14 decision and stopping, core: one calibrated threshold on the fused posterior, plus the stop rule.

Owner: T4 (Shreyas).  Spec: plan.md 6.15, tracks.md 6.5.  FCAC three-way (stretch) is not here.

Decision: escalate iff posterior >= threshold. The posterior is calibrated (fusion, C8), so the cost-optimal
threshold is c_fp / (c_fp + c_fn); `fit_threshold` checks it on validation and also supports the daily review
capacity view of plan.md 9.2. The threshold is fitted on validation alerts that went through the same earlier
steps (intake, fusion) as the alerts it will decide (plan.md 6.15), and fixed in prereg/ before test.

Stop rule, checked after every answer:
  budget        no eps for the smallest question, no LLM calls, or no simulated time left
  fast_clear    the fast model (or the 8B) says the case is clear (C13)
  cannot_cross  no remaining question can move the posterior across the threshold. A DP answer bought with eps
                shifts log-odds by at most eps * delta (plan.md 6.6 ceiling: delta = 1 for yes/no, the typical
                launderer-vs-clean gap for counts), and fusion scales that by at most `max_weight`, so all
                remaining questions together move it by at most max_weight * delta * eps_left.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

P_CLIP = 1e-6


def _logit(p: float) -> float:
    p = min(max(float(p), P_CLIP), 1 - P_CLIP)
    return float(np.log(p / (1 - p)))


def bayes_threshold(cost_fn: float, cost_fp: float) -> float:
    """Cost-optimal threshold for a calibrated probability: escalate when p * c_fn >= (1 - p) * c_fp."""
    if cost_fn <= 0 or cost_fp <= 0:
        raise ValueError("costs must be positive")
    return cost_fp / (cost_fp + cost_fn)


def fit_threshold(p, y, objective: str = "cost", *, cost_fn: float = 1.0, cost_fp: float = 1.0,
                  day=None, capacity_per_day: int | None = None) -> float:  # fmt: skip
    """Threshold chosen on validation.

    cost      minimise mean cost c_fn * misses + c_fp * false alarms
    f1        maximise F1 of the escalate class
    capacity  the lowest threshold whose escalations stay within `capacity_per_day` on every day (needs `day`)
    Candidates are the observed probabilities (plus "escalate nothing" above 1). Ties go to the higher threshold.
    One sort and cumulative sums, so it is O(n log n).
    """
    p, y = np.asarray(p, dtype="f8"), np.asarray(y, dtype="f8")
    if p.shape != y.shape or len(p) == 0:
        raise ValueError("p and y need the same, non-zero length")
    nothing = float(np.nextafter(1.0, 2.0))  # a threshold above every probability
    if objective == "capacity":
        if day is None or capacity_per_day is None:
            raise ValueError("capacity needs day and capacity_per_day")
        day = np.asarray(day)
        need = -np.inf  # every day: threshold must exceed its (capacity+1)-th largest probability
        for d in np.unique(day):
            pd_ = np.sort(p[day == d])[::-1]
            if len(pd_) > capacity_per_day:
                need = max(need, pd_[capacity_per_day])
        cands = np.unique(p)
        above = cands[cands > need]
        return float(above[0]) if len(above) else nothing
    if objective not in ("cost", "f1"):
        raise ValueError(f"unknown objective {objective!r}")
    o = np.argsort(-p, kind="stable")
    ps, ys = p[o], y[o]
    last = np.r_[np.flatnonzero(np.diff(ps) != 0), len(ps) - 1]  # end of each block of equal probabilities
    tp = np.r_[0.0, np.cumsum(ys)[last]]                         # index 0 = escalate nothing
    fp = np.r_[0.0, np.cumsum(1 - ys)[last]]
    thr = np.r_[nothing, ps[last]]
    pos = ys.sum()
    if objective == "cost":
        score = -(cost_fn * (pos - tp) + cost_fp * fp) / len(p)
    else:
        denom = (tp + fp) + pos
        score = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)
    return float(thr[int(np.argmax(score))])  # argmax takes the first max = the highest threshold


def decide(p, threshold: float) -> np.ndarray:
    """'escalate' where p >= threshold, else 'clear'."""
    return np.where(np.asarray(p, dtype="f8") >= threshold, "escalate", "clear")


def max_log_odds_shift(eps_left: float, *, max_delta: float = 1.0, max_weight: float = 1.0) -> float:
    """Upper bound on how far all remaining answers can move the fused log-odds (privacy ceiling)."""
    return max(0.0, eps_left) * max_delta * max_weight


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    reason: str  # continue | budget | fast_clear | cannot_cross


def should_stop(posterior: float, threshold: float, *, eps_left: float, min_eps: float, calls_left: int,
                sim_days_left: float, max_delta: float = 1.0, max_weight: float = 1.0,
                fast_says_clear: bool = False) -> StopDecision:  # fmt: skip
    """The stop rule, in the order above. Budget comes first so a stop is never blamed on the evidence."""
    if eps_left + 1e-12 < min_eps or calls_left <= 0 or sim_days_left <= 0:
        return StopDecision(True, "budget")
    if fast_says_clear:
        return StopDecision(True, "fast_clear")
    gap = abs(_logit(posterior) - _logit(threshold))
    if gap > max_log_odds_shift(eps_left, max_delta=max_delta, max_weight=max_weight):
        return StopDecision(True, "cannot_cross")
    return StopDecision(False, "continue")
