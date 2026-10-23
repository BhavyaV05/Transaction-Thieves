"""Two-tier conformal cascade (C13, stretch): fast model first, the 8B when the fast model is unsure.

Owner: T3 (Akshat).  Spec: plan.md 6.14, tracks.md 5.6 (Ref [35]).

Per decision type (Mondrian), split conformal with the LAC score s = 1 - p(true label), fitted on a calibration part
of VALIDATION:  q = the ceil((n+1)(1-alpha))/n empirical quantile of s;  prediction set = {y : p(y) >= 1 - q}.
  fast set has exactly one label  -> accept the fast answer
  else the 8B set has exactly one  -> accept the 8B answer
  else                             -> the decision's safe default (keep investigating / hand to a human)
Marginal miscoverage of each tier is <= alpha, and an accepted singleton is wrong only if the true label is outside
that tier's set, so P(an answer is accepted AND wrong) <= 2*alpha (union bound over the two tiers). This is a
MARGINAL bound: the error rate among accepted answers alone is not bounded and can exceed 2*alpha when few are
accepted. `evaluate` reports both, so the tighter condition can be checked on held-out validation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from ftm.fastmodel.decisions import DECISIONS


class ConformalSet:
    """Mondrian split-conformal prediction sets from label probabilities."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.q: dict[str, float] = {}
        self.n: dict[str, int] = {}

    def fit(self, decisions: list[str], probs: list[dict[str, float]], labels: list[str]) -> ConformalSet:
        scores: dict[str, list[float]] = defaultdict(list)
        for d, p, y in zip(decisions, probs, labels, strict=True):
            scores[d].append(1.0 - p[y])
        for d, s in scores.items():
            n = len(s)
            k = math.ceil((n + 1) * (1 - self.alpha))
            self.q[d] = sorted(s)[k - 1] if k <= n else 1.0  # too few calibration points: the full set
            self.n[d] = n
        return self

    def predict_set(self, decision: str, probs: dict[str, float]) -> list[str]:
        q = self.q.get(decision, 1.0)
        return [y for y, p in probs.items() if p >= 1.0 - q - 1e-12]


@dataclass
class CascadeDecision:
    decision: str
    answer: str | None
    source: str  # fast | large | default
    fast_set: list[str] = field(default_factory=list)
    large_set: list[str] = field(default_factory=list)


class ConformalCascade:
    def __init__(self, fast_sets: ConformalSet, large_sets: ConformalSet):
        self.fast_sets, self.large_sets = fast_sets, large_sets

    def decide(self, decision: str, fast_probs: dict[str, float], large_probs_fn) -> CascadeDecision:
        """`large_probs_fn()` is only called when the fast tier defers (that is where the savings come from)."""
        fs = self.fast_sets.predict_set(decision, fast_probs)
        if len(fs) == 1:
            return CascadeDecision(decision, fs[0], "fast", fs)
        ls = self.large_sets.predict_set(decision, large_probs_fn())
        if len(ls) == 1:
            return CascadeDecision(decision, ls[0], "large", fs, ls)
        return CascadeDecision(decision, DECISIONS[decision].safe_default, "default", fs, ls)


def evaluate(results: list[CascadeDecision], labels: list[str], alpha: float) -> dict:
    """Per decision type: share kept by the fast tier, error among accepted answers, deferrals to default."""
    by: dict[str, list[tuple[CascadeDecision, str]]] = defaultdict(list)
    for r, y in zip(results, labels, strict=True):
        by[r.decision].append((r, y))
    out = {}
    for d, rs in by.items():
        n = len(rs)
        accepted = [(r, y) for r, y in rs if r.source in ("fast", "large")]
        wrong = sum(r.answer != y for r, y in accepted)
        out[d] = {"n": n, "kept_fast": sum(r.source == "fast" for r, _ in rs) / n,
                  "to_large": sum(r.source == "large" for r, _ in rs) / n,
                  "to_default": sum(r.source == "default" for r, _ in rs) / n,
                  "error_marginal": wrong / n,  # P(accepted and wrong): the quantity bounded by 2*alpha
                  "error_given_accepted": wrong / len(accepted) if accepted else float("nan"),  # not bounded
                  "bound_2alpha": 2 * alpha, "within_bound": wrong / n <= 2 * alpha}  # fmt: skip
    return out
