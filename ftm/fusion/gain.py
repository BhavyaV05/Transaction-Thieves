"""Expected Brier reduction of asking a back-tested question under the fitted fusion: ONE implementation.

Owner: T3 (Akshat).  Spec: wave 4 (docs/t3_system1.md), prompt budget (docs/t3_prompt_budget.md).

Two callers share it: the System 1 `worth_asking` labels (`ftm.orchestrator.decision_set`: yes iff the best question's gain is at
least GAIN_MIN) and the planner's menu ranking (`ftm.graph.menu_rank`: show the top-K questions by this gain at the current
posterior, drop those with gain <= 0).  Anything with `lr` (cell -> LR), `p_cell` (cell -> (P(cell | clean), P(cell | launder)))
and `family` works as a question: a `ftm.planners.pool.Candidate`, a `ftm.graph.policy.MenuItem` (via `Cells`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from ftm.fusion.fuse import sigmoid


@dataclass(frozen=True)
class Cells:
    """The three things `expected_gain` reads of a question."""

    family: str
    lr: dict
    p_cell: dict | None


class UnitFusion:
    """Naive-Bayes stand-in for a fusion with no fitted weights (shrink 1, every family weight 1): the arithmetic of
    `NaiveLRFusion`, so the gain of a menu is defined under it too."""

    shrink = 1.0

    @staticmethod
    def weight(_family) -> float:
        return 1.0


def fused_logit(model, p0: float, answered: Iterable[tuple[str, float]]) -> float:
    """logit of the fused posterior after answers [(family, LR)]: the arithmetic of `ConditionalLR.posterior`."""
    x = sum(model.weight(fam) * math.log(lr) for fam, lr in answered if lr and lr > 0)
    return float(model.logit_base([p0])[0]) + model.shrink * x


def expected_gain(model, z: float, cand) -> float | None:
    """Expected Brier reduction of asking `cand` at the fused log-odds z: p(1-p) - sum_cell [p P(cell|launder) (1-p')^2 +
    (1-p) P(cell|clean) p'^2] with the cell probabilities from the back-test table and p' the fused posterior after that cell.
    None when the table has no cell probabilities for the question."""
    if not cand.p_cell:
        return None
    p = float(sigmoid(z))
    after = 0.0
    for cell, lr in cand.lr.items():
        if cell not in cand.p_cell or not lr or lr <= 0:
            continue
        pc, pl_ = cand.p_cell[cell]
        q = float(sigmoid(z + model.shrink * model.weight(cand.family) * math.log(lr)))
        after += p * pl_ * (1 - q) ** 2 + (1 - p) * pc * q**2  # the fused forecast q scored against the table's outcome mix in this cell
    return p * (1 - p) - after
