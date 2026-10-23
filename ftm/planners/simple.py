"""C9: the non-LLM planners. Random, fixed order, EIG, and the enumerator (EIG over every program up to size n).

Owner: T2 (Bhavya).  Spec: plan.md 6.10, tracks.md 4.7.

All of them are Investigators over the same harness, the same budgets (`Limits`), the same stop rule (T4's
`should_stop`) and the same fusion (T4's `posterior`), choosing from a pool of back-tested accepted questions
(`QuestionPool`). Only the choice differs:

  random      uniform over the questions that fit the eps left
  fixed       the expert ranking (the family order of configs/questions/expert_specs.yaml), at one eps level
  eig         the question with the largest expected reduction in Brier per unit eps (see `expected_brier_gain`)
  enumerator  eig over a pool built from every DSL program up to size n, each back-tested like the expert list

By default a family is asked at most once per alert (`one_per_family`): two questions of one template are not
independent evidence, and fusion would otherwise double-count them. Learned policy: week 9.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ftm.fusion.fuse import Fusion, logit, sigmoid
from ftm.harness.interface import ToolBox
from ftm.planners.pool import Candidate, QuestionPool
from ftm.planners.session import PlanSession
from ftm.types import AlertContext, Limits, Verdict

MIN_GAIN = 1e-9


def expected_brier_gain(p: float, c: Candidate) -> float:
    """Expected drop in Brier score from asking `c` when the current posterior is `p`.

    An answer cell a has probability P(a) = p P(a | launder) + (1 - p) P(a | clean) (measured on the noisy back-test
    answers) and moves the posterior to p'(a) = sigmoid(logit p + log LR(a)), the same shift fusion applies. A
    calibrated posterior q has expected Brier q (1 - q), so the gain is
        p (1 - p)  -  sum_a P(a) p'(a) (1 - p'(a)).
    """
    if not c.p_cell or not (0.0 < p < 1.0):
        return 0.0  # no measured cell probabilities, or nothing left to learn
    z = float(logit(p))
    after = 0.0
    for cell, (p_clean, p_launder) in c.p_cell.items():
        lr = c.lr.get(cell)
        if lr is None or lr <= 0:
            continue
        pa = p * p_launder + (1 - p) * p_clean
        q = float(sigmoid(z + math.log(lr)))
        after += pa * q * (1 - q)
    return p * (1 - p) - after


@dataclass
class _Base:
    pool: QuestionPool
    fusion: Fusion | None = None
    threshold: float | None = None
    one_per_family: bool = True
    max_delta: float = 1.0
    max_weight: float = 1.0
    name: str = "planner"

    def choose(self, s: PlanSession, options: list[Candidate]) -> Candidate | None:
        raise NotImplementedError

    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict:
        s = PlanSession(alert, tools, limits, self.fusion, self.threshold, self.max_delta, self.max_weight)
        pool = self.pool.applicable(s.direction)
        while len(s.log) < limits.max_llm_calls * 5:  # a hard cap on loop length: refusals also use a turn
            if s.stop().stop:
                break
            options = s.affordable(pool, self.one_per_family)
            pick = self.choose(s, options) if options else None
            if pick is None:
                break
            s.ask(pick)
        return s.verdict()


@dataclass
class RandomPlanner(_Base):
    seed: int = 0
    name: str = "random"

    def investigate(self, alert, tools, limits):
        self._rng = random.Random(f"{self.seed}|{alert.alert_id}")  # per alert: order of alerts does not matter
        return super().investigate(alert, tools, limits)

    def choose(self, s, options):
        return self._rng.choice(options)


@dataclass
class FixedOrderPlanner(_Base):
    ranking: tuple[str, ...] = ()  # DSL family keys (ftm.dsl.expert.expert_ranking), best first; others last
    eps: float = 1.0  # the one eps level this planner asks at
    name: str = "fixed"

    def choose(self, s, options):
        rank = {f: i for i, f in enumerate(self.ranking)}
        atlevel = [c for c in options if abs(c.eps - self.eps) < 1e-9] or sorted(options, key=lambda c: -c.eps)[:1]
        if not atlevel:
            return None
        return min(atlevel, key=lambda c: (rank.get(c.family, len(rank)), c.family, c.program))


@dataclass
class EIGPlanner(_Base):
    name: str = "eig"

    def choose(self, s, options):
        scored = [(expected_brier_gain(s.p, c) / c.eps, c) for c in options]
        best = max(scored, key=lambda t: (t[0], t[1].program, -t[1].eps), default=None)
        if best is None or best[0] * best[1].eps <= MIN_GAIN:
            return None  # nothing left that is expected to help
        return best[1]


@dataclass
class EnumeratorPlanner(EIGPlanner):
    """EIG over the back-tested enumeration of every DSL program up to `max_size` (build the pool with
    `scripts/backtest.py --questions enumerate --max-size N`). Same choice rule as EIG, a much bigger menu."""

    max_size: int = 4
    name: str = "enumerator"
