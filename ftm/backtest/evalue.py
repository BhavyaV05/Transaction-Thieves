"""E-values for the question-acceptance test (C7): a label-permutation e-process, e-BH, and e-LOND.

Owner: T2 (Bhavya); statistics reviewed by T4.  Spec: plan.md 6.8, tracks.md 4.6.  References in plan.md: POPPER [33]
(sequential tests by betting), e-BH [37] (Wang & Ramdas 2022), e-LOND (Xu & Ramdas 2024).

Null hypothesis for one question: its (noisy) answer carries no information about the alert's label.

**The e-process** conditions on what the null leaves fixed: the answers (in an order that does not depend on the
labels, here the alert-time order) and the number of laundering alerts. Under the null the labels are then a uniformly
random arrangement, so at step t the chance that alert t is laundering, given the labels already seen, is exactly
`remaining_launder / remaining_alerts` (sampling without replacement). We bet against it: any *predictable* guess
q_t of P(launder | answer_t) gives the factor q_t / null_t (or (1-q_t)/(1-null_t)), and the product E_t is a
nonnegative martingale with E_0 = 1 under the null, so E_t is a valid e-value at every stopping time (Ville), which
is what makes it safe under optional stopping. The guess is the running launder rate in the same answer cell,
shrunk toward the null, so a question whose cells separate the classes grows exponentially and a useless one stays
near 1. Validity does not depend on the guess being good, only on its being computed from the past.

e-BH over m hypotheses at level alpha rejects the k largest e-values where k is the largest index with
e_(k) >= m / (alpha k). It controls FDR at alpha under *arbitrary* dependence, which matters because the same
question at several eps, and overlapping questions, are dependent.

e-LOND is the online version for questions that arrive one at a time at test time (new LLM-written questions):
alpha_t = alpha * gamma_t * (R_{t-1} + 1) and reject when e_t >= 1/alpha_t; FDR <= alpha under arbitrary dependence.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field

Q_FLOOR = 1e-3  # keep the bet strictly inside (0, 1) so one surprise cannot zero the process


def log_evalue_path(cells: Sequence[Hashable], labels: Sequence[bool], shrink: float = 2.0) -> list[float]:
    """log E_t after each alert t (the whole e-process). `cells[i]` is the answer cell of alert i, in a
    label-independent order."""
    n = len(labels)
    if n != len(cells):
        raise ValueError("cells and labels differ in length")
    n_launder = sum(bool(y) for y in labels)
    seen_launder = 0
    counts: dict = defaultdict(lambda: [0, 0])  # cell -> [launder, total]
    total, path = 0.0, []
    for t, (cell, y) in enumerate(zip(cells, labels, strict=True)):
        remaining, rem_launder = n - t, n_launder - seen_launder
        if 0 < rem_launder < remaining:  # otherwise the rest is determined and there is nothing to bet on
            null = rem_launder / remaining
            c = counts[cell]
            q = (c[0] + shrink * null) / (c[1] + shrink)
            q = min(max(q, Q_FLOOR), 1 - Q_FLOOR)
            total += math.log(q / null) if y else math.log((1 - q) / (1 - null))
        counts[cell][0] += bool(y)
        counts[cell][1] += 1
        seen_launder += bool(y)
        path.append(total)
    return path


def log_evalue(cells: Sequence[Hashable], labels: Sequence[bool], shrink: float = 2.0) -> float:
    """log of the final e-value (the end of `log_evalue_path`)."""
    path = log_evalue_path(cells, labels, shrink)
    return path[-1] if path else 0.0


def evalue(cells: Sequence[Hashable], labels: Sequence[bool], shrink: float = 2.0) -> float:
    le = log_evalue(cells, labels, shrink)
    return math.exp(le) if le < 700 else math.inf


def e_bh(evalues: Sequence[float], alpha: float = 0.1) -> list[bool]:
    """e-BH: which hypotheses are rejected (accepted as informative questions) at FDR level alpha."""
    m = len(evalues)
    order = sorted(range(m), key=lambda i: -evalues[i])
    k_star = 0
    for k, i in enumerate(order, start=1):
        if evalues[i] >= m / (alpha * k):
            k_star = k
    out = [False] * m
    for i in order[:k_star]:
        out[i] = True
    return out


def null_false_acceptance(cells_by_hypothesis: Sequence[Sequence[Hashable]], labels: Sequence[bool], n_rep: int,
                          alpha: float = 0.10, seed: int = 0, shrink: float = 2.0) -> dict:  # fmt: skip
    """Control experiment: permute the labels (answers stay put), so every question is a null question, and count how
    often e-BH accepts anything. With all hypotheses null the FDR equals P(at least one acceptance) <= alpha."""
    import random

    rng = random.Random(seed)
    labels = list(labels)
    any_accept, accepted = 0, 0
    for _ in range(n_rep):
        rng.shuffle(labels)
        flags = e_bh([evalue(c, labels, shrink) for c in cells_by_hypothesis], alpha)
        any_accept += any(flags)
        accepted += sum(flags)
    return {"n_rep": n_rep, "false_acceptance_rate": any_accept / n_rep,
            "mean_accepted": accepted / n_rep}  # fmt: skip


@dataclass
class ELond:
    """Online FDR control with e-values (Xu & Ramdas). `test(e)` is called once per arriving hypothesis."""

    alpha: float = 0.1
    t: int = 0
    rejections: int = 0
    history: list = field(default_factory=list)

    @staticmethod
    def gamma(t: int) -> float:
        return 1.0 / (t * (t + 1))  # sums to 1 over t >= 1

    def level(self) -> float:
        """The level the next hypothesis will be tested at."""
        return self.alpha * self.gamma(self.t + 1) * (self.rejections + 1)

    def test(self, e: float) -> bool:
        level = self.level()
        self.t += 1
        reject = e >= 1.0 / level
        self.rejections += reject
        self.history.append((self.t, float(e), level, bool(reject)))
        return bool(reject)

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "t": self.t, "rejections": self.rejections}

    @classmethod
    def from_dict(cls, d: dict) -> ELond:
        return cls(d["alpha"], d["t"], d["rejections"])
