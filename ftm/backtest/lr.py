"""Likelihood-ratio estimation for one back-tested question (C7).

Owner: T2 (Bhavya); statistics reviewed by T4.  Spec: plan.md 6.8.

From the noisy answers a question received on past alerts with known labels, estimate for every answer cell
    LR(cell) = (P(cell | launder) + alpha) / (P(cell | clean) + alpha)
and clip it to the DP odds-shift ceiling (`ftm.privacy.mechanisms.lr_ceiling`). The answers are already noisy, so
the estimate is the likelihood ratio of the noisy answer the system will see at run time, not of the true value.

Answer cells use the same keys as T3's `LRTable` (ftm/llm/investigation.py: `answer_key`), so the table this module
feeds can be read there unchanged:  bool -> "true"/"false";  counts -> "b<i>" with i = number of bin edges <= value;
k-ary -> str(value).
"""

from __future__ import annotations

import bisect
import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ftm.privacy.mechanisms import is_exact, lr_ceiling
from ftm.types import AnswerKind


def answer_key(value, bins: Sequence[float] | None = None) -> str:
    """Same contract as `ftm.llm.investigation.answer_key` (T3)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if bins is not None and isinstance(value, numbers.Real):
        return f"b{bisect.bisect_right(list(bins), value)}"
    return str(value)


def choose_bins(values: Sequence[float], max_bins: int = 6, min_per_bin: int = 20) -> tuple[float, ...]:
    """Bin edges for count answers: quantiles of the pooled noisy answers, merged until every bin is big enough.

    Uses answers only, never labels, so the cells are fixed before the labels are looked at."""
    v = np.sort(np.asarray(values, dtype=float))
    if len(v) == 0:
        return ()
    qs = np.quantile(v, np.linspace(0, 1, max_bins + 1)[1:-1], method="lower")
    edges = sorted({float(q) for q in qs if q > v[0]})
    while edges:  # drop the edge that leaves the smallest bin until all bins hold min_per_bin
        cuts = [-math.inf, *edges, math.inf]
        sizes = [int(((v >= lo) & (v < hi)).sum()) for lo, hi in zip(cuts[:-1], cuts[1:], strict=True)]
        if min(sizes) >= min_per_bin:
            break
        i = int(np.argmin(sizes))  # merge the smallest bin into its smaller neighbour
        if i == 0:
            edges.pop(0)
        elif i == len(edges):
            edges.pop(-1)
        else:
            edges.pop(i - 1 if sizes[i - 1] <= sizes[i + 1] else i)
    return tuple(edges)


def cells_for(kind: AnswerKind, values: Sequence, bins: Sequence[float] | None) -> list[str]:
    if kind in ("count", "set_count"):
        return [answer_key(v, bins or ()) for v in values]
    return [answer_key(v) for v in values]


@dataclass(frozen=True)
class LREstimate:
    lr: dict[str, float]  # cell -> clipped LR
    counts: dict[str, tuple[int, int]]  # cell -> (n clean, n launder)
    ceiling: float
    n_clean: int
    n_launder: int
    bins: tuple[float, ...]


def estimate_lr(kind: AnswerKind, values: Sequence, labels: Sequence[bool], eps: float | None, sensitivity: float,
                arity: int | None = None, alpha: float = 0.01, bins: Sequence[float] | None = None,
                ceiling_diff: float | None = None) -> LREstimate:  # fmt: skip
    """LRs per answer cell. `bins` are the count-cell edges (from `choose_bins`); `ceiling_diff` is the typical gap
    between launder and clean true counts for the count ceiling (default: the gap of the noisy medians, at least
    the sensitivity)."""
    cells = cells_for(kind, values, bins)
    y = np.asarray(labels, dtype=bool)
    if kind == "bool":
        universe = ["false", "true"]
    elif kind == "kary":
        universe = [str(i) for i in range(arity)]
    else:
        universe = [f"b{i}" for i in range(len(bins or ()) + 1)]
    n_l, n_c = int(y.sum()), int((~y).sum())
    cnt = {k: [0, 0] for k in universe}
    for c, lab in zip(cells, y, strict=True):
        cnt[c][1 if lab else 0] += 1
    if kind in ("count", "set_count") and ceiling_diff is None:
        v = np.asarray(values, dtype=float)
        gap = abs(float(np.median(v[y])) - float(np.median(v[~y]))) if n_l and n_c else 0.0
        ceiling_diff = max(float(sensitivity), gap)
    ceiling = lr_ceiling(kind, eps, sensitivity or 1, diff=ceiling_diff or 1)
    lr = {}
    for k, (c0, c1) in cnt.items():
        p_l = c1 / n_l if n_l else 0.0
        p_c = c0 / n_c if n_c else 0.0
        v = (p_l + alpha) / (p_c + alpha)
        if not is_exact(eps):
            v = min(max(v, 1.0 / ceiling), ceiling)
        lr[k] = v
    return LREstimate(lr, {k: (a, b) for k, (a, b) in cnt.items()}, ceiling, n_c, n_l, tuple(bins or ()))
