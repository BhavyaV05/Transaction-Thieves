"""Ceiling checks on the privacy code (G2: "stay under the ceiling, a check on our privacy code"; tracks.md 6.7).

Owner: T4 (Shreyas), with T2 (Bhavya), whose code it checks: ftm.privacy.mechanisms (C5) and the back-test LRs (C7).

A DP answer bought with eps can shift the odds by at most the ceiling: e^eps for yes/no and k-ary answers,
e^(eps * diff / s) for counts whose true values differ by `diff` (plan.md 6.6). Three checks:

  exact      the mechanism's exact likelihoods never give a log LR above the log ceiling, over a grid of outputs
  empirical  draws from the mechanism the partner banks actually call (`randomize`) never give an empirical LR
             above the ceiling beyond sampling error (one-sided z on the log LR, delta-method standard error)
  backtest   the UNCLIPPED LRs implied by back-test cell counts (ftm.backtest cache entries, `cells`) stay under
             each entry's ceiling beyond sampling error. The stored LRs are clipped, so checking them would pass
             trivially; an unclipped excess points at a bug (noise not applied, wrong eps or sensitivity).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

import numpy as np

from ftm.privacy import mechanisms as mech

Z_CRIT = 3.29  # one-sided, about 0.0005 per cell: many cells are checked, false alarms must stay rare
MIN_CELL = 30  # cells with fewer draws on either side are reported but not judged


def exact_check(kind: str, eps: float, sensitivity: float = 1, arity: int | None = None, true_a=0, true_b=1,
                outputs: Iterable | None = None) -> dict:  # fmt: skip
    """Max exact |log LR| between true answers a and b over the outputs, against the log ceiling."""
    if kind == "bool":
        outs, a, b, diff = [False, True], bool(true_a), bool(true_b), 1
    elif kind == "kary":
        outs, a, b, diff = list(range(arity)), int(true_a), int(true_b), 1
    else:
        a, b = int(true_a), int(true_b)
        diff = abs(a - b)
        outs = list(outputs) if outputs is not None else list(range(min(a, b) - 30, max(a, b) + 31))
    log_c = math.log(mech.lr_ceiling(kind, eps, sensitivity, diff=diff))
    worst = max(abs(mech.log_lr(kind, eps, sensitivity, a, b, o, arity)) for o in outs)
    return {"kind": kind, "eps": eps, "sensitivity": sensitivity, "diff": diff, "max_abs_log_lr": worst,
            "log_ceiling": log_c, "ok": worst <= log_c + 1e-9}  # fmt: skip


def _cell_excess(c_a: int, n_a: int, c_b: int, n_b: int, log_c: float) -> tuple[float, float]:
    """(log LR, z of its excess over the log ceiling) for one cell; uses the larger of LR and 1/LR."""
    lr = math.log((c_a / n_a) / (c_b / n_b))
    se = math.sqrt(max(1 / c_a - 1 / n_a + 1 / c_b - 1 / n_b, 1e-12))
    return lr, (abs(lr) - log_c) / se


def empirical_check(kind: str, eps: float, sensitivity: float = 1, arity: int | None = None, true_a=0, true_b=1,
                    n: int = 20_000, seed: int = 0, bin_width: int = 1) -> dict:  # fmt: skip
    """Draw n answers from `mechanisms.randomize` for each of two true values; compare per-output-cell LRs."""
    rng = np.random.default_rng(seed)
    diff = 1 if kind in ("bool", "kary") else abs(int(true_a) - int(true_b))
    log_c = math.log(mech.lr_ceiling(kind, eps, sensitivity, diff=diff))

    def draws(v):
        out = [mech.randomize(kind, v, eps, sensitivity, rng, arity=arity) for _ in range(n)]
        return Counter(o if kind in ("bool", "kary") else int(o) // bin_width for o in out)

    ca, cb = draws(true_a), draws(true_b)
    cells, worst_z, judged = [], -math.inf, 0
    for cell in sorted(set(ca) | set(cb), key=str):
        a, b = ca.get(cell, 0), cb.get(cell, 0)
        if a < MIN_CELL or b < MIN_CELL:
            continue
        lr, z = _cell_excess(a, n, b, n, log_c)
        judged += 1
        worst_z = max(worst_z, z)
        cells.append({"cell": str(cell), "log_lr": lr, "z_excess": z})
    return {"kind": kind, "eps": eps, "sensitivity": sensitivity, "diff": diff, "n": n, "cells_judged": judged,
            "log_ceiling": log_c, "worst_z_excess": worst_z, "cells": cells,
            # fewer than 2 judged cells: the two answer distributions barely overlap, so sampling cannot test the
            # ceiling here (the exact check still does). Reported as inconclusive (None), never as a pass.
            "ok": (worst_z <= Z_CRIT) if judged >= 2 else None}  # fmt: skip


def backtest_check(entries: Iterable[dict], z_crit: float = Z_CRIT) -> dict:
    """Back-test cache entries ({canonical, eps, ceiling, cells: {cell: [n_clean, n_launder]}}) -> violations."""
    violations, judged, skipped = [], 0, 0
    for e in entries:
        if not e.get("cells") or e.get("ceiling") in (None, math.inf) or e.get("eps") in (None, math.inf):
            skipped += 1
            continue
        n_c = sum(c for c, _ in e["cells"].values())
        n_l = sum(lv for _, lv in e["cells"].values())
        log_c = math.log(e["ceiling"])
        for cell, (c0, c1) in e["cells"].items():
            if c0 < MIN_CELL or c1 < MIN_CELL:
                continue
            judged += 1
            lr, z = _cell_excess(c1, n_l, c0, n_c, log_c)
            if z > z_crit:
                violations.append({"canonical": e.get("canonical") or e.get("program_canonical"), "eps": e["eps"], "cell": cell,
                                   "log_lr_unclipped": lr, "log_ceiling": log_c, "z_excess": z})  # fmt: skip
    return {"cells_judged": judged, "entries_skipped": skipped, "violations": violations, "ok": not violations}


def default_grid() -> list[dict]:
    """The configurations scripts/ceiling_check.py runs: every answer kind at every eps level (plus 4)."""
    grid = []
    for eps in (0.25, 0.5, 1.0, 2.0, 4.0):
        grid.append({"kind": "bool", "eps": eps})
        grid.append({"kind": "kary", "eps": eps, "arity": 4, "true_a": 0, "true_b": 2})
        for s, (a, b) in ((1, (0, 1)), (1, (0, 3)), (2, (0, 2)), (3, (1, 4))):
            grid.append({"kind": "count", "eps": eps, "sensitivity": s, "true_a": a, "true_b": b})
        grid.append({"kind": "set_count", "eps": eps, "sensitivity": 1, "true_a": 0, "true_b": 2})
    return grid
