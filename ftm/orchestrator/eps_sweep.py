"""The eps sweep (E2, the privacy-detection curve): one system at per-alert budgets 0.5, 1, 2, 4 and no noise.

Owner: T2 (Bhavya).  Spec: plan.md 6.6 ("eps sweep: 0.5, 1, 2, 4 and no noise"), tracks.md 4.11.

For each per-alert budget B the same system runs on the same validation alerts through the same harness; only the budget
changes (`Limits.eps_per_alert`, the accountant, and with it how many questions of the available eps levels fit). The
no-noise point answers every question exactly (eps = inf, an explicit opt-in of the harness, `Harness.allow_exact`) from
a pool back-tested the same way. Every row carries the theoretical ceiling: with basic composition and unit sensitivity,
B eps of questions can move the log-odds by at most B, so the likelihood ratio by at most e^B. That is the line to draw on
the privacy plot; results are read against it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

from ftm.backtest.engine import BacktestConfig, BtAlert
from ftm.orchestrator.runner import build_planner, make_harness, pool_from_backtest, run_alerts
from ftm.planners.pool import QuestionPool
from ftm.privacy.accountant import PrivacyAccountant
from ftm.types import Limits

BUDGETS = (0.5, 1.0, 2.0, 4.0, math.inf)
EPS_LEVELS = (0.25, 0.5, 1.0, 2.0)


def ceiling(budget: float) -> dict:
    return {"ceiling_log_odds": budget, "ceiling_lr": math.exp(budget) if math.isfinite(budget) else math.inf}


def fixed_level(budget: float, levels: Sequence[float]) -> float:
    """The eps level the fixed-order planner asks at: the largest level that leaves room for two questions."""
    ok = [lv for lv in levels if lv <= budget / 2 + 1e-9]
    return max(ok) if ok else min(levels)


def eps_sweep(system: str, banks, val: Sequence[BtAlert], pool_noisy: QuestionPool, pool_exact: QuestionPool | None,
              budgets: Sequence[float] = BUDGETS, eps_levels: Sequence[float] = EPS_LEVELS, seed: int = 0,
              threshold: float | None = None) -> list[dict]:  # fmt: skip
    """One row per budget: Brier before and after, questions and eps per alert, and the ceiling."""
    bindings = {a.alert_id: a.binding for a in val}
    rows = []
    for budget in budgets:
        exact = not math.isfinite(budget)
        if exact and pool_exact is None:
            continue
        acc = PrivacyAccountant(eps_per_alert=2.0 if exact else budget)
        harness = make_harness(banks, bindings, acc, eps_levels=eps_levels, allow_exact=exact)
        planner = build_planner(system, pool_exact if exact else pool_noisy, seed,
                                eps=math.inf if exact else fixed_level(budget, eps_levels), threshold=threshold)  # fmt: skip
        limits = Limits(eps_per_alert=2.0 if exact else budget)  # the no-noise arm is not eps-limited, only call-limited
        base = {"run_id": f"sweep-{budget:g}", "commit": "", "machine": "", "system": system,
                "variant": "eps_sweep", "seed": seed}  # fmt: skip
        res = run_alerts(planner, val, harness, limits, base)
        rows.append({**res.summary, "budget": budget, **ceiling(budget)})
    return rows


def synthetic_eps_sweep(system: str = "eig", n_alerts: int = 360, seed: int = 0, programs: Sequence[str] | None = None,
                        budgets: Sequence[float] = BUDGETS, train_frac: float = 0.6) -> list[dict]:  # fmt: skip
    """End to end on the synthetic world (smoke test, never a result)."""
    from ftm.backtest.synth import make_world
    from ftm.dsl.expert import expert_programs

    w = make_world(n_alerts, seed)
    alerts = sorted(w.alerts, key=lambda a: a.binding.t0)
    cut = int(len(alerts) * train_frac)
    train, val = alerts[:cut], alerts[cut:]
    rate = sum(a.label for a in train) / len(train)
    val = [replace(a, p0=rate, split="val") for a in val]
    if programs is None:
        allp = [p["program"] for p in expert_programs("receiver")]
        programs = allp[:: max(1, len(allp) // 14)]
    cfg = BacktestConfig(min_answers=50, min_per_class=8, min_per_bin=12)
    pool_noisy, _ = pool_from_backtest(w.banks, train, programs, EPS_LEVELS[1:], cfg)  # 0.5, 1, 2
    pool_exact, _ = pool_from_backtest(w.banks, train, programs, (math.inf,), cfg)
    return eps_sweep(system, w.banks, val, pool_noisy, pool_exact, budgets, EPS_LEVELS, seed)
