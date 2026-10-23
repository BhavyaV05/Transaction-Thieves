"""Sequential vs batched questions (E13, H8): what asking in waves buys in time and costs in accuracy.

Owner: T2 (Bhavya).  Spec: tracks.md 4.9 ("Async mode: batched questions (k=2 then one more) for H8/E13"), 4.11.

The same planner, pool, budget and validation alerts, run twice: one question at a time (each waits for the last answer)
and in waves (default (2, 1): two questions asked together, then one more). A wave costs its slowest answer, the rule T3's
LLMCompiler port also uses, so the two arms are comparable. Per arm: Brier before and after, questions and eps per alert,
simulated days per alert (mean and p95), and from the SimPy run the makespan and the most investigations open at once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from ftm.backtest.engine import BacktestConfig, BtAlert
from ftm.orchestrator.runner import build_planner, make_harness, pool_from_backtest, run_alerts
from ftm.orchestrator.simpy_run import run_simpy
from ftm.planners.pool import QuestionPool
from ftm.types import Limits

WAVES = (2, 1)


def batching_study(system: str, banks, val: Sequence[BtAlert], pool: QuestionPool, waves: tuple[int, ...] = WAVES,
                   eps_levels=(0.25, 0.5, 1.0, 2.0), seed: int = 0, limits: Limits | None = None,
                   arrival_gap_days: float = 0.05) -> list[dict]:  # fmt: skip
    limits = limits or Limits()
    bindings = {a.alert_id: a.binding for a in val}
    rows = []
    for arm, w in (("sequential", None), ("batched", waves)):
        base = {"run_id": f"async-{arm}", "commit": "", "machine": "", "system": system, "variant": arm, "seed": seed}
        res = run_alerts(build_planner(system, pool, seed, eps=1.0, waves=w), val,
                         make_harness(banks, bindings, eps_levels=eps_levels), limits, base)
        sim = run_simpy(build_planner(system, pool, seed, eps=1.0, waves=w), val,
                        make_harness(banks, bindings, eps_levels=eps_levels), limits, arrival_gap_days)
        rows.append({**res.summary, "arm": arm, "waves": list(w) if w else None, "makespan_days": sim.makespan_days,
                     "max_in_flight": sim.max_in_flight})  # fmt: skip
    return rows


def synthetic_batching_study(system: str = "eig", n_alerts: int = 360, seed: int = 0, programs: Sequence[str] | None = None,
                             waves: tuple[int, ...] = WAVES) -> list[dict]:  # fmt: skip
    """End to end on the synthetic world (smoke test, never a result)."""
    from ftm.backtest.synth import make_world
    from ftm.dsl.expert import expert_programs

    w = make_world(n_alerts, seed)
    alerts = sorted(w.alerts, key=lambda a: a.binding.t0)
    cut = int(len(alerts) * 0.6)
    rate = sum(a.label for a in alerts[:cut]) / cut
    val = [replace(a, p0=rate, split="val") for a in alerts[cut:]]
    if programs is None:
        allp = [p["program"] for p in expert_programs("receiver")]
        programs = allp[:: max(1, len(allp) // 14)]
    pool, _ = pool_from_backtest(w.banks, alerts[:cut], programs, (0.5, 1.0, 2.0),
                                 BacktestConfig(min_answers=50, min_per_class=8, min_per_bin=12))  # fmt: skip
    return batching_study(system, w.banks, val, pool, waves, seed=seed)
