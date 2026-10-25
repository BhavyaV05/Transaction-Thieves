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
from ftm.privacy.accountant import PrivacyAccountant
from ftm.types import Limits

WAVES = (2, 1)


def batching_study(system: str, banks, val: Sequence[BtAlert], pool: QuestionPool, waves: tuple[int, ...] = WAVES,
                   eps_levels=(0.25, 0.5, 1.0, 2.0), seed: int = 0, limits: Limits | None = None,
                   arrival_gap_days: float = 0.05, budget: float | None = None, policy=None) -> list[dict]:  # fmt: skip
    """Sequential vs batched rows. `budget` sets the per-alert eps of both arms (the default is the harness's 2.0). A wave
    only holds two questions if two fit the eps left: `share_2plus` is the share of alerts that asked two or more at all, the
    ceiling on how much batching can have changed anything (T3 found a (2, 1) schedule never reached its second slot when
    every accepted question cost eps >= 1 against a budget of 2)."""
    limits = limits or Limits()
    if budget is not None:
        limits = replace(limits, eps_per_alert=budget)
    bindings = {a.alert_id: a.binding for a in val}

    def harness():
        acc = PrivacyAccountant(eps_per_alert=budget) if budget is not None else None
        return make_harness(banks, bindings, acc, eps_levels=eps_levels)

    rows = []
    for arm, w in (("sequential", None), ("batched", waves)):
        base = {"run_id": f"async-{arm}", "commit": "", "machine": "", "system": system, "variant": arm, "seed": seed}
        res = run_alerts(build_planner(system, pool, seed, eps=1.0, waves=w, policy=policy), val, harness(), limits, base)
        sim = run_simpy(build_planner(system, pool, seed, eps=1.0, waves=w, policy=policy), val, harness(), limits,
                        arrival_gap_days)  # fmt: skip
        asked = [sum(e.response is not None for e in v.questions_log) for v in res.verdicts]
        rows.append({**res.summary, "arm": arm, "waves": list(w) if w else None, "makespan_days": sim.makespan_days,
                     "max_in_flight": sim.max_in_flight, "budget": limits.eps_per_alert,
                     "share_2plus": sum(n >= 2 for n in asked) / len(asked) if asked else 0.0})  # fmt: skip
    return rows


def h8_check(rows: Sequence[dict], gain_share: float = 0.8, time_share: float = 0.6) -> dict:
    """H8's claim ("two questions at once, then one more keeps 80 percent of the benefit in 60 percent of the time") on the
    two rows of `batching_study`: the share of the sequential Brier gain the batched arm keeps and its share of the
    sequential simulated time. `claim_holds` is None when the sequential arm gained nothing (a share of nothing says nothing)
    or when the batched arm never asked a second question (nothing was batched)."""
    seq = next(r for r in rows if r["arm"] == "sequential")
    bat = next(r for r in rows if r["arm"] == "batched")
    kept = bat["brier_gain"] / seq["brier_gain"] if seq["brier_gain"] > 1e-9 else None
    time = bat["mean_sim_days"] / seq["mean_sim_days"] if seq["mean_sim_days"] > 1e-9 else None
    holds = None
    if kept is not None and time is not None and bat["share_2plus"] > 0:
        holds = bool(kept >= gain_share and time <= time_share)
    return {"gain_kept": kept, "time_ratio": time, "share_2plus_batched": bat["share_2plus"], "claim_holds": holds}


def batching_grid(systems: Sequence[str], banks, val: Sequence[BtAlert], pool: QuestionPool,
                  budgets: Sequence[float] = (2.0, 4.0), waves: tuple[int, ...] = WAVES, eps_levels=(0.25, 0.5, 1.0, 2.0),
                  seed: int = 0, policy=None) -> list[dict]:  # fmt: skip
    """E13 for several planners and per-alert budgets: a row per (system, budget, arm), then the H8 check per (system,
    budget) in `h8`. The larger budget is what lets a (2, 1) schedule fill its second slot when questions cost eps >= 1."""
    out, h8 = [], []
    for system in systems:
        for b in budgets:
            rows = batching_study(system, banks, val, pool, waves, eps_levels, seed, budget=b, policy=policy)
            out += [{**r, "system": system} for r in rows]
            h8.append({"system": system, "budget": b, **h8_check(rows)})
    return [*out, *({"arm": "h8", **h} for h in h8)]


def load_study(system: str, banks, val: Sequence[BtAlert], pool: QuestionPool, gaps: Sequence[float] = (0.5, 0.1, 0.02, 0.005),
               waves: tuple[int, ...] = WAVES, eps_levels=(0.25, 0.5, 1.0, 2.0), seed: int = 0,
               limits: Limits | None = None) -> list[dict]:  # fmt: skip
    """The SimPy picture under load: the same alerts arriving every `gap` days (a busy or a quiet queue), sequential vs
    batched. Per row the makespan, the most investigations open at once and the mean time an investigation took. The
    answers do not depend on the gap (banks reply by question), only the overlap does."""
    limits = limits or Limits()
    bindings = {a.alert_id: a.binding for a in val}
    rows = []
    for gap in gaps:
        for arm, w in (("sequential", None), ("batched", waves)):
            sim = run_simpy(build_planner(system, pool, seed, eps=1.0, waves=w), val,
                            make_harness(banks, bindings, eps_levels=eps_levels), limits, gap)  # fmt: skip
            lat = list(sim.latency_days.values())
            rows.append({"system": system, "arrival_gap_days": gap, "arm": arm, "alerts": len(val),
                         "makespan_days": sim.makespan_days, "max_in_flight": sim.max_in_flight,
                         "mean_latency_days": sum(lat) / len(lat) if lat else 0.0})  # fmt: skip
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
