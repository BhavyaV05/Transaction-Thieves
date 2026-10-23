"""Run a non-LLM planner over a set of alerts through the harness; summarise; write the run log.

Owner: T2 (Bhavya).  The orchestrator's sequential mode for the non-LLM arms (C15). Used by `scripts/run_experiment.py`
and by the tests; the same harness (gate, accountant, clock) every other system goes through.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from ftm.backtest.engine import BacktestConfig, Backtester, BtAlert, accept, lr_table_rows
from ftm.banks.binding import alert_context
from ftm.compliance.gate import Gate
from ftm.dsl.expert import expert_programs, expert_ranking
from ftm.dsl.parser import gate_validator
from ftm.harness.runlog_rows import verdict_rows
from ftm.harness.toolbox import Harness
from ftm.planners.pool import QuestionPool
from ftm.planners.simple import EIGPlanner, EnumeratorPlanner, FixedOrderPlanner, RandomPlanner
from ftm.privacy.accountant import PrivacyAccountant
from ftm.types import Limits

SYSTEMS = ("random", "fixed", "eig", "enumerator")


def build_planner(system: str, pool: QuestionPool, seed: int = 0, eps: float = 1.0, threshold: float | None = None,
                  ranking: Sequence[str] | None = None, max_size: int = 4):  # fmt: skip
    if system == "random":
        return RandomPlanner(pool, threshold=threshold, seed=seed)
    if system == "fixed":
        return FixedOrderPlanner(pool, threshold=threshold, ranking=tuple(ranking or expert_ranking()), eps=eps)
    if system == "eig":
        return EIGPlanner(pool, threshold=threshold)
    if system == "enumerator":
        return EnumeratorPlanner(pool, threshold=threshold, max_size=max_size)
    raise ValueError(f"system must be one of {SYSTEMS}")


@dataclass
class RunResult:
    rows: list[dict]
    summary: dict
    verdicts: list = field(default_factory=list)


def run_alerts(planner, alerts: Sequence[BtAlert], harness: Harness, limits: Limits, base: dict,
               split: str = "val") -> RunResult:  # fmt: skip
    """Investigate every alert in a fresh harness session and score the verdicts against the labels."""
    tools = harness.toolbox()
    system = tools.wrap(planner)
    rows, verdicts, p0s, ps, ys, secs = [], [], [], [], [], []
    for a in alerts:
        ctx = alert_context(a.binding, a.p0, split)
        t = time.perf_counter()
        v = system.investigate(ctx, tools, limits)
        secs.append(time.perf_counter() - t)
        rows += verdict_rows(v, {**base, "split": split, "alert_id": a.alert_id}, int(a.label))
        verdicts.append(v)
        p0s.append(a.p0)
        ps.append(a.p0 if v.probability is None else v.probability)
        ys.append(float(a.label))
    p0s, ps, ys = np.array(p0s), np.array(ps), np.array(ys)
    n_q = [sum(e.gate_decision == "allowed" and e.response is not None for e in v.questions_log) for v in verdicts]
    summary = {
        "system": base["system"], "alerts": len(alerts), "positives": int(ys.sum()),
        "brier_p0": float(np.mean((p0s - ys) ** 2)), "brier_final": float(np.mean((ps - ys) ** 2)),
        "mean_questions": float(np.mean(n_q)) if n_q else 0.0,
        "mean_eps": float(np.mean([sum(e.eps for e in v.questions_log if hasattr(e.response, "eps_spent"))
                                   for v in verdicts])) if verdicts else 0.0,
        "mean_seconds_per_alert": float(np.mean(secs)) if secs else 0.0,
        "eps_cal": None,
    }  # fmt: skip
    summary["brier_gain"] = summary["brier_p0"] - summary["brier_final"]
    return RunResult(rows, summary, verdicts)


def make_harness(world_banks, bindings, accountant: PrivacyAccountant | None = None, eps_levels=None) -> Harness:
    acc = accountant or PrivacyAccountant()
    gate = Gate.from_config(validator=gate_validator, accountant=acc)
    if eps_levels:
        gate.eps_levels = tuple(eps_levels)
    return Harness(world_banks, gate, acc, bindings)


def pool_from_backtest(banks, train: Sequence[BtAlert], programs: Sequence[str], eps_levels: Sequence[float],
                       config: BacktestConfig | None = None, fdr: float = 0.10) -> tuple[QuestionPool, list[dict]]:  # fmt: skip
    """Back-test `programs` on the training alerts, accept with e-BH, and return the pool of accepted questions."""
    bt = Backtester(banks, train, config or BacktestConfig())
    entries = accept(bt.run_batch(programs, eps_levels), fdr)
    return QuestionPool.from_rows(lr_table_rows(entries)), entries


def synthetic_experiment(system: str, n_alerts: int = 360, seed: int = 0, eps_levels=(1.0, 2.0), signal: float = 1.0,
                         programs: Sequence[str] | None = None, train_frac: float = 0.6, limits: Limits | None = None,
                         max_size: int = 3, max_programs: int = 14) -> RunResult:  # fmt: skip
    """End to end on the synthetic world: back-test on the first `train_frac` of the alerts in time, run the planner on
    the rest. For smoke tests and the planner unit tests; never a result."""
    from ftm.backtest.synth import make_world
    from ftm.dsl.enumerate import enumerate_programs
    from ftm.dsl.parser import canonical

    w = make_world(n_alerts, seed, signal)
    alerts = sorted(w.alerts, key=lambda a: a.binding.t0)
    cut = int(len(alerts) * train_frac)
    train, val = alerts[:cut], alerts[cut:]
    rate = sum(a.label for a in train) / len(train)
    val = [replace(a, p0=rate, split="val") for a in val]  # p0 = the base rate of the training period
    if programs is None:
        if system == "enumerator":
            programs = [canonical(n) for n in enumerate_programs(max_size, ("receiver",))]
        else:
            programs = [p["program"] for p in expert_programs("receiver")]
            programs = programs[:: max(1, len(programs) // max_programs)]  # an even spread: it is a smoke test
    cfg = BacktestConfig(min_answers=50, min_per_class=8, min_per_bin=12)
    pool, _ = pool_from_backtest(w.banks, train, programs, eps_levels, cfg)
    harness = make_harness(w.banks, {a.alert_id: a.binding for a in val}, eps_levels=eps_levels)
    planner = build_planner(system, pool, seed, eps=max(eps_levels) / 2 if len(eps_levels) > 1 else eps_levels[0],
                            max_size=max_size)  # fmt: skip
    base = {"run_id": uuid.uuid4().hex[:8], "commit": "", "machine": "t2", "system": system, "variant": "synthetic",
            "seed": seed}  # fmt: skip
    res = run_alerts(planner, val, harness, limits or Limits(), base)
    res.summary["pool"] = len(pool)
    return res
