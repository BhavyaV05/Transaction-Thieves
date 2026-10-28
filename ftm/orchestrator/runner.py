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
from ftm.planners.learned import LearnedPolicyPlanner
from ftm.planners.pool import QuestionPool
from ftm.planners.simple import EIGPlanner, EnumeratorPlanner, FixedOrderPlanner, RandomPlanner
from ftm.privacy.accountant import PrivacyAccountant
from ftm.types import Limits

SYSTEMS = ("random", "fixed", "eig", "enumerator")  # need nothing but the pool
ALL_SYSTEMS = (*SYSTEMS, "learned")  # the learned policy also needs a fitted `LearnedPolicy` (scripts/train_policy.py)


def build_planner(system: str, pool: QuestionPool, seed: int = 0, eps: float = 1.0, threshold: float | None = None,
                  ranking: Sequence[str] | None = None, max_size: int = 4, waves: tuple[int, ...] | None = None,
                  policy=None, fusion=None, one_per_family: bool = True):  # fmt: skip
    """`waves`: batched mode (E13), e.g. (2, 1) = two questions asked together, then one more. `policy`: the fitted
    `LearnedPolicy` for system "learned". `fusion`: T4's fitted `Fusion` (None: summed log-LRs). `one_per_family=False`
    lets a template be asked more than once per alert (an ablation)."""
    kw = {"threshold": threshold, "waves": waves, "fusion": fusion, "one_per_family": one_per_family}
    if system == "learned":
        if policy is None:
            raise ValueError("system 'learned' needs policy= (train it with scripts/train_policy.py)")
        return LearnedPolicyPlanner(pool, policy=policy, **kw)
    if system == "random":
        return RandomPlanner(pool, seed=seed, **kw)
    if system == "fixed":
        return FixedOrderPlanner(pool, ranking=tuple(ranking or expert_ranking()), eps=eps, **kw)
    if system == "eig":
        return EIGPlanner(pool, **kw)
    if system == "enumerator":
        return EnumeratorPlanner(pool, max_size=max_size, **kw)
    raise ValueError(f"system must be one of {ALL_SYSTEMS}")


@dataclass
class RunResult:
    rows: list[dict]
    summary: dict
    verdicts: list = field(default_factory=list)


def run_alerts(planner, alerts: Sequence[BtAlert], harness: Harness, limits: Limits, base: dict,
               split: str = "val", blind: bool = False) -> RunResult:  # fmt: skip
    """Investigate every alert in a fresh harness session and score the verdicts against the labels. The run-log rows
    are v1: p0, eps spent, simulated days and the stop reason on the verdict row; family, kind and log LR per answer.

    `blind` is the final run (G5): no label goes into a row and the summary carries no Brier score, so nothing here can
    print or write a test result. The analysis joins the labels in later."""
    tools = harness.toolbox()
    system = tools.wrap(planner)
    rows, verdicts, p0s, ps, ys, secs, sims = [], [], [], [], [], [], []
    for a in alerts:
        ctx = alert_context(a.binding, a.p0, split)
        t = time.perf_counter()
        v = system.investigate(ctx, tools, limits)
        secs.append(time.perf_counter() - t)
        sims.append(tools.last_elapsed_days)
        facts = {"eps_spent": tools.eps_spent, "sim_days": tools.last_elapsed_days,
                 "stop_reason": getattr(planner, "last_stop_reason", None)}  # fmt: skip
        rows += verdict_rows(v, {**base, "split": split, "alert_id": a.alert_id}, None if blind else int(a.label), ctx=ctx,
                             pool=getattr(planner, "pool", None), facts=facts)  # fmt: skip
        verdicts.append(v)
        p0s.append(a.p0)
        ps.append(a.p0 if v.probability is None else v.probability)
        ys.append(float("nan") if blind else float(a.label))
    p0s, ps, ys = np.array(p0s), np.array(ps), np.array(ys)
    n_q = [sum(e.gate_decision == "allowed" and e.response is not None for e in v.questions_log) for v in verdicts]
    summary = {
        "system": base["system"], "alerts": len(alerts), "positives": None if blind else int(ys.sum()),
        "brier_p0": None if blind else float(np.mean((p0s - ys) ** 2)),
        "brier_final": None if blind else float(np.mean((ps - ys) ** 2)),
        "mean_questions": float(np.mean(n_q)) if n_q else 0.0,
        "mean_eps": float(np.mean([sum(e.eps for e in v.questions_log if hasattr(e.response, "eps_spent"))
                                   for v in verdicts])) if verdicts else 0.0,
        "mean_seconds_per_alert": float(np.mean(secs)) if secs else 0.0,
        "mean_sim_days": float(np.mean(sims)) if sims else 0.0,
        "p95_sim_days": float(np.percentile(sims, 95)) if sims else 0.0,
        "eps_cal": None,
    }  # fmt: skip
    summary["brier_gain"] = None if blind else summary["brier_p0"] - summary["brier_final"]
    return RunResult(rows, summary, verdicts)


def make_harness(world_banks, bindings, accountant: PrivacyAccountant | None = None, eps_levels=None,
                 allow_exact: bool = False) -> Harness:  # fmt: skip
    acc = accountant or PrivacyAccountant()
    gate = Gate.from_config(validator=gate_validator, accountant=acc)
    if eps_levels:
        gate.eps_levels = tuple(eps_levels)
    return Harness(world_banks, gate, acc, bindings, allow_exact=allow_exact)


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
