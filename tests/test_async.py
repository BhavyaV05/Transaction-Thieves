"""Week 8: question waves (batched mode, E13), the SimPy runner, sequential vs batched."""

import math
from dataclasses import replace

import pytest

from ftm.backtest.synth import make_world
from ftm.banks.binding import alert_context
from ftm.banks.partner import Question
from ftm.compliance.gate import BudgetRefused, GateBlocked
from ftm.dsl.parser import canonical, parse
from ftm.harness.interface import BatchToolBox
from ftm.orchestrator.async_study import batching_study
from ftm.orchestrator.runner import build_planner, make_harness, run_alerts
from ftm.orchestrator.simpy_run import run_simpy
from ftm.planners.pool import Candidate, QuestionPool
from ftm.types import Decline, Limits, Reply, Timeout

PROGS = {
    "returns_home(ALERT.receiver, window=after(72))": "returns_home",
    "passed_on(ALERT.receiver, window=after(24), min_frac=0.5)": "passed_on",
    "dormant(ALERT.receiver, days=7)": "dormant",
}


def pool(eps=0.5):
    cells = {"lr": {"true": 2.0, "false": 0.7}, "p": {"true": (0.3, 0.6), "false": (0.7, 0.4)}}
    return QuestionPool(Candidate(canonical(parse(p)), eps, fam, "bool", "receiver", cells["lr"], (), cells["p"], 1)
                        for p, fam in PROGS.items())  # fmt: skip


@pytest.fixture(scope="module")
def world():
    w = make_world(120, seed=5)
    alerts = sorted(w.alerts, key=lambda a: a.binding.t0)
    return w, [replace(a, p0=0.2, split="val") for a in alerts[60:]]


def session(w, a, limits=None, **kw):
    h = make_harness(w.banks, {a.alert_id: a.binding}, eps_levels=(0.25, 0.5, 1.0, 2.0), **kw)
    t = h.toolbox()
    t.begin(alert_context(a.binding, 0.2), limits or Limits())
    return h, t


def reqs(a, n=3, eps=0.5):
    return [(a.binding.receiver_bank, p, eps) for p in list(PROGS)[:n]]


def test_a_wave_costs_its_slowest_answer_not_the_sum(world):
    w, val = world
    a = val[0]
    h, t = session(w, a)
    assert isinstance(t, BatchToolBox)
    res = t.ask_banks(reqs(a))
    replies = [r for r in res if isinstance(r, (Reply, Decline, Timeout))]
    waits = [r.latency_days if isinstance(r, Reply) else getattr(r, "after_days", 0.0) for r in replies]
    assert len(res) == 3 and t.elapsed_days == pytest.approx(max(waits)) and t.elapsed_days < sum(waits) or len(waits) == 1
    # sequentially the same questions take their sum
    h2, t2 = session(w, a)
    seq = [t2.ask_bank(*q) for q in reqs(a)]
    seq_wait = sum(r.latency_days for r in seq if isinstance(r, Reply))
    assert t2.elapsed_days == pytest.approx(seq_wait) and t2.elapsed_days > t.elapsed_days


def test_every_question_in_a_wave_is_asked_at_the_same_moment_and_charged_alone(world):
    w, val = world
    a = val[1]
    h, t = session(w, a)
    res = t.ask_banks(reqs(a, 2, 0.5))
    for (bank, prog, eps), r in zip(reqs(a, 2, 0.5), res, strict=True):
        if isinstance(r, Reply):
            assert r == w.banks[bank].handle(Question(canonical(parse(prog)), eps, a.binding), a.binding.t0)
    assert t.eps_spent == pytest.approx(0.5 * sum(isinstance(r, Reply) for r in res))
    assert [e.step for e in t.questions_log] == [1, 2] and all(e.gate_decision == "allowed" for e in t.questions_log)
    assert h.accountant.alert_spent[a.alert_id] == pytest.approx(t.eps_spent)


def test_a_refusal_inside_a_wave_is_returned_not_raised_and_the_rest_still_run(world):
    w, val = world
    a = val[2]
    h, t = session(w, a, Limits(eps_per_alert=1.0))
    bank = a.binding.receiver_bank
    res = t.ask_banks([(bank, "nope(", 0.5), (bank, list(PROGS)[0], 0.5), (bank, list(PROGS)[1], 0.5), (bank, list(PROGS)[2], 0.5)])
    assert isinstance(res[0], GateBlocked)  # blocked: does not parse
    answered = [r for r in res[1:] if isinstance(r, Reply)]
    over = [r for r in res[1:] if isinstance(r, BudgetRefused)]
    assert len(answered) + len(over) + sum(not isinstance(r, (Reply, BudgetRefused)) for r in res[1:]) == 3
    assert t.eps_spent <= 1.0 + 1e-9  # the run limit binds inside a wave too
    decisions = [e.gate_decision for e in t.questions_log]
    assert decisions[0] == "blocked" and set(decisions) <= {"blocked", "allowed", "over_budget"}


def test_a_wave_is_cut_at_the_deadline(world):
    w, val = world
    a = val[3]
    h, t = session(w, a, Limits(max_sim_time_days=0.01))
    res = t.ask_banks(reqs(a, 2))
    assert t.elapsed_days <= 0.01 + 1e-9
    assert all(not isinstance(r, Reply) for r in res)  # nothing arrives before the deadline


def test_planners_ask_in_waves_and_the_wave_saves_time(world):
    w, val = world
    h = lambda: make_harness(w.banks, {a.alert_id: a.binding for a in val}, eps_levels=(0.25, 0.5, 1.0, 2.0))  # noqa: E731
    base = {"system": "random", "run_id": "x", "variant": "t", "seed": 0, "commit": "", "machine": ""}
    seq = run_alerts(build_planner("random", pool(), waves=None), val, h(), Limits(), base)
    bat = run_alerts(build_planner("random", pool(), waves=(2, 1)), val, h(), Limits(), base)
    assert all(len(v.questions_log) <= 3 for v in bat.verdicts) and seq.summary["mean_questions"] > 2.5
    first_wave = [v.questions_log[:2] for v in bat.verdicts if len(v.questions_log) >= 2]
    assert first_wave and all(len({e.program for e in fw}) == 2 for fw in first_wave)  # two different questions per wave
    assert bat.summary["mean_sim_days"] < seq.summary["mean_sim_days"] - 0.1
    assert bat.summary["p95_sim_days"] <= seq.summary["p95_sim_days"] + 1e-9
    assert bat.summary["mean_eps"] <= 2.0 + 1e-9 and seq.summary["mean_eps"] <= 2.0 + 1e-9


def test_waves_never_ask_two_questions_of_one_family_and_respect_the_eps_left(world):
    w, val = world
    one_family = QuestionPool(replace(c, family="same") for c in pool(1.0).candidates)
    base = {"system": "eig", "run_id": "x", "variant": "t", "seed": 0, "commit": "", "machine": ""}
    r = run_alerts(build_planner("eig", one_family, waves=(2, 1)), val[:10],
                   make_harness(w.banks, {a.alert_id: a.binding for a in val}, eps_levels=(1.0,)), Limits(), base)
    # a declined question releases nothing, so its family may be tried again; two ANSWERED questions of one family never
    assert all(sum(isinstance(e.response, Reply) for e in v.questions_log) <= 1 for v in r.verdicts)
    assert all(len(v.questions_log) <= 2 for v in r.verdicts)
    tight = run_alerts(build_planner("random", pool(1.0), waves=(3,)), val[:10],
                       make_harness(w.banks, {a.alert_id: a.binding for a in val}, eps_levels=(1.0,)), Limits(eps_per_alert=2.0), base)
    assert all(sum(1 for e in v.questions_log if e.gate_decision == "allowed") <= 2 for v in tight.verdicts)  # 3 x 1.0 > 2.0


def test_simpy_runs_one_process_per_alert_on_a_shared_clock(world):
    w, val = world
    h = make_harness(w.banks, {a.alert_id: a.binding for a in val}, eps_levels=(0.25, 0.5, 1.0, 2.0))
    res = run_simpy(build_planner("random", pool()), val[:30], h, Limits(), arrival_gap_days=0.05)
    assert len(res.verdicts) == 30 and set(res.finished) == {a.alert_id for a in val[:30]}
    for i, a in enumerate(val[:30]):
        assert res.started[a.alert_id] == pytest.approx(i * 0.05)
        assert res.finished[a.alert_id] == pytest.approx(res.started[a.alert_id] + res.latency_days[a.alert_id])
    assert res.max_in_flight > 1  # latencies of about a day against an arrival every 0.05: many are open together
    assert res.makespan_days == pytest.approx(max(res.finished.values())) and res.makespan_days >= 29 * 0.05
    # slow arrivals: the investigations no longer overlap
    h2 = make_harness(w.banks, {a.alert_id: a.binding for a in val}, eps_levels=(0.25, 0.5, 1.0, 2.0))
    slow = run_simpy(build_planner("random", pool()), val[:10], h2, Limits(), arrival_gap_days=20.0)
    assert slow.max_in_flight == 1


def test_simpy_gives_the_same_answers_as_the_plain_runner(world):
    w, val = world
    mk = lambda: make_harness(w.banks, {a.alert_id: a.binding for a in val}, eps_levels=(0.25, 0.5, 1.0, 2.0))  # noqa: E731
    base = {"system": "random", "run_id": "x", "variant": "t", "seed": 0, "commit": "", "machine": ""}
    plain = run_alerts(build_planner("random", pool()), val[:20], mk(), Limits(), base)
    sim = run_simpy(build_planner("random", pool()), val[:20], mk(), Limits())
    by = {v.alert_id: [(e.program, e.eps, e.response) for e in v.questions_log] for v in sim.verdicts}
    assert by == {v.alert_id: [(e.program, e.eps, e.response) for e in v.questions_log] for v in plain.verdicts}


def test_batching_study_reports_both_arms_with_the_same_budget(world):
    w, val = world
    rows = batching_study("random", w.banks, val[:40], pool(), seed=1)
    seq, bat = rows
    assert (seq["arm"], bat["arm"]) == ("sequential", "batched") and bat["waves"] == [2, 1] and seq["waves"] is None
    assert bat["mean_sim_days"] < seq["mean_sim_days"] and bat["makespan_days"] > 0 and seq["alerts"] == bat["alerts"] == 40
    assert abs(bat["mean_questions"] - seq["mean_questions"]) < 0.5 and math.isfinite(bat["brier_final"])  # same menu, same budget
    assert abs(seq["brier_p0"] - bat["brier_p0"]) < 1e-12  # the same alerts
