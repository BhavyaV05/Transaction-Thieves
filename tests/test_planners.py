"""C9: pool, EIG, the four non-LLM planners end to end, the enumerator's search space, run-log rows, the CLI."""

import importlib.util
import json
import math
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ftm.backtest.engine import BacktestConfig
from ftm.backtest.synth import make_world
from ftm.banks.binding import alert_context
from ftm.dsl.enumerate import DEFAULT_GRID, enumerate_programs
from ftm.dsl.executor import ExecContext, execute
from ftm.dsl.expert import expert_programs, expert_ranking, load_specs
from ftm.dsl.parser import canonical, family, parse, size
from ftm.eval.runlog import RUN_LOG_COLUMNS
from ftm.harness.runlog_rows import to_frame, verdict_rows, write_runlog
from ftm.orchestrator.runner import (
    build_planner,
    make_harness,
    pool_from_backtest,
    run_alerts,
    synthetic_experiment,
)
from ftm.planners.pool import Candidate, QuestionPool
from ftm.planners.simple import EIGPlanner, FixedOrderPlanner, RandomPlanner, expected_brier_gain
from ftm.privacy.accountant import PrivacyAccountant
from ftm.types import AlertContext, Limits

PROGS = [
    "n_counterparties(ALERT.receiver, dir=in, window=before(72))",
    "n_tx(ALERT.receiver, dir=in, window=around(72), max_amount=2000)",
    "passed_on(ALERT.receiver, window=after(24), min_frac=0.5)",
    "returns_home(ALERT.receiver, window=after(72))",
    "dormant(ALERT.receiver, days=7)",
]


def cand(lr, p_clean, p_launder, eps=1.0, fam="f", prog="p(ALERT.receiver)", slot="receiver"):
    cells = [f"c{i}" for i in range(len(lr))]
    return Candidate(prog, eps, fam, "kary", slot, dict(zip(cells, lr, strict=True)), (),
                     {c: (a, b) for c, a, b in zip(cells, p_clean, p_launder, strict=True)}, 1)  # fmt: skip


# --- expected information gain ------------------------------------------------------------------------------------
def test_eig_is_zero_for_a_useless_question_and_positive_for_a_useful_one():
    useless = cand([1.0, 1.0], [0.5, 0.5], [0.5, 0.5])
    useful = cand([0.4286, 2.3333], [0.7, 0.3], [0.3, 0.7])
    assert expected_brier_gain(0.2, useless) == pytest.approx(0.0, abs=1e-12)
    assert 0 < expected_brier_gain(0.2, useful) < 0.2 * 0.8
    assert expected_brier_gain(0.0, useful) == 0 == expected_brier_gain(1.0, useful)
    assert expected_brier_gain(0.2, cand([1.0], [1.0], [1.0])) == pytest.approx(0.0, abs=1e-12)


def test_eig_grows_with_separation_and_a_calibrated_question_is_a_martingale():
    weak = cand([0.5 / 0.55, 0.5 / 0.45], [0.55, 0.45], [0.5, 0.5])
    strong = cand([0.1, 9.0], [0.9, 0.1], [0.09, 0.91])
    assert 0 < expected_brier_gain(0.3, weak) < expected_brier_gain(0.3, strong)
    # with lr = P(a | launder) / P(a | clean) exactly, the expected posterior equals the prior
    p, lr = 0.3, [0.09 / 0.9, 0.91 / 0.1]
    mean_post = sum(
        (p * pl_ + (1 - p) * pc) / 1 * (1 / (1 + math.exp(-(math.log(p / (1 - p)) + math.log(l_)))))
        for l_, pc, pl_ in zip(lr, [0.9, 0.1], [0.09, 0.91], strict=True)
    )
    assert mean_post == pytest.approx(p, abs=1e-3)
    assert expected_brier_gain(0.3, replace(strong, p_cell=None)) == 0.0  # no measured cell probabilities: no estimate


# --- the pool ------------------------------------------------------------------------------------------------------
def rows():
    base = {"eps": 1.0, "family": "n_counterparties:in", "bins": [2.0], "lr": {"b0": 0.5, "b1": 2.0},
            "p_cell": {"b0": [0.7, 0.3], "b1": [0.3, 0.7]}}  # fmt: skip
    return [
        {**base, "program_canonical": "n_counterparties(ALERT.receiver, dir=in, window=before(72))", "accepted": True},
        {**base, "program_canonical": "n_counterparties(ALERT.sender, dir=in, window=before(72))", "accepted": True},
        {**base, "program_canonical": "n_counterparties(ALERT.receiver, dir=out, window=after(24))", "accepted": False},
        {**base, "program_canonical": "set_contacts(ALERT.payers, dir=in, window=before(72))", "accepted": True,
         "family": "set_contacts:in"},
    ]  # fmt: skip


def test_pool_keeps_accepted_questions_and_filters_by_alert_direction(tmp_path):
    pool = QuestionPool.from_rows(rows())
    assert len(pool) == 3 and len(QuestionPool.from_rows(rows(), only_accepted=False)) == 4
    out = {c.program.split("(")[0] + ":" + (c.slot or "-") for c in pool.applicable("out")}
    assert out == {"n_counterparties:receiver", "set_contacts:-"}
    assert {c.slot for c in pool.applicable("in")} == {"sender", None}
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows()))
    assert [c.program for c in QuestionPool.from_jsonl(path)] == [c.program for c in pool.candidates]
    c = pool.candidates[0]
    assert c.cell(1) == "b0" and c.cell(5) == "b1" and c.log_lr(5) == pytest.approx(math.log(2.0))
    assert c.log_lr("zzz") is None


# --- planners on the synthetic world --------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def results():
    return {s: synthetic_experiment(s, 360, 3, programs=PROGS) for s in ("random", "fixed", "eig")}


@pytest.mark.parametrize("system", ["random", "fixed", "eig"])
def test_every_planner_improves_the_brier_score_inside_the_budget(results, system):
    r = results[system]
    s = r.summary
    assert s["alerts"] > 100 and s["pool"] >= 1
    assert s["brier_final"] < s["brier_p0"] - 0.02 and s["mean_questions"] >= 1
    for v in r.verdicts:
        spent = sum(e.response.eps_spent for e in v.questions_log if hasattr(e.response, "eps_spent"))
        assert spent <= 2.0 + 1e-9 and all(e.gate_decision == "allowed" for e in v.questions_log)
        assert len({(e.program, e.bank) for e in v.questions_log}) == len(v.questions_log)  # never the same question twice


def test_runs_are_reproducible():
    a = synthetic_experiment("eig", 300, 5, programs=PROGS[:3])
    b = synthetic_experiment("eig", 300, 5, programs=PROGS[:3])
    strip = lambda r: [{k: v for k, v in x.items() if k != "run_id"} for x in r.rows]  # noqa: E731
    assert strip(a) == strip(b) and a.summary["brier_final"] == b.summary["brier_final"]


def test_the_enumerator_runs_on_the_enumerated_search_space():
    space = enumerate_programs(3, ("receiver",))
    assert len(space) > 200
    progs = [canonical(n) for n in space if n.name == "n_counterparties" and n.arg("dir") == "in"]
    assert 5 <= len(progs) <= 12
    r = synthetic_experiment("enumerator", 300, 4, eps_levels=(2.0,), programs=progs)
    assert r.summary["system"] == "enumerator" and r.summary["pool"] >= 1
    assert r.summary["brier_final"] < r.summary["brier_p0"]


def make_env(n=240, seed=9):
    w = make_world(n, seed)
    alerts = sorted(w.alerts, key=lambda a: a.binding.t0)
    train, val = alerts[:150], alerts[150:]
    val = [replace(a, p0=0.2) for a in val]
    cfg = BacktestConfig(min_answers=50, min_per_class=8, min_per_bin=12)
    pool, _ = pool_from_backtest(w.banks, train, PROGS[:2], (1.0, 2.0), cfg)
    return w, val, pool


def test_the_stop_rule_and_threshold():
    w, val, pool = make_env()
    h = make_harness(w.banks, {a.alert_id: a.binding for a in val}, eps_levels=(1.0, 2.0))
    lim = Limits()
    base = {"system": "eig", "run_id": "x", "variant": "t", "seed": 0, "commit": "", "machine": ""}
    # p0 = 0.2 against a threshold of 0.99: no answer budget (2 eps, shift <= 2) can reach it, so it stops at once
    far = run_alerts(build_planner("eig", pool, threshold=0.999), val[:20], h, lim, base)
    assert far.summary["mean_questions"] == 0
    near = run_alerts(build_planner("eig", pool, threshold=0.3), val[:20], h, lim, base)
    assert near.summary["mean_questions"] > 0


def test_a_refused_question_does_not_crash_or_loop():
    w, val, pool = make_env()
    acc = PrivacyAccountant(eps_per_alert=0.25)  # nothing in the pool fits (eps 1 and 2)
    h = make_harness(w.banks, {a.alert_id: a.binding for a in val}, acc, eps_levels=(1.0, 2.0))
    base = {"system": "random", "run_id": "x", "variant": "t", "seed": 0, "commit": "", "machine": ""}
    r = run_alerts(build_planner("random", pool), val[:10], h, Limits(), base)
    assert r.summary["mean_questions"] == 0 and all(v.probability == pytest.approx(0.2) for v in r.verdicts)
    assert all(e.gate_decision == "over_budget" for v in r.verdicts for e in v.questions_log)
    assert not acc.alert_spent


def test_fixed_order_follows_the_ranking_and_the_eps_level():
    a = cand([0.5, 2.0], [0.7, 0.3], [0.3, 0.7], 1.0, "alpha", "a(ALERT.receiver)")
    b = cand([0.5, 2.0], [0.7, 0.3], [0.3, 0.7], 1.0, "beta", "b(ALERT.receiver)")
    b2 = cand([0.5, 2.0], [0.7, 0.3], [0.3, 0.7], 2.0, "beta", "b(ALERT.receiver)")
    ctx = AlertContext("A", "1", datetime(2022, 9, 5).isoformat(), "val", 0.2,
                       {"direction": "out", "counterparty_bank": 2})  # fmt: skip
    from ftm.planners.session import PlanSession

    s = PlanSession(ctx, None, Limits())
    assert FixedOrderPlanner(QuestionPool([a, b, b2]), ranking=("beta", "alpha"), eps=1.0).choose(s, [a, b, b2]) is b
    assert FixedOrderPlanner(QuestionPool([a, b, b2]), ranking=("beta", "alpha"), eps=2.0).choose(s, [a, b, b2]) is b2
    assert FixedOrderPlanner(QuestionPool([a, b]), ranking=("alpha", "beta"), eps=1.0).choose(s, [a, b]) is a
    assert FixedOrderPlanner(QuestionPool([a, b]), eps=0.5).choose(s, [a, b]) is not None  # no such level: best affordable


def test_random_planner_is_reproducible_per_alert_whatever_the_order():
    w, val, pool = make_env()
    h = make_harness(w.banks, {a.alert_id: a.binding for a in val}, eps_levels=(1.0, 2.0))
    base = {"system": "random", "run_id": "x", "variant": "t", "seed": 0, "commit": "", "machine": ""}
    fwd = run_alerts(RandomPlanner(pool, seed=1), val[:8], make_harness(w.banks, h.bindings, eps_levels=(1.0, 2.0)), Limits(), base)
    rev = run_alerts(RandomPlanner(pool, seed=1), val[:8][::-1], make_harness(w.banks, h.bindings, eps_levels=(1.0, 2.0)), Limits(), base)
    by = lambda r: {v.alert_id: [(e.program, e.eps) for e in v.questions_log] for v in r.verdicts}  # noqa: E731
    assert by(fwd) == by(rev)


def test_eig_picks_the_most_informative_question_per_eps():
    strong = cand([0.1, 9.0], [0.9, 0.1], [0.09, 0.91], 1.0, "strong", "s(ALERT.receiver)")
    weak = cand([0.9, 1.15], [0.55, 0.45], [0.5, 0.5], 1.0, "weak", "w(ALERT.receiver)")
    ctx = AlertContext("A", "1", "2022-09-05T00:00:00", "val", 0.3, {"direction": "out", "counterparty_bank": 2})
    from ftm.planners.session import PlanSession

    s = PlanSession(ctx, None, Limits())
    pl_ = EIGPlanner(QuestionPool([strong, weak]))
    assert pl_.choose(s, [weak, strong]) is strong
    assert pl_.choose(s, []) is None
    assert pl_.choose(s, [cand([1.0], [1.0], [1.0])]) is None  # nothing is expected to help: stop asking


def test_planners_need_the_direction_and_counterparty_in_the_alert_summary():
    from ftm.planners.session import PlanSession

    with pytest.raises(KeyError, match="direction"):
        PlanSession(AlertContext("A", "1", "2022-09-05T00:00:00", "val", 0.1, {}), None, Limits())


def test_alert_context_is_aliased_and_gives_the_direction():
    w = make_world(20, seed=2)
    b = w.alerts[0].binding
    ctx = alert_context(b, 0.2)
    assert ctx.summary["direction"] == "out" and ctx.summary["counterparty_bank"] == b.receiver_bank
    assert set(ctx.summary) == {"direction", "counterparty_bank", "partner_banks", "amount", "currency"}
    assert ctx.as_of == b.t0.isoformat() and ctx.home_bank == str(b.home_bank)  # no node ids, no payer or payee sets


# --- run log -------------------------------------------------------------------------------------------------------
def test_run_log_rows_follow_the_schema(results, tmp_path):
    r = results["eig"]
    frame = to_frame(r.rows)
    assert frame.columns == list(RUN_LOG_COLUMNS) and frame.height == len(r.rows)
    assert set(frame["event"]) == {"step", "verdict"} and frame.filter(pl.col("event") == "verdict").height == len(r.verdicts)
    v = frame.filter(pl.col("event") == "verdict")
    assert v["label"].null_count() == 0 and v["probability"].null_count() == 0
    path = write_runlog(r.rows, tmp_path / "x" / "runlog.parquet")
    assert pl.read_parquet(path).equals(frame)
    steps = frame.filter(pl.col("event") == "step")
    assert set(steps["gate"]) <= {"allowed", "blocked", "over_budget"}
    assert steps.filter(pl.col("response_kind") == "reply")["answer_value"].null_count() == 0


def test_verdict_rows_keep_refusals_as_steps():
    w, val, pool = make_env()
    acc = PrivacyAccountant(eps_per_alert=0.25)
    h = make_harness(w.banks, {a.alert_id: a.binding for a in val}, acc, eps_levels=(1.0, 2.0))
    base = {"system": "random", "run_id": "x", "variant": "t", "seed": 0, "commit": "", "machine": ""}
    r = run_alerts(build_planner("random", pool), val[:3], h, Limits(), base)
    steps = [x for x in r.rows if x["event"] == "step"]
    assert steps and all(x["gate"] == "over_budget" and x["response_kind"] is None for x in steps)
    assert verdict_rows(r.verdicts[0], {"alert_id": "a"})[-1]["event"] == "verdict"


def test_run_log_rows_fill_the_v1_fields_scoring_needs():
    from ftm.eval.runlog import RUN_LOG_COLUMNS, validate

    w, val, pool = make_env()
    h = make_harness(w.banks, {a.alert_id: a.binding for a in val}, eps_levels=(1.0, 2.0))
    base = {"system": "random", "run_id": "x", "variant": "t", "seed": 0, "commit": "", "machine": ""}
    r = run_alerts(build_planner("random", pool), val[:6], h, Limits(), base)
    frame = to_frame(r.rows)
    assert not validate(frame.columns, set(frame["event"]))
    v = frame.filter(pl.col("event") == "verdict")
    for c in ("home_bank", "p0", "eps_spent", "sim_days", "stop_reason", "trials"):
        assert v[c].null_count() == 0, c
    assert set(v["stop_reason"]) <= {"budget", "fast_clear", "cannot_cross", "finish"}
    assert (v["trials"] == 1).all()
    asked = frame.filter((pl.col("event") == "step") & (pl.col("response_kind") == "reply"))
    assert len(asked) and asked["family"].null_count() == 0 and asked["kind"].null_count() == 0
    assert asked["log_lr"].null_count() == 0  # the pool only holds accepted questions, so every answer has an LR
    assert set(RUN_LOG_COLUMNS) == set(frame.columns)


# --- the enumerator's search space ---------------------------------------------------------------------------------
def test_enumeration_respects_the_size_bound_and_is_canonical_and_stable():
    for n in (2, 3, 4):
        ps = enumerate_programs(n, ("receiver",))
        assert ps and all(size(p) <= n for p in ps)
        cs = [canonical(p) for p in ps]
        assert cs == sorted(set(cs)) and all(canonical(parse(c)) == c for c in cs)
    assert len(enumerate_programs(2)) < len(enumerate_programs(3)) < len(enumerate_programs(4)) < 2000
    assert enumerate_programs(3) == enumerate_programs(3)


def test_enumeration_covers_the_language():
    names = {p.name for p in enumerate_programs(4, ("receiver", "sender"))}
    assert {"n_counterparties", "n_tx", "passed_on", "overlap", "set_contacts", "at_least", "time_to_forward"} <= names
    assert "entity_accounts" not in names and "top_outflow_bank" not in names  # restricted / needs bank sets
    banks = {p.name for p in enumerate_programs(4, bank_sets=[[1, 2], [2, 3]])}
    assert "top_outflow_bank" in banks
    only_sender = enumerate_programs(3, ("sender",))
    assert all("ALERT.receiver" not in canonical(p) for p in only_sender)
    with pytest.raises(ValueError):
        enumerate_programs(3, ("payers",))
    assert all(k in DEFAULT_GRID for k in ("hours", "k", "days"))


def test_every_enumerated_program_executes():
    w = make_world(40, seed=3)
    a = w.alerts[0].binding
    led = w.ledgers[a.receiver_bank]
    ctx = ExecContext(a.receiver_bank, a.t0, a.t0 + timedelta(days=2), a.amount, a.currency, a.home_bank, a.sender,
                      a.receiver, a.payers, a.payees, {n: n % 3 for n in range(2000)})  # fmt: skip
    ps = enumerate_programs(3, ("receiver", "sender"), bank_sets=[[1, 2, 3]])
    assert len(ps) > 300
    for p in ps:
        v = execute(p, led, ctx)
        assert isinstance(v, (bool, int)), canonical(p)


def test_expert_ranking_follows_the_spec_order():
    rk = expert_ranking()
    assert len(rk) == len(set(rk)) and rk[0] == family(parse(expert_programs()[0]["program"]))
    assert load_specs()[0]["family"] == "pass_through" and rk[0] == "passed_on"


# --- the CLI ---------------------------------------------------------------------------------------------------------
def test_run_experiment_cli_synthetic(tmp_path):
    spec = importlib.util.spec_from_file_location("run_exp", Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    summary = cli.main(["--system", "random", "--synthetic", "240", "--out", str(tmp_path), "--seed", "2"])
    (run,) = tmp_path.iterdir()
    log = pl.read_parquet(run / "runlog.parquet")
    assert log.columns == list(RUN_LOG_COLUMNS) and set(log["variant"]) == {"synthetic"}
    assert json.loads((run / "summary.json").read_text())["alerts"] == summary["alerts"]
    assert np.isfinite(summary["brier_final"])
