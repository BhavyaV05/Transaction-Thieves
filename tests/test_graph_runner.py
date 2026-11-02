"""Concurrent alert runner (ftm/graph/runner.py): rows identical to sequential, per-worker sessions, real overlap of LLM waits."""

import time
from dataclasses import replace

from ftm.graph.runner import Worker, lock_world, run_alerts, share_session_toolbox
from ftm.harness import make_toolbox
from ftm.llm.mock import MockLLM
from ftm.planners.llm import LLMPlanner
from ftm.types import AlertContext, Limits

Q = "passed_on(ALERT.receiver, window=after(24), min_frac=0.8)"
Q2 = "at_least(n_counterparties(ALERT.receiver, dir=in, window=before(48)), k=5)"


def step(program):
    import json

    return json.dumps({"hypothesis": {"typology": "fan_in", "rationale": "r"}, "action": "compose", "question_id": None,
                       "program": program, "bank": 1, "eps": 0.5, "rule_id": None})


def alerts(n):
    return [AlertContext(f"A{i:03d}", "70", "2022-09-05T12:00:00", "val", 0.1 + 0.05 * i,
                         {"sender": "S", "receiver": f"R{i}", "evidence": "e", "partner_banks": [1, 2, 3]}) for i in range(n)]


def responder(messages, kw):  # stateless: the reply depends on the prompt only (answers so far), as a real model's does
    text = messages[-1]["content"]
    return step(Q if Q not in text else Q2) if Q2 not in text else '{"hypothesis": {"typology": "none", "rationale": "r"}, "action": "stop"}'


def setup(c, latency=0.0):
    llm = MockLLM(responder=responder, latency_s=latency)
    tools = make_toolbox(seed=3, mode="mock")
    workers = [Worker(LLMPlanner(llm), tools, tools)]
    if c > 1:
        lock_world(tools)
        for _ in range(c - 1):
            sess = share_session_toolbox(tools)
            workers.append(Worker(LLMPlanner(llm), sess, sess))
    return workers


def run_one(wk, item):
    wk.session.begin(item, Limits())
    try:
        v = wk.system.investigate(item, wk.tools, Limits())
    finally:
        log = wk.session.end()
    return replace(v, questions_log=log), wk.system.last_trace


def sig(runs):
    return [(r.result[0].probability, r.result[0].questions_log, r.result[1]) for r in runs]


def test_concurrent_rows_equal_sequential_and_in_order():
    items = alerts(12)
    seq, _ = run_alerts(items, setup(1), run_one, concurrency=1)
    con, info = run_alerts(items, setup(4), run_one, concurrency=4, timing_sample=3)
    assert sig(seq) == sig(con) and [r.index for r in con] == list(range(12))
    assert info["concurrency"] == 4 and info["timing_sample"] == 3 and len({r.worker for r in con}) > 1
    assert any(len(r.result[0].questions_log) for r in seq)


def test_llm_waits_overlap():
    items = alerts(8)
    t0 = time.perf_counter()
    run_alerts(items, setup(1, 0.03), run_one, concurrency=1)
    t_seq = time.perf_counter() - t0
    t0 = time.perf_counter()
    run_alerts(items, setup(4, 0.03), run_one, concurrency=4)
    t_con = time.perf_counter() - t0
    assert t_con < t_seq * 0.6, (t_seq, t_con)


def test_worker_exception_propagates():
    import pytest

    def boom(wk, item):
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        run_alerts(alerts(3), setup(2), boom, concurrency=2)


def test_run_script_concurrency_4_equals_1(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    import polars as pl

    spec = importlib.util.spec_from_file_location("run_llm_agent", Path(__file__).resolve().parents[1] / "scripts" / "run_llm_agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    outs = {}
    for c in (1, 4):
        d = tmp_path / f"c{c}"
        assert mod.main(["--system", "planner", "--alerts", "synthetic", "--n", "6", "--llm", "mock", "--toolbox", "harness",
                         "--concurrency", str(c), "--timing-sample", "2", "--profile", "none", "--run-dir", str(d), "--run-id", "t", "--blind"]) == 0
        outs[c] = d
    a, b = (pl.read_parquet(outs[c] / "runlog.parquet").drop("commit") for c in (1, 4))
    assert a.equals(b) and a.height > 6
    assert (outs[1] / "traces.jsonl").read_text() == (outs[4] / "traces.jsonl").read_text()
    s1, s4 = (json.loads((outs[c] / "summary.json").read_text()) for c in (1, 4))
    assert s1["concurrency"]["concurrency"] == 1 and s4["concurrency"]["concurrency"] == 4 and s1["planner"]["outputs"] == s4["planner"]["outputs"]


def test_fusion_params_conditional_loads(tmp_path):
    import importlib.util

    from ftm.fusion.conditional import ConditionalFusion
    from ftm.llm.fusion_adapter import ConditionalLR

    m = ConditionalFusion(families=["passed_on"], w=__import__("numpy").array([0.5]))
    path = tmp_path / "cf.json"
    m.to_json(path)
    assert isinstance(ConditionalLR.from_json(path), ConditionalLR)
    assert importlib.util.find_spec("ftm") is not None
