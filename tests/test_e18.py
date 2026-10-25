"""T3: E18 backbone sweep: the decision rule on synthetic per-alert results, and the driver end to end on a mock endpoint."""

import functools
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from ftm.llm.devtools import dev_responder
from ftm.llm.mock import MockLLM, start_server
from ftm.planners import e18

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "e18_backbone_sweep.yaml"
N = 400
Y = (np.arange(N) % 10 < 3).astype("f8")  # 30% positives, deterministic
IDS = tuple(f"A{i:04d}" for i in range(N))
CL = np.arange(N) // 2  # clusters of two alerts
NB = 300  # bootstrap resamples: plenty for these separated cases


NOISE = np.random.default_rng(0).normal(0, 0.12, N)  # shared by all arms


def arm(name, active_b, skill, valid=1.0, median_s=12.0, params_b=None, ids=IDS, p=None, jitter=0.0):
    """Probability 0.3 + skill * (y - 0.3) + shared noise: equal skill = identical arms, a larger skill is better.
    `jitter` adds per-arm noise (seeded by the name): a near-identical pair then has a CI that includes 0, where a
    noiseless pair would always look 'clearly worse'."""
    if p is None:
        j = np.random.default_rng(sum(map(ord, name))).normal(0, jitter, N)
        p = np.clip(0.3 + skill * (Y - 0.3) + NOISE + j, 0, 1)
    return e18.Arm(name, active_b, params_b or active_b, ids, Y, p, 0.0, valid, np.full(N, median_s))


def run1(arms, **kw):
    return e18.stage1_select({a.name: a for a in arms}, CL, n_boot=NB, **kw)


# -- stage 1 ----------------------------------------------------------------------------------------------
def test_stage1_drops_invalid_dsl_and_clearly_worse_and_keeps_survivors_by_brier():
    out = run1([arm("a", 8, 0.90), arm("b", 14, 0.90, jitter=0.03), arm("worse", 12, 0.30), arm("badfmt", 20, 0.90, valid=0.94)])
    assert out["best"] == "a"
    assert out["selected"] == ["a", "b"]
    pc = out["per_candidate"]
    assert "Brier CI" in pc["worse"]["dropped"][0] and "valid_dsl" in pc["badfmt"]["dropped"][0]
    assert pc["a"]["ci_vs_best"] == (0.0, 0.0) and pc["b"]["diff_vs_best"] > 0 and pc["b"]["dropped"] == []


def test_stage1_keeps_exactly_95_percent_and_top_k_of_survivors():
    arms = [arm(f"m{i}", 8 + i, 0.90, valid=0.95, jitter=0.02) for i in range(5)]
    out = run1(arms, drop_if_clearly_worse=False)
    by_brier = [a.name for a in sorted(arms, key=lambda a: a.brier)]
    assert out["selected"] == by_brier[:3] and len(out["survivors"]) == 5  # 0.95 is not below 0.95


def test_stage1_clearly_worse_drop_can_be_switched_off_and_nan_rate_fails():
    arms = [arm("a", 8, 0.90), arm("worse", 12, 0.30), arm("silent", 14, 0.80, valid=float("nan"))]
    out = run1(arms, drop_if_clearly_worse=False)
    assert out["selected"] == ["a", "worse"] and out["per_candidate"]["silent"]["dropped"]


def test_stage1_best_is_over_all_candidates_even_one_that_fails_the_dsl_filter():
    # literal reading of prereg/e18_decision.yaml (question 2 for T4): "best model" is not restricted to survivors
    out = run1([arm("best_but_bad_dsl", 8, 0.99, valid=0.5), arm("a", 8, 0.90)])
    assert out["best"] == "best_but_bad_dsl" and out["selected"] == []  # 'a' is clearly worse than it


def test_stage1_requires_the_same_alerts():
    other = arm("b", 14, 0.8, ids=tuple(reversed(IDS)))
    with pytest.raises(e18.ParityError):
        run1([arm("a", 8, 0.9), other])


# -- stage 2 pick -----------------------------------------------------------------------------------------
def pick(arms, margin, **kw):
    return e18.pick({a.name: a for a in arms}, CL, margin, n_boot=NB, **kw)


def test_pick_is_the_smallest_equivalent_model():
    arms = [arm("s8", 8, 0.80), arm("m14", 14, 0.90), arm("l27", 27, 0.90)]
    assert pick(arms, 0.05)["chosen"] == "s8"  # diff 0.0063 < 0.05
    out = pick(arms, 0.001)  # s8 is no longer equivalent; m14 ties l27 and is smaller
    assert out["chosen"] == "m14" and out["reason"] == "smallest_equivalent_to_best"
    assert not out["per_candidate"]["s8"]["equivalent"] and out["best"] == "m14"


def test_pick_equivalence_uses_the_ci_upper_bound_strictly_below_the_margin():
    arms = [arm("s", 8, 0.80), arm("b", 14, 0.90)]
    d = pick(arms, 1.0)["per_candidate"]["s"]["ci_vs_best"][1]
    assert pick(arms, d)["chosen"] == "b"  # upper bound == margin is not below it
    assert pick(arms, d + 1e-9)["chosen"] == "s"


def test_pick_latency_filters_the_equivalent_set_then_falls_back_to_best_meeting_it():
    slow_small = arm("s8", 8, 0.90, median_s=25.0)
    ok_mid = arm("m14", 14, 0.89, median_s=15.0)
    big = arm("l27", 27, 0.90, median_s=18.0)
    assert pick([slow_small, ok_mid, big], 0.05)["chosen"] == "m14"
    out = pick([arm("s8", 8, 0.90, median_s=25.0), arm("m14", 14, 0.60, median_s=15.0)], 0.001)
    assert (out["chosen"], out["reason"]) == ("m14", "fallback_best_meeting_latency")  # best is too slow, none else equivalent
    out = pick([arm("s8", 8, 0.90, median_s=25.0), arm("m14", 14, 0.60, median_s=30.0)], 0.001)
    assert out["chosen"] is None and out["reason"] == "no_candidate_meets_latency"
    assert pick([arm("fast", 8, 0.9, median_s=2.0)], 0.001)["chosen"] == "fast"  # below 8 s is not a failure


def test_pick_ties_by_median_seconds_then_total_parameters():
    a = arm("a", 3.6, 0.9, median_s=14.0, params_b=21)
    b = arm("b", 3.6, 0.9, median_s=10.0, params_b=25)
    c = arm("c", 3.6, 0.9, median_s=10.0, params_b=20)
    assert pick([a, b, c], 0.05)["chosen"] == "c"
    assert pick([a, b], 0.05)["chosen"] == "b"


# -- thinking arm -----------------------------------------------------------------------------------------
def test_thinking_adopted_only_if_clearly_better_by_more_than_the_margin_and_within_latency():
    off = arm("m", 14, 0.70)
    big_gain = arm("m", 14, 0.95, median_s=19.0)
    kw = {"n_boot": NB}
    assert e18.thinking_decision(off, big_gain, CL, 0.01, **kw)["adopt"]
    slow = e18.thinking_decision(off, arm("m", 14, 0.95, median_s=40.0), CL, 0.01, **kw)
    assert not slow["adopt"] and not slow["meets_latency"]
    assert not e18.thinking_decision(off, arm("m", 14, 0.72), CL, 0.01, **kw)["adopt"]  # better, but not by the margin
    assert not e18.thinking_decision(off, arm("m", 14, 0.50), CL, 0.01, **kw)["adopt"]


def test_margin_and_cluster_helpers():
    assert e18.margin_from(brier_random=0.25, brier_best_simple=0.15) == pytest.approx(0.05)
    labels, src = e18.clusters_for(("a", "b", "c"), None)
    assert list(labels) == [0, 1, 2] and "row-level" in src
    labels, _ = e18.clusters_for(("a", "b", "c"), {"a": 7, "b": 9, "c": 7})
    assert labels[0] == labels[2] != labels[1]
    with pytest.raises(ValueError):
        e18.clusters_for(("a", "z"), {"a": 1})


def test_candidates_from_the_real_config():
    cfg = e18.load_config(CONFIG)
    core = e18.candidates(cfg)
    assert [c.name for c in core] == ["qwen3-8b", "gemma4-12b", "qwen3-14b", "gpt-oss-20b", "gemma4-26b-a4b", "qwen3.8-27b-q3"]
    assert len(e18.candidates(cfg, ("core", "if_time"))) == 8
    assert e18.candidates(cfg, only=["gpt-oss-20b"])[0].size_gb_est == (12.1, "config")
    assert e18.candidates(cfg, only=["phi4-reasoning"])[0].size_gb_est == (7.9, "estimate")
    with pytest.raises(ValueError):
        e18.candidates(cfg, only=["nope"])


# -- the driver -------------------------------------------------------------------------------------------
@pytest.fixture
def driver():
    spec = importlib.util.spec_from_file_location("run_e18", ROOT / "scripts" / "run_e18.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_endpoints(tmp_path):
    """Two in-process OpenAI-compatible servers (the mock LLM with the dev responder), a config that points at them."""
    mocks = {
        "qwen3-8b": MockLLM(responder=functools.partial(dev_responder, n_questions=1), model="qwen3-8b"),
        "gpt-oss-20b": MockLLM(responder=functools.partial(dev_responder, n_questions=2), model="gpt-oss-20b"),
    }
    servers = {}
    for name, m in mocks.items():
        servers[name] = start_server(m)
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["candidates"] = [c for c in cfg["candidates"] if c["name"] in mocks]  # the sweep is these two
    for name, (_, url) in servers.items():
        cfg["serving"][name]["endpoint"] = url
    path = tmp_path / "e18.yaml"
    path.write_text(yaml.safe_dump(cfg))
    yield path, mocks
    for srv, _ in servers.values():
        srv.shutdown()


def run(driver, cfg, out, *args):
    return driver.main([*args, "--config", str(cfg), "--out", str(out), "--alerts", "synthetic", "--n-boot", str(NB)])


def test_driver_end_to_end_two_candidates_resume_parity_and_decisions(driver, fake_endpoints, tmp_path):
    cfg, mocks = fake_endpoints
    out = tmp_path / "e18"
    assert run(driver, cfg, out, "stage1", "--n", "8", "--only", "qwen3-8b,gpt-oss-20b") == 0
    s1 = out / "stage1"
    for name in mocks:
        for f in ("runlog.parquet", "per_alert.jsonl", "summary.json", "e18_meta.json"):
            assert (s1 / name / f).exists()
    summ = json.loads((s1 / "summary.json").read_text())
    assert summ["pending"] == [] and {r["candidate"] for r in summ["rows"]} == set(mocks)
    row = {r["candidate"]: r for r in summ["rows"]}
    assert row["qwen3-8b"]["n_alerts"] == 8 and row["qwen3-8b"]["brier"] > 0 and row["qwen3-8b"]["no_verdict_rate"] == 0
    assert row["gpt-oss-20b"]["calls_per_alert"] > row["qwen3-8b"]["calls_per_alert"]  # 2 questions vs 1
    assert row["qwen3-8b"]["stack"] == "vllm" and row["qwen3-8b"]["size_gb"] == 6.0
    assert (s1 / "summary.csv").read_text().startswith("candidate,family")
    assert summ["decision"]["best"] in mocks and (s1 / "decision.json").exists()
    assert summ["latency_counts"] is False  # machine is not the reference machine
    # parity: same alerts, seed and limits in both run dirs; thinking off, gpt-oss at reasoning_effort low
    sm = {n: json.loads((s1 / n / "summary.json").read_text()) for n in mocks}
    assert len({m["alerts_hash"] for m in sm.values()}) == 1 and len({json.dumps(m["limits"]) for m in sm.values()}) == 1
    assert sm["qwen3-8b"]["limits"]["max_tokens"] == 6000 and sm["gpt-oss-20b"]["serving_stack"] == "llama.cpp"
    assert mocks["qwen3-8b"].requests[0]["extra"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert mocks["gpt-oss-20b"].requests[0]["extra"]["reasoning_effort"] == "low"
    assert {r["seed"] for m in mocks.values() for r in m.requests} == {0}
    # resume: a complete run dir is skipped, no further requests
    seen = {n: len(m.requests) for n, m in mocks.items()}
    run(driver, cfg, out, "stage1", "--n", "8", "--only", "qwen3-8b,gpt-oss-20b")
    assert {n: len(m.requests) for n, m in mocks.items()} == seen
    # an incomplete run dir (no summary.json) is rerun
    (s1 / "qwen3-8b" / "summary.json").unlink()
    run(driver, cfg, out, "stage1", "--n", "8", "--only", "qwen3-8b")
    assert len(mocks["qwen3-8b"].requests) > seen["qwen3-8b"] and len(mocks["gpt-oss-20b"].requests) == seen["gpt-oss-20b"]
    # stage 2 on 12 alerts (the 8 stage-1 alerts included); no margin yet, so the pick is pending
    run(driver, cfg, out, "stage2", "--n", "12", "--models", "qwen3-8b,gpt-oss-20b")
    s2 = json.loads((out / "stage2" / "summary.json").read_text())
    assert s2["n_alerts"] == 12 and "margin" in s2["decision"]["pending"]
    run(driver, cfg, out, "stage2", "--n", "12", "--models", "qwen3-8b,gpt-oss-20b", "--margin", "0.5")
    d2 = json.loads((out / "stage2" / "decision.json").read_text())
    assert d2["chosen"] in mocks and d2["margin"] == 0.5
    # thinking arm for the winner: thinking on, 1,024 output tokens per step, same 12 alerts
    winner = d2["chosen"]
    before = len(mocks[winner].requests)
    run(driver, cfg, out, "thinking", "--n", "12", "--margin", "0.5")
    new = mocks[winner].requests[before:]
    assert new and all(r["max_tokens"] == 1024 for r in new)
    want = {"reasoning_effort": "medium"} if winner == "gpt-oss-20b" else {"chat_template_kwargs": {"enable_thinking": True}}
    assert all(r["extra"] == want for r in new)
    dt = json.loads((out / "thinking" / "decision.json").read_text())
    assert dt["model"] == winner and dt["adopt"] in (True, False)


def test_driver_waits_for_all_candidates_before_deciding_and_skips_unserved(driver, fake_endpoints, tmp_path):
    cfg, mocks = fake_endpoints
    out = tmp_path / "e18"
    run(driver, cfg, out, "stage1", "--n", "6", "--only", "qwen3-8b")  # one server at a time
    summ = json.loads((out / "stage1" / "summary.json").read_text())
    assert summ["pending"] == ["gpt-oss-20b"] and summ["decision"] is None and not mocks["gpt-oss-20b"].requests
    assert not (out / "stage1" / "decision.json").exists()
    dead = yaml.safe_load(cfg.read_text())
    dead["serving"]["gpt-oss-20b"]["endpoint"] = "http://127.0.0.1:9/v1"  # nothing listens
    cfg.write_text(yaml.safe_dump(dead))
    run(driver, cfg, out, "stage1", "--n", "6")
    summ = json.loads((out / "stage1" / "summary.json").read_text())
    assert summ["status"]["qwen3-8b"] == "skipped (complete)" and summ["status"]["gpt-oss-20b"].startswith("not served")
    assert summ["decision"] is None
    with pytest.raises(SystemExit, match="not in this stage"):
        run(driver, cfg, out, "stage1", "--only", "qwen3-14b")


def test_driver_refuses_the_test_split_and_unlabelled_val_parts(driver, tmp_path, monkeypatch):
    monkeypatch.setenv("FTM_FINAL_RUN", "1")  # even with the final-run switch set
    with pytest.raises(SystemExit, match="never touches the test split"):
        driver.main(["stage1", "--alerts", "test", "--out", str(tmp_path)])
    monkeypatch.delenv("FTM_FINAL_RUN")
    with pytest.raises(SystemExit):  # --val-part is required with val alerts
        driver.main(["stage1", "--alerts", "val", "--out", str(tmp_path)])
    with pytest.raises(SystemExit):  # random-answer toolbox is never a val result
        driver.main(["stage1", "--alerts", "val", "--val-part", "eval", "--toolbox", "dev", "--out", str(tmp_path)])
    assert not list(tmp_path.iterdir())


def test_driver_refuses_a_draft_prereg_for_real_alerts_but_dry_run_prints_the_plan(driver, tmp_path, capsys):
    if driver.prereg_state()["status"] == "committed":
        pytest.skip("prereg is committed")
    with pytest.raises(SystemExit, match="commit it before stage 1"):
        driver.main(["stage1", "--alerts", "val", "--val-part", "eval", "--out", str(tmp_path)])
    assert driver.main(["stage1", "--alerts", "val", "--val-part", "eval", "--dry-run", "--out", str(tmp_path)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["n_alerts"] == 200 and len(plan["candidates"]) == 6 and plan["estimate"]["total_investigations"] == 1200
    assert plan["estimate"]["gpu_h_low"] == pytest.approx(2.67) and plan["estimate"]["gpu_h_high"] == pytest.approx(6.67)
    gpt = next(c for c in plan["candidates"] if c["name"] == "gpt-oss-20b")
    assert gpt["reasoning_effort"] == "low" and plan["data"] == "hi_medium" and not list(tmp_path.iterdir())


def test_dry_run_estimates_for_the_later_stages(driver, tmp_path, capsys):
    driver.main(["thinking", "--models", "qwen3-14b", "--alerts", "synthetic", "--dry-run", "--out", str(tmp_path)])
    plan = json.loads(capsys.readouterr().out)
    assert plan["thinking"] and plan["n_alerts"] == 460
    assert (plan["estimate"]["gpu_h_low"], plan["estimate"]["gpu_h_high"]) == pytest.approx((4.6, 8.28))
    assert plan["candidates"][0]["max_output_tokens"] == 1024 and plan["candidates"][0]["thinking"]
    driver.main(["stage2", "--models", "qwen3-14b,qwen3-8b,gpt-oss-20b", "--alerts", "synthetic", "--dry-run", "--out", str(tmp_path)])
    est = json.loads(capsys.readouterr().out)["estimate"]
    assert est["total_investigations"] == 1380 and est["gpu_h_low"] == pytest.approx(3.07, abs=0.01) and est["gpu_h_high"] == pytest.approx(7.67, abs=0.01)


def test_parity_check_catches_different_alerts_seed_or_limits():
    a, b = arm("a", 8, 0.9), arm("b", 14, 0.9)
    base = {"seed": 0, "data": "x", "val_part": "eval", "alerts": "val", "system": "ours_compose",
            "limits": {"max_tokens": 6000}, "n_alerts": N, "alerts_hash": "h"}  # fmt: skip
    e18.check_parity({"a": base, "b": dict(base)}, {"a": a, "b": b})
    for k, v in (("seed", 1), ("limits", {"max_tokens": 4000}), ("alerts_hash", "z")):
        with pytest.raises(e18.ParityError, match=k):
            e18.check_parity({"a": base, "b": {**base, k: v}}, {"a": a, "b": b})
    with pytest.raises(e18.ParityError):
        e18.check_parity({"a": base, "b": dict(base)}, {"a": a, "b": arm("b", 14, 0.9, ids=tuple(reversed(IDS)))})


def test_val_runs_forward_the_backtest_table_and_need_it(driver, tmp_path):
    """Without the LR table the planner has no accepted questions and every verdict is p0 (the 2026-10-25 bug)."""
    import argparse

    cfg = e18.load_config(CONFIG)
    cand = e18.candidates(cfg, only=["qwen3-8b"])[0]
    a = argparse.Namespace(alerts="val", val_part="eval", data="hi_medium", backtest=Path("x/lr_table.jsonl"),
                           toolbox="harness", machine="m", endpoint=None)  # fmt: skip
    argv = driver.agent_argv(cfg, cand, a, 5, False, tmp_path / "stage1" / "qwen3-8b")
    assert argv[argv.index("--backtest") + 1] == "x/lr_table.jsonl"
    assert driver.expected_summary(cfg, cand, a, False)["backtest"] == {"path": "x/lr_table.jsonl"}
    assert not e18.run_complete(tmp_path, 5, {"backtest": {"path": "x"}})  # no summary yet
    (tmp_path / "per_alert.jsonl").write_text("")
    (tmp_path / "summary.json").write_text(json.dumps({"n_alerts": 5, "backtest": None}))
    assert not e18.run_complete(tmp_path, 5, {"backtest": {"path": "x"}})  # a run without the table is not complete
    with pytest.raises(SystemExit, match="backtest"):
        driver.main(["stage1", "--val-part", "eval", "--backtest", str(tmp_path / "missing.jsonl"), "--dry-run"])
