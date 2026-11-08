"""scripts/t3_rerun.sh and its helpers (scripts/t3_rerun_tools.py, scripts/t3_rerun_report.py): parsing, rebalance, DRY_RUN command lists."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tools = _load("t3_rerun_tools")


def _rows(n_q: int, n_s: int) -> list[dict]:
    return [{"alert_id": f"A{i}", "step": i, "kind": "ask"} for i in range(n_q)] + [{"alert_id": f"S{i}", "step": 0, "kind": "stop"} for i in range(n_s)]


def test_chunks_cover_the_set_with_a_short_last_chunk():
    assert tools.chunks(1983, 80)[0] == (0, 0)
    assert tools.chunks(1983, 80)[-1] == (1920, 24)
    assert len(tools.chunks(1983, 80)) == 25
    assert tools.chunks(109, 80) == [(0, 0), (80, 1)]
    assert tools.chunks(160, 80) == [(0, 0), (80, 1)]


def test_rebalance_caps_stops_without_upsampling_and_is_deterministic():
    rows = _rows(30, 100)
    out, rep = tools.rebalance(rows, 0.25)
    assert rep["stops_in"] == 100 and rep["stops_kept"] == 10 and rep["questions_or_other"] == 30  # 10 / 40 = 25%
    assert rep["stop_share_out"] == pytest.approx(0.25)
    assert sum(r["kind"] == "stop" for r in out) == 10 and len(out) == 40
    assert out == tools.rebalance(rows, 0.25)[0]
    assert len({(r["alert_id"], r["step"]) for r in out}) == len(out)  # no duplicates


def test_rebalance_keeps_all_stops_when_few():
    out, rep = tools.rebalance(_rows(30, 3), 0.25)
    assert rep["stops_kept"] == 3 and len(out) == 33
    out, rep = tools.rebalance(_rows(0, 0), 0.25)
    assert out == [] and rep["stop_share_out"] == 0.0
    with pytest.raises(ValueError):
        tools.rebalance([], 1.0)


def test_s1_eval_args_reads_the_report_selection():
    sel = {"selection": {"functions": ["worth", "inject"], "tiers": "fast", "per_function": {}}}
    assert tools.s1_eval_args(sel) == ("worth,inject", "fast")
    assert tools.s1_eval_args({"functions": [], "tiers": "cascade"}) == ("", "cascade")


def test_revision_reads_the_serving_profile():
    assert tools.revision("gemma4-12b") == "1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee"
    assert tools.revision("qwen3-8b-lora-v3")  # the v3 student profile exists
    with pytest.raises(KeyError):
        tools.revision("no-such-model")


def test_runlog_system_needs_exactly_one_system(tmp_path):
    for n, s in (("a", "react"), ("b", "react"), ("c", "lats")):
        (tmp_path / n).mkdir()
        pl.DataFrame({"system": [s, s]}).write_parquet(tmp_path / n / "runlog.parquet")
    assert tools.runlog_system([tmp_path / "a", tmp_path / "b"]) == "react"
    with pytest.raises(ValueError):
        tools.runlog_system([tmp_path / "a", tmp_path / "c"])


def test_hours_use_measured_seconds_and_the_ported_run_stands_in_for_adapted(tmp_path):
    for spec, sec in (("ours_compose", 6.0), ("react", 10.0)):
        d = tmp_path / spec / "c000"
        d.mkdir(parents=True)
        (d / "per_alert.jsonl").write_text("".join(json.dumps({"seconds": sec}) + "\n" for _ in range(4)))
    r = tools.hours(tmp_path)
    assert r["react"]["s_per_alert"] == 10.0 and r["react_adapted"]["s_per_alert"] == 10.0
    assert "stands in" in r["react_adapted"]["basis"]
    assert r["lats"]["basis"].startswith("ours_compose")  # no run: the planner's seconds, labelled
    assert r["react"]["c4_h"] == pytest.approx(10.0 * 460 / 3600 / tools.SPEEDUP)
    assert "| **total**" in tools.hours_md(tmp_path)


def test_parse_windows_pairs_start_and_end():
    text = ("2026-11-01T11:03:58+05:30 START e16_runs3 (ff94937)\n2026-11-01T11:38:28+05:30 END   e16_runs3 rc=0\n"
            "2026-11-02T03:40:14+05:30 START gnn (17d001a)\n")
    w = tools.parse_windows(text)
    assert list(w) == ["e16_runs3"] and w["e16_runs3"][0]["hours"] == pytest.approx(34.5 / 60) and w["e16_runs3"][0]["rc"] == 0


def test_report_wrapper_targets_names_that_the_analysis_modules_define():
    rep = _load("t3_rerun_report")
    va = rep._load("t3_val_analysis")
    rep.point_at(va, Path("outputs/x"))
    assert va.EVAL == Path("outputs/x/eval") and va.FIT == [Path("outputs/x/seed/fit"), Path("outputs/x/seed/cal")]
    assert rep.TABLE.exists() and hasattr(va, "CLUSTERS") and hasattr(va, "load_table")
    e16 = rep._load("t3_e16_report")
    assert e16.QWEN == rep.E16_ARMS and hasattr(e16.va, "EVAL")


def _dry(*stages, env=None):
    e = {**os.environ, "DRY_RUN": "1", "PY": sys.executable, "PYTHONPATH": str(ROOT), **(env or {})}
    return subprocess.run(["bash", str(ROOT / "scripts" / "t3_rerun.sh"), *stages], cwd=ROOT, env=e, capture_output=True, text=True, timeout=120)


def test_shell_script_parses():
    assert subprocess.run(["bash", "-n", str(ROOT / "scripts" / "t3_rerun.sh")]).returncode == 0


def test_teacher_dry_run_lists_25_chunks_with_the_tuned_profile_and_no_test():
    r = _dry("teacher", env={"V": "outputs/_t3_rerun_test_v3", "TEACHER": "outputs/_t3_rerun_test_teacher"})  # empty paths: finished real chunks are skipped
    assert r.returncode == 0, r.stderr
    runs = [ln for ln in r.stdout.splitlines() if "run_llm_agent.py" in ln]
    assert len(runs) == 25 + 3 + 2  # teacher chunks, seeding fit (191), seeding cal (109)
    assert all("--profile tuned --concurrency 4" in ln and "--keep-prompts" in ln for ln in runs[:25])
    assert all("--final" not in ln and "--alerts test" not in ln for ln in runs)


def test_teacher_refuses_without_the_prompt_budget_ack():
    e = {k: v for k, v in os.environ.items() if k != "DRY_RUN"}
    r = subprocess.run(["bash", str(ROOT / "scripts" / "t3_rerun.sh"), "teacher"], cwd=ROOT, env={**e, "PROMPTFIT_MERGED": "0"}, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "PROMPTFIT" in r.stdout


def test_val_dry_run_covers_every_system_on_every_part_with_the_right_library():
    r = _dry("val-fitcal", "val-eval", env={"V": "outputs/_t3_rerun_test_v3", "FORCE_EVAL": "1"})
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stdout.splitlines() if "run_llm_agent.py" in ln]
    for spec in ("ours_compose", "ours_library_account", "react_adapted", "dspy_react", "debate", "uot", "bed_llm", "sharma_native", "finfre_rag_ported"):
        for part in ("fit", "cal", "eval"):
            assert any(f"--run-dir outputs/_t3_rerun_test_v3/{part}/{spec}/c000 " in ln + " " for ln in lines), (spec, part)
    fit = [ln for ln in lines if "--val-part fit" in ln or "--val-part cal" in ln]
    ev = [ln for ln in lines if "--val-part eval" in ln]
    assert all("_gemma_train.json" in ln for ln in fit) and all("_gemma.json" in ln and "_train" not in ln.split("--library")[1].split()[0] for ln in ev)
    assert all("--alerts test" not in ln and "--final" not in ln for ln in lines)


def test_val_eval_refuses_before_the_calibration_maps_exist():
    e = {**os.environ, "PY": sys.executable, "V": "outputs/_t3_rerun_test_v3"}
    e.pop("DRY_RUN", None)
    e.pop("FORCE_EVAL", None)
    r = subprocess.run(["bash", str(ROOT / "scripts" / "t3_rerun.sh"), "val-eval"], cwd=ROOT, env=e, capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and "calibration maps missing" in r.stdout


def test_other_stage_dry_runs_name_the_expected_commands():
    out = _dry("distill", "library", "student-train", "e15", env={"V": "outputs/_t3_rerun_test_v3", "SFT": "outputs/_t3_rerun_test_sft"}).stdout
    assert "--filter expected" in out and "--stop-threshold 0.3227" in out and "rebalance" in out
    assert "frozen_v3/shape_gemma_train.json" in out and "frozen_v3/account_gemma.json" in out
    assert out.count("sft_planner_v3.yaml") == 3
    assert "run_injection_tests.py --system planner" in out and "run_injection_tests.py --system react" in out


def test_machine_label_is_overridable_and_defaults_to_the_pc():
    base = {"V": "outputs/_t3_rerun_test_v3", "TEACHER": "outputs/_t3_rerun_test_teacher"}
    pc = _dry("teacher", env=base)
    assert pc.returncode == 0, pc.stderr
    assert all("--machine t3-5070ti" in ln for ln in pc.stdout.splitlines() if "run_llm_agent.py" in ln)
    ada = _dry("teacher", env={**base, "MACHINE": "ada-2080ti-tp2", "EXTERNAL_SERVER": "1"})
    runs = [ln for ln in ada.stdout.splitlines() if "run_llm_agent.py" in ln]
    assert runs and all("--machine ada-2080ti-tp2" in ln and "t3-5070ti" not in ln for ln in runs)
    assert "serve_llm.py" not in ada.stdout  # the external server is used as it is


def test_e15_stage_commands_state_the_backtest_mode():
    """configs/fusion/headroom_v1_table.jsonl has no cache entries beside it, so load_backtest needs the mode: without it every E15 call stopped at startup."""
    r = _dry("e15", env={"V": "outputs/_t3_rerun_test_v3"})
    assert r.returncode == 0, r.stderr
    calls = [ln for ln in r.stdout.splitlines() if "run_injection_tests.py" in ln]
    assert len(calls) == 2 and all("--backtest-mode realistic" in ln for ln in calls)


def _chunk(tmp_path, name, system, calls, probs=None, ptl=0.0, system1=None):
    d = tmp_path / name
    d.mkdir()
    n = len(calls)
    probs = probs if probs is not None else [0.2] * n
    (d / "per_alert.jsonl").write_text("".join(json.dumps({"alert_id": f"A{i}", "llm_calls": c, "probability": p}) + "\n" for i, (c, p) in enumerate(zip(calls, probs, strict=True))))
    (d / "summary.json").write_text(json.dumps({"system": system, "system1": system1}))
    (tmp_path / f"{name}.log").write_text("PromptTooLong[x] A0: over the model's context length\n" * round(ptl * n))
    return d


def test_chunk_check_stops_our_dead_arms(tmp_path):
    cc = _load("t3_chunk_check")
    ok, msg = cc.check(_chunk(tmp_path, "good", "ours_compose", [3, 0, 2, 4, 1, 2, 3, 2, 0, 3]))  # 8 of 10 alerts called the model
    assert ok and "with an LLM call 0.80" in msg
    ok, msg = cc.check(_chunk(tmp_path, "silent", "ours_compose", [0] * 10))
    assert not ok and "FAIL arm=ours_compose rule=share of alerts with an LLM call value=0.00" in msg
    ok, msg = cc.check(_chunk(tmp_path, "toolong", "react_adapted", [1] * 10, ptl=0.9))
    assert not ok and "rule=PromptTooLong per alert value=0.90" in msg
    ok, msg = cc.check(_chunk(tmp_path, "noverdict", "ours_llm_decides", [1] * 10, probs=[None] * 4 + [0.1] * 6))
    assert not ok and "without a verdict value=0.40" in msg
    assert cc.main([str(tmp_path / "good")]) == 0 and cc.main([str(tmp_path / "silent")]) == 4


def test_chunk_check_reports_but_does_not_stop_external_agents_at_their_real_values(tmp_path):
    cc = _load("t3_chunk_check")
    cases = {"lats": dict(probs=[0.2] * 13 + [None] * 7, calls=[2] * 20),  # verdict rate 0.65
             "co_investigator_native": dict(probs=[0.2] * 9 + [None] * 11, calls=[3] * 20),  # 0.46
             "react": dict(probs=[0.2] * 18 + [None] * 2, calls=[3] * 20),  # 0.90
             "reflexion": dict(probs=[0.2] * 19 + [None], calls=[3] * 20),  # 0.95
             "dspy_react": dict(probs=[0.2] * 20, calls=[3] * 20, ptl=0.2)}  # PromptTooLong 0.2 per alert
    for system, kw in cases.items():
        ok, msg = cc.check(_chunk(tmp_path, system, system, **kw))
        assert ok, msg  # reported, not policed
    ok, msg = cc.check(_chunk(tmp_path, "lats2", "lats", [2] * 20, probs=[0.2] * 13 + [None] * 7))
    assert "WARN arm=lats" in msg and "FAIL" not in msg
    ok, msg = cc.check(_chunk(tmp_path, "dead", "sharma_ported", [0] * 10))  # no LLM call at all
    assert not ok and "FAIL arm=sharma_ported rule=no alert made an LLM call" in msg
    ok, msg = cc.check(_chunk(tmp_path, "alltoolong", "react", [1] * 10, ptl=0.9))  # nearly every call too long
    assert not ok and "maximum=0.8" in msg


def test_chunk_check_skips_the_call_share_rule_when_system1_answers(tmp_path):
    cc = _load("t3_chunk_check")
    ok, msg = cc.check(_chunk(tmp_path, "s1", "ours_compose", [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], system1="outputs/s1v1/system1_frozen_s0.json"))  # the fast tier answered
    assert ok, msg
    ok, msg = cc.check(_chunk(tmp_path, "s1bad", "ours_compose", [0] * 10, probs=[None] * 10, system1="x"))  # but no verdicts at all is still a failure
    assert not ok and "without a verdict" in msg


def test_test_alerts_v2_applies_the_frozen_map_and_keeps_rows():
    import importlib.util

    import numpy as np
    import polars as pl

    from ftm.baselines import p0_recal

    spec = importlib.util.spec_from_file_location("rla", ROOT / "scripts" / "run_llm_agent.py")
    rla = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rla)
    v1 = pl.DataFrame({"alert_id": ["F1", "F2", "F3"], "p0": [0.1, 0.5, 0.9]})
    out = rla.test_alerts_v2(v1, "hi_medium")
    m = p0_recal.load_map(ROOT / "configs" / "t1_final" / "p0_recal_hi_medium.json")
    assert out["alert_id"].to_list() == ["F1", "F2", "F3"] and out["p0_v1"].to_list() == [0.1, 0.5, 0.9]
    assert np.allclose(out["p0"].to_numpy(), p0_recal.apply_map(m, np.array([0.1, 0.5, 0.9])))


def test_chunk_check_ignores_recovered_planner_prompt_too_long(tmp_path):
    cc = _load("t3_chunk_check")
    d = _chunk(tmp_path, "ptl", "ours_compose", [1] * 10)
    (tmp_path / "ptl.log").write_text("PromptTooLong[planner] V1: estimate ~7000 > cap 6750\n" * 9)
    ok, _ = cc.check(d)
    assert ok


def test_alert_table_builds_v2_test_alerts_when_the_variant_comes_from_the_environment(tmp_path, monkeypatch):
    """The final run's LLM jobs pass no --p0-variant: the tuned profile sets FTM_P0_VARIANT=v2. The v2 test table does not exist,
    so the frozen map must be applied to T1's v1 table (8 Nov: every job failed on the missing data/alerts/<data>_v2/test.parquet)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("rla_alert_table", ROOT / "scripts" / "run_llm_agent.py")
    rla = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rla)
    (tmp_path / "v1").mkdir()
    v1 = pl.DataFrame({"alert_id": ["F1", "F2"], "tx_id": [1, 2], "bank": [7, 7], "p0": [0.2, 0.7], "label": [0, 1]})
    v1.write_parquet(tmp_path / "v1" / "test.parquet")
    cfg = {"alerts_dir": str(tmp_path / "v1"), "alerts_dir_v2": str(tmp_path / "v2")}
    with pytest.raises(rla.guards.TestSplitLocked):  # outside the final run the test split stays locked
        rla.alert_table(cfg, "hi_medium", "test")
    monkeypatch.setattr(rla.guards, "require_final_run", lambda *a, **k: None)  # stands in for run_final's lock
    monkeypatch.setenv("FTM_P0_VARIANT", "v2")
    out = rla.alert_table(cfg, "hi_medium", "test")
    assert "label" not in out.columns and out["alert_id"].to_list() == ["F1", "F2"]
    assert out["p0_v1"].to_list() == [0.2, 0.7] and out["p0"].to_list() == rla.test_alerts_v2(v1.drop("label"), "hi_medium")["p0"].to_list()
    monkeypatch.setenv("FTM_P0_VARIANT", "v1")
    assert rla.alert_table(cfg, "hi_medium", "test")["p0"].to_list() == [0.2, 0.7]
