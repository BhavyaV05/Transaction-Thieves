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
