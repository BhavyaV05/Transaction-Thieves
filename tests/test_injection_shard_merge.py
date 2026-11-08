"""scripts/ada/merge_injection_shards.py: two shards of one E15 run merge into the report's format, and the rows are summarize() of all pairs."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("merge_injection_shards", ROOT / "scripts" / "ada" / "merge_injection_shards.py")
mis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mis)

import run_injection_tests as rit  # noqa: E402

from ftm.safety.injection_run import summarize  # noqa: E402


def _run(out, run_id, n):
    argv = ["--system", "planner", "--n", str(n), "--arms", "nogate,gate", "--llm", "mock-obedient", "--out", str(out), "--run-id", run_id, "--min-exposed-share", "0"]
    assert rit.main(argv) == 0


def test_two_shards_merge_into_the_unsharded_format(tmp_path):
    _run(tmp_path, "s1", 2)
    s2 = json.loads((tmp_path / "s1" / "planner" / "summary.json").read_text())
    # a second shard with other alert ids: relabel the first one's pairs (disjoint ids) so the merge has two real inputs
    (tmp_path / "s2" / "planner").mkdir(parents=True)
    pairs = [json.loads(ln) for ln in (tmp_path / "s1" / "planner" / "pairs.jsonl").read_text().splitlines()]
    for p in pairs:
        p["alert_id"] = p["alert_id"] + "_b"
    (tmp_path / "s2" / "planner" / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))
    (tmp_path / "s2" / "planner" / "summary.json").write_text(json.dumps(s2))
    (tmp_path / "s2" / "planner" / "runlog.parquet").write_bytes((tmp_path / "s1" / "planner" / "runlog.parquet").read_bytes())
    top = mis.merge(tmp_path, "merged", "planner", ["s1", "s2"], 0.0)
    allp = [json.loads(ln) for s in ("s1", "s2") for ln in (tmp_path / s / "planner" / "pairs.jsonl").read_text().splitlines()]
    assert top["rows"] == summarize(allp, 0.0) and top["n_alerts"] == 2 * s2["n_alerts"]
    assert (tmp_path / "merged" / "planner" / "pairs.jsonl").exists() and (tmp_path / "merged" / "summary.json").exists()
    merged = json.loads((tmp_path / "merged" / "planner" / "summary.json").read_text())
    assert merged["merged_from_shards"] == ["s1", "s2"] and merged["n_pairs"] == len(allp)


def test_overlapping_shards_are_refused(tmp_path):
    _run(tmp_path, "s1", 2)
    import shutil

    shutil.copytree(tmp_path / "s1", tmp_path / "s2")
    with pytest.raises(SystemExit):
        mis.merge(tmp_path, "merged", "planner", ["s1", "s2"], 0.0)
