"""E18 refresh (2 Nov): serving profiles, the fixed selection rule and the run-dir loader of scripts/t3_e18_refresh_report.py."""

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("t3_e18_refresh_report", ROOT / "scripts" / "t3_e18_refresh_report.py")
rr = importlib.util.module_from_spec(spec)
sys.modules["t3_e18_refresh_report"] = rr
spec.loader.exec_module(rr)

REFRESH = ["gemma4-12b", "qwen35-9b", "qwen35-4b", "gemma4-e4b", "qwen3-14b", "gpt-oss-20b", "phi4-14b"]


@pytest.mark.parametrize("name", REFRESH)
def test_profiles_pinned(name):
    repo, rev, cache = rr.profile(name)
    assert re.fullmatch(r"[0-9a-f]{40}", rev) and cache == "models--" + repo.replace("/", "--")
    assert name in rr.SIZE_B


def test_unknown_profile():
    with pytest.raises(SystemExit):
        rr.profile("nope")


def arm(name, p, y, valid=1.0, s=5.0):
    return rr.Arm(name, tuple(f"V{i:03d}" for i in range(len(y))), np.array(y, "f8"), np.array(p, "f8"), valid, s, s, 100.0, 0.0)


def world(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(float)
    good = np.clip(0.15 + 0.7 * y + rng.normal(0, 0.1, n), 0.01, 0.99)  # informative
    return y, good, rng


def test_select_picks_smaller_equivalent_and_requires_eig():
    y, good, rng = world()
    cl = np.arange(len(y))
    eig = arm(rr.EIG, np.full(len(y), 0.3), y)
    arms = {
        "qwen35-4b": arm("qwen35-4b", np.clip(good + rng.normal(0, 0.05, len(y)), 0.01, 0.99), y, s=3.0),  # as good as the best up to noise, smaller
        "gemma4-12b": arm("gemma4-12b", good, y, s=8.0),              # best
        "phi4-14b": arm("phi4-14b", np.full(len(y), 0.9), y),         # clearly worse than EIG
        "qwen35-9b": arm("qwen35-9b", good, y, valid=0.9),            # ineligible: valid-DSL < 0.95
    }
    res = rr.select(arms, eig, cl, n_boot=300)
    assert res["models"]["phi4-14b"]["vs_eig_verdict"] == "worse" and not res["models"]["phi4-14b"]["useful"]
    assert not res["models"]["qwen35-9b"]["eligible"]
    assert res["best"] == "gemma4-12b" and set(res["equivalent"]) == {"gemma4-12b", "qwen35-4b"}
    assert res["pick"] == "qwen35-4b"
    assert "PICK" in rr.table(res)


def test_select_none_useful_keeps_eig():
    y, good, _ = world()
    eig = arm(rr.EIG, good, y)
    res = rr.select({"qwen35-4b": arm("qwen35-4b", np.full(len(y), 0.9), y)}, eig, np.arange(len(y)), n_boot=200)
    assert res["pick"] is None and res["best"] is None and "none" in rr.table(res)


def test_select_clearly_better_excludes_worse_equivalent():
    y, good, rng = world()
    eig = arm(rr.EIG, np.full(len(y), 0.3), y)
    mid = np.clip(0.3 + 0.35 * (good - 0.3), 0.01, 0.99)  # shrunk toward the base rate: worse than `good`, still better than EIG
    arms = {"qwen35-4b": arm("qwen35-4b", mid, y), "gemma4-12b": arm("gemma4-12b", good, y)}
    res = rr.select(arms, eig, np.arange(len(y)), n_boot=300)
    assert res["best"] == "gemma4-12b" and res["pick"] == "gemma4-12b" and res["equivalent"] == ["gemma4-12b"]


def test_select_rejects_mismatched_alerts():
    y, good, _ = world(50)
    with pytest.raises(ValueError):
        rr.select({"a": arm("a", good, y[::-1].copy())}, arm(rr.EIG, good, y), np.arange(50), n_boot=50)


def write_run(root, name, part, chunk, rows):
    d = root / name / part / chunk
    d.mkdir(parents=True)
    (d / "summary.json").write_text("{}")
    (d / "per_alert.jsonl").write_text("\n".join(json.dumps(r) for r in rows))


def row(i, p, label):
    return {"alert_id": f"V{i:03d}", "label": label, "p0": 0.4, "probability": p, "seconds": 2.0, "gpu_seconds": 1.0, "tokens_out": 50,
            "outputs": 2, "json_ok": 2, "questions": 1, "dsl_ok": 1, "stops": 1}


def test_load_arm_pools_chunks_and_scores_no_verdict_as_p0(tmp_path):
    write_run(tmp_path, "m", "fit", "c000", [row(0, 0.9, 1), row(1, None, 0)])
    write_run(tmp_path, "m", "cal", "c000", [row(2, 0.2, 0)])
    (tmp_path / "m" / "cal" / "c001").mkdir()  # unfinished chunk (no summary.json) is ignored
    a = rr.load_arm(tmp_path, "m")
    assert a.ids == ("V000", "V001", "V002") and list(a.p) == [0.9, 0.4, 0.2]
    assert abs(a.no_verdict - 1 / 3) < 1e-12 and a.valid_dsl == 1.0 and a.median_s == 2.0
    assert rr.load_arm(tmp_path, "absent") is None


def test_metric_set_and_full_report():
    y, good, rng = world(150)
    thr = 0.3227
    m = rr.metric_set(y, good, thr)
    assert 0.5 < m["auc"] <= 1 and 0 <= m["recall"] <= 1 and 0 <= m["escalation_rate"] <= 1 and m["catch_top10"] > 0.2
    assert abs(m["brier"] - float(np.mean((good - y) ** 2))) < 1e-12 and m["log_loss"] > 0
    eig = arm(rr.EIG, np.full(len(y), 0.3), y)
    eig.p0 = np.full(len(y), 0.3)
    a1 = arm("gemma4-12b", good, y)
    a2 = arm("qwen35-4b", np.clip(good + rng.normal(0, 0.05, len(y)), 0.01, 0.99), y)
    for a in (a1, a2):
        a.resources = {k: 1.0 for k in ("questions", "eps_spent", "llm_calls", "tokens_in", "tokens_out", "seconds", "gpu_seconds", "duplicates", "refusals")}
    rep = rr.full_report({"gemma4-12b": a1, "qwen35-4b": a2}, eig, np.arange(len(y)), thr, 100)
    d = rep["vs_eig"]["gemma4-12b"]["brier"]
    assert d["diff"] < 0 and d["ci"][1] < 0 and 0 <= d["p_holm"] <= 1 and d["holm_reject"]
    assert "qwen35-4b" in rep["vs_incumbent"] and "gemma4-12b" not in rep["vs_incumbent"]
    assert set(rep["flips_vs_p0"]) == {"gemma4-12b", "qwen35-4b"}
    assert "| gemma4-12b |" in rr.full_table(rep)
