"""The production profile (ftm/llm/profile.py, configs/llm/tuned.yaml): the evidence-view rule, the decision threshold, run_llm_agent wiring."""

import json
from pathlib import Path

import pytest

from ftm.llm.profile import decision_threshold, load_profile, pick_evidence_view

ROOT = Path(__file__).resolve().parents[1]


def test_evidence_view_rule_rederives_the_profile_choice():
    rep = json.loads((ROOT / "configs/llm/evidence_view_fitcal_v2.json").read_text())
    assert rep["parts"] == ["fit", "cal"]  # never eval
    assert pick_evidence_view(rep) == load_profile("tuned")["evidence_view"] == "scorer_pcst"


def _rep(**rec):
    base = {"one_time_respecting_trail": 1.0, "max_tokens_est": 3000.0, "mean_tokens_est": 1000.0}
    return {"views": [{"view": k, **base, "recall_at_k": v} for k, v in rec.items()]}


def test_rule_floor_and_ties():
    assert pick_evidence_view(_rep(amount_1hop=0.80, scorer_pcst=0.70, scorer_greedy=0.60)) == "amount_1hop"  # scorer worse: reference
    assert pick_evidence_view(_rep(amount_1hop=0.80, scorer_pcst=0.85, scorer_greedy=0.78)) == "scorer_pcst"
    r = _rep(amount_1hop=0.80, scorer_pcst=0.85, scorer_greedy=0.85)
    r["views"][2]["mean_tokens_est"] = 900.0
    assert pick_evidence_view(r) == "scorer_greedy"  # tie on recall: fewer tokens
    r = _rep(amount_1hop=0.80, scorer_pcst=0.9, scorer_greedy=0.9)
    for v in r["views"][1:]:
        v["max_tokens_est"] = 5000.0  # over the prompt budget
    assert pick_evidence_view(r) == "amount_1hop"


def test_decision_threshold_and_profiles():
    assert decision_threshold("configs/decision/headroom_v1.yaml") == pytest.approx(0.322699)
    with pytest.raises(ValueError):
        decision_threshold("configs/decision/default.yaml")  # no fitted threshold
    assert load_profile("none") == {}
    with pytest.raises(ValueError):
        load_profile("bogus")
    t = load_profile("tuned")
    assert t["p0_variant"] == "v2" and t["edge_scorer"] == "v2" and t["backtest_mode"] == "realistic" and t["concurrency"] == 8
    for k in ("backtest", "fusion_params", "decision"):
        assert (ROOT / t[k]).exists()


def test_run_llm_agent_defaults_to_the_production_profile(tmp_path, monkeypatch):
    for k in ("FTM_EDGE_SCORER", "FTM_P0_VARIANT"):  # main() exports the profile's choice; keep it out of other tests
        monkeypatch.setenv(k, "")
        monkeypatch.delenv(k)
    import importlib.util

    spec = importlib.util.spec_from_file_location("rla", ROOT / "scripts/run_llm_agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "o"
    mod.main(["--system", "planner", "--n", "3", "--toolbox", "harness", "--out", str(out), "--run-id", "r", "--synthetic-labels"])
    s = json.loads((out / "r/summary.json").read_text())
    assert s["profile"] == "tuned" and s["fusion"] == "ConditionalLR" and s["threshold"] == pytest.approx(0.322699)
    assert s["stop_cannot_cross"] and s["evidence_view"] == "scorer_pcst" and s["edge_scorer"] == "v2" and s["concurrency"]["concurrency"] == 8
    # the naive fusion is reached only by naming it
    mod.main(["--system", "planner", "--n", "2", "--toolbox", "harness", "--out", str(out), "--run-id", "n", "--fusion", "naive", "--synthetic-labels"])
    assert json.loads((out / "n/summary.json").read_text())["fusion"] == "NaiveLRFusion"
    # an explicit table without its fitted fusion is refused (no silent fallback to naive)
    with pytest.raises(SystemExit):
        mod.main(["--toolbox", "harness", "--backtest", "configs/fusion/headroom_v1_table.jsonl", "--n", "1", "--out", str(out)])
