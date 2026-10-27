"""T4: pre-registration files parse and agree with plan.md section 2 and 9.3."""

from pathlib import Path

import yaml

PREREG = Path("prereg")


def _load(name):
    return yaml.safe_load((PREREG / name).read_text())


def test_hypotheses_tiers_and_holm_families():
    h = _load("hypotheses.yaml")
    hyp = h["hypotheses"]
    corr = h["statistics"]["correction"]
    assert corr["primary"]["family"] == ["H1", "H2", "H3", "H4", "H9"]
    assert corr["primary"]["method"] == "holm"
    for tier in ("primary", "secondary", "exploratory"):
        assert set(corr[tier]["family"]) == {k for k, v in hyp.items() if v["tier"] == tier}
    assert h["statistics"]["bootstrap"]["resamples"] == 10_000
    assert hyp["H1"]["threshold"] == 0.30 and hyp["H2"]["threshold"] == 0.70 and hyp["H4"]["threshold_fdr"] == 0.10


def test_every_hypothesis_names_an_experiment():
    for name, v in _load("hypotheses.yaml")["hypotheses"].items():
        assert v.get("experiment", "").startswith("E"), name


def test_e18_rule_matches_run_config():
    rule = _load("e18_decision.yaml")
    assert rule["stage1"]["alerts"] == 160 and rule["stage2"]["alerts"] == 460  # amended 2026-10-25: the askable HI-Medium val alerts
    assert rule["status"] == "committed" and rule["stage1"]["val_part"] == "eval" and rule["stage2"]["val_part"] == "all"
    run_cfg = Path("configs/experiments/e18_backbone_sweep.yaml")
    if run_cfg.exists():  # the run config lands with the plan-sync PR (#9); once both exist they must agree
        run = yaml.safe_load(run_cfg.read_text())
        assert run["stage1"]["alerts"] == rule["stage1"]["alerts"]
        assert run["stage2"]["alerts"] == rule["stage2"]["alerts"]


def test_external_list_core_systems():
    ext = _load("external_list.yaml")
    core = {s["name"] for s in ext["systems"] if s["tier"] == "core"}
    assert {"multi_gnn", "dp_fedavg_gnn", "torres", "sharma", "react", "reflexion", "rewoo"} <= core
    assert ext["backbone"] == "e18"
    for s in ext["systems"]:
        if s["group"] == "C":
            assert s["modes"] == ["native", "ported"], s["name"]
