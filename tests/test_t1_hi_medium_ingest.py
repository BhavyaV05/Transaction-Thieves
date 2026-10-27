"""T1: the pieces that must pick up Akshat's HI-Medium see-everything val files and the FraudGT val file with no code change, and the frozen subset rule."""

import json
import sys
from pathlib import Path

import polars as pl
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import baselines_val_table as bvt  # noqa: E402
import make_t1_final_configs as gen  # noqa: E402


def _gnn_json(model):
    hist = [{"epoch": 0, "fit_ap": 0.4}, {"epoch": 1, "fit_ap": 0.6}]
    g = {"history": hist, "best_fit_ap": 0.6, "calibration": "isotonic", "train_edges": 10, "train_pos": 3}
    return {"dataset": "hi_medium", "mode": "see_everything", "commit": "abc", "seed": 1, "batch_size": 2048, "neighbours": [100, 100], "val_cut": ["a", "b"], "graphs": {"all": g}, "skipped": {}}


def _gfp_json():
    g = {"status": "ok", "params": {"num_leaves": 31}, "grid": [{"num_leaves": 31, "val_ap": 0.5, "best_iteration": 200}], "calibration": "isotonic", "train_pos": 5, "val_fit_pos": 4}
    return {"dataset": "hi_medium", "mode": "see_everything", "commit": "abc", "seed": 0, "val_cut": ["a", "b"], "graphs": {"all": g}}


def test_generator_marks_hi_medium_see_everything_done_once_the_val_json_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gen, "RUNS", tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    assert gen.gnn("multi_gnn", "hi_medium", "see_everything")["status"] == "blocked_compute"
    assert gen.gfp("hi_medium", "see_everything")["status"] == "blocked_compute"
    for model in ("multi_gnn", "mega_gnn"):
        (tmp_path / "runs" / f"p0_hi_medium_{model}_see_everything.json").write_text(json.dumps(_gnn_json(model)))
        m = gen.gnn(model, "hi_medium", "see_everything")
        assert m["status"] == "val_done" and m["models"]["best_epoch_by_val_fit_ap"] == 1 and m["inputs"][-1] == f"data/scores/hi_medium/p0_{model}__see_everything.parquet"
    (tmp_path / "runs" / "p0_hi_medium_see_everything.json").write_text(json.dumps(_gfp_json()))
    g = gen.gfp("hi_medium", "see_everything")
    assert g["status"] == "val_done" and g["models"]["best_iteration"] == 200


def test_generator_never_rewrites_the_frozen_subset_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "OUT", tmp_path)
    gen.dump("subset_rule.yaml", "# h\n", {"code_commit": "aaa"}, keep=True)
    gen.dump("subset_rule.yaml", "# h\n", {"code_commit": "bbb"}, keep=True)  # a later run of the generator
    assert yaml.safe_load((tmp_path / "subset_rule.yaml").read_text())["code_commit"] == "aaa"
    gen.dump("other.yaml", "# h\n", {"x": 1})
    gen.dump("other.yaml", "# h\n", {"x": 2})  # every other config is regenerated
    assert yaml.safe_load((tmp_path / "other.yaml").read_text())["x"] == 2


def test_the_committed_subset_rule_is_written_with_keep():
    src = (Path(gen.__file__)).read_text()
    assert 'dump("subset_rule.yaml"' in src and "keep=True)" in src.split('dump("subset_rule.yaml"')[1]


def test_val_table_registry_picks_up_see_everything_and_fraudgt_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "data" / "scores" / "hi_medium"
    d.mkdir(parents=True)
    one = pl.DataFrame({"mode": ["one_bank"]})
    one.write_parquet(d / "p0_multi_gnn.parquet")  # the combined file written after the one-bank run
    assert "Multi-GNN see-everything" not in bvt.registry("hi_medium")
    pl.DataFrame({"mode": ["see_everything"]}).write_parquet(d / "p0_multi_gnn__see_everything.parquet")  # arrives from the 5070 Ti
    pl.DataFrame({"mode": ["see_everything"]}).write_parquet(d / "p0_mega_gnn__see_everything.parquet")
    pl.DataFrame({"mode": ["see_everything"]}).write_parquet(d / "p0_see_everything.parquet")
    pl.DataFrame({"mode": ["see_everything"]}).write_parquet(d / "p0_fraudgt.parquet")
    reg = bvt.registry("hi_medium")
    for name in ("Multi-GNN one-bank", "Multi-GNN see-everything", "MEGA-GNN (MEGA-GenAgg) see-everything", "GFP see-everything", "FraudGT see-everything"):
        assert name in reg, name
    assert reg["FraudGT see-everything"]["model"] == "fraudgt" and reg["FraudGT see-everything"]["kind"] == "tx"


def test_compare_reads_the_per_mode_multi_gnn_files(tmp_path):
    pytest.importorskip("torch")
    import train_gnn as tg

    pl.DataFrame({"mode": ["one_bank"], "tx_id": [1]}).write_parquet(tmp_path / "p0_multi_gnn__one_bank.parquet")
    pl.DataFrame({"mode": ["see_everything"], "tx_id": [1]}).write_parquet(tmp_path / "p0_multi_gnn__see_everything.parquet")
    pl.DataFrame({"mode": ["one_bank"], "tx_id": [1]}).write_parquet(tmp_path / "p0_multi_gnn.parquet")  # stale: one-bank only
    assert sorted(tg.read_multi_gnn(tmp_path)["mode"].to_list()) == ["one_bank", "see_everything"]


def test_generator_turns_fraudgt_to_val_done_once_its_val_json_exists():
    doc = {"status": "blocked_compute", "final_entry": {"status": "needs_val_run", "commands": ["python scripts/fraudgt_final.py"]}}
    assert gen.fraudgt(None, dict(doc))["status"] == "blocked_compute"
    val = {"commit": "abc", "dump": "d", "unscored_share": 0.0025, "calibration": "isotonic", "pred_meta": {"best_epoch": 3, "best_val_f1": 0.6}, "eval": {"n": 5, "n_pos": 1, "brier": 0.001, "pr_auc": 0.5, "f1": 0.6}}
    out = gen.fraudgt(val, {**doc, "final_entry": dict(doc["final_entry"])})
    assert out["status"] == "val_done" and out["final_entry"]["status"] == "ready" and out["val_result"]["best_epoch"] == 3 and out["scores_file"].endswith("p0_fraudgt.parquet")
