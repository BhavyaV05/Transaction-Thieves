"""Case memory built from train alerts (finding 10), stratified draws, configurable p0 source."""

import argparse
import importlib.util
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from ftm.external.memory_build import build_case_memory, load_p0_override

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_llm_agent", ROOT / "scripts" / "run_llm_agent.py")
rla = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rla)


def _train(n=40):
    toy = rla.synthetic_alerts(n, 1, labelled=True)
    return [(replace(c, split="train"), y) for c, y in toy]


def test_memory_takes_train_alerts_and_answers_two_per_class():
    tr = _train()
    mem = build_case_memory([c for c, _ in tr], [y for _, y in tr])
    out = mem.nearest([0.3, 0, 0, 0, 0, 0], k=2)
    assert len(out["closest_known_laundering"]) == 2 and len(out["closest_known_clean"]) == 2
    with pytest.raises(ValueError):
        build_case_memory([replace(c, split="val") for c, _ in tr], [y for _, y in tr])
    with pytest.raises(ValueError, match="both classes"):
        build_case_memory([c for c, y in tr if y == 0], [0 for _, y in tr if y == 0])


def test_p0_override_replaces_the_p0_feature(tmp_path):
    tr = _train(10)
    ids = [c.alert_id for c, _ in tr]
    f = tmp_path / "p0.parquet"
    pl.DataFrame({"alert_id": ids, "p0v2": [0.01 * i for i in range(10)]}).write_parquet(f)
    ov = load_p0_override(f, "p0v2")
    mem = build_case_memory([c for c, _ in tr], [max(y, i % 2) for i, (_, y) in enumerate(tr)], ov)
    assert sorted(r["x"][0] for r in mem.rows) == pytest.approx([0.01 * i for i in range(10)])


def test_case_memory_factory_and_pool_stratification():
    a = argparse.Namespace(alerts="synthetic", seed=0, k=40, data="x", memory_n=60, memory_pos_frac=0.5, memory_p0=None, memory_p0_col="p0")
    assert rla.case_memory(a).rows
    df = pl.DataFrame({"label": [1] * 30 + [0] * 270, "alert_id": [str(i) for i in range(300)]})
    out = rla.stratified_order(df, 0, 40, 0.5)
    assert out["label"].sum() == 20 and out.height == 40
    assert rla.stratified_order(df, 0, 40, 0.5)["alert_id"].to_list() == out["alert_id"].to_list()  # seeded
    dfb = df.with_columns(pl.col("label").cast(pl.Boolean))  # the real train alerts store the label as Boolean (polars 2.0 raises on Boolean == int)
    assert rla.stratified_order(dfb, 0, 40, 0.5)["alert_id"].to_list() == out["alert_id"].to_list()
