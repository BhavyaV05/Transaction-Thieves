"""T1 Multi-GNN our-split driver: no test rows into the graph, p0 table schema for model=multi_gnn. Synthetic, no GPU."""

import importlib.util
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from test_baselines_gfp import _frame

from ftm.baselines import multi_gnn


def test_edge_arrays_refuse_test_rows_and_keep_int64_ids(monkeypatch):
    from ftm import guards

    monkeypatch.delenv("FTM_FINAL_RUN", raising=False)
    f = _frame(with_test=True).sort(["timestamp", "tx_id"])
    with pytest.raises(guards.TestSplitLocked):
        multi_gnn.edge_arrays(f)
    ei, ea, y, tx, n_nodes, codes = multi_gnn.edge_arrays(f.filter(pl.col("split") != "test"))
    assert ei.dtype == np.int64 and ei.max() < n_nodes and ea.shape == (len(y), 4) and tx.dtype == np.int64
    assert ea[0, 0] == 0 and (np.diff(ea[:, 0]) >= 0).all()  # seconds since the first row


def test_p0_table_schema_matches_gfp(monkeypatch):
    pytest.importorskip("sklearn")
    spec = importlib.util.spec_from_file_location("train_gnn", Path(__file__).resolve().parents[1] / "scripts" / "train_gnn.py")
    tg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tg)
    f = _frame().sort(["timestamp", "tx_id"])
    v = f.filter(pl.col("split") == "val")
    rng = np.random.default_rng(0)
    raw = np.clip(0.02 + 0.3 * v["is_laundering"].to_numpy() * rng.random(v.height) + 0.01 * rng.random(v.height), 0, 1)
    raw[:3] = np.nan  # seed edges that were not sampled stay null
    monkeypatch.setattr(multi_gnn, "train_and_score", lambda *a, **k: {"raw": raw, "tx_id": v["tx_id"].to_numpy(), "history": []})
    cut = (v["timestamp"].min(), v["timestamp"].max())
    tab, meta = tg.run_graph(f, cut, "see_everything", None, 1.0, 1, Path("."), "abc1234", 0, None)
    assert tab.columns == ["tx_id", "bank", "mode", "model", "split", "val_part", "p0", "score_raw", "label", "seed", "commit"]
    assert set(tab["model"]) == {"multi_gnn"} and set(tab["split"]) == {"val"} and set(tab["val_part"]) == {"fit", "cal", "eval"}
    assert tab["p0"].null_count() == 3 and tab["bank"].null_count() == tab.height
    assert tab["p0"].drop_nulls().is_between(0, 1).all() and meta["calibration"] in {"platt", "isotonic"}


def test_final_run_edge_arrays_keep_the_codes_and_the_val_features(monkeypatch):
    """Appending test rows (final run) must not shift the category codes or any feature of an earlier row."""
    f = _frame(with_test=True).sort(["timestamp", "tx_id"])
    tv = f.filter(pl.col("split") != "test")
    ei0, ea0, y0, tx0, n0, codes0 = multi_gnn.edge_arrays(tv)
    # a new currency that sorts before every known one appears only in the test rows
    f = f.with_columns(pl.when(pl.col("split") == "test").then(pl.lit("AAA new currency")).otherwise(pl.col("receiving_currency")).alias("receiving_currency"))
    monkeypatch.setenv("FTM_FINAL_RUN", "1")
    ei1, ea1, y1, tx1, n1, codes1 = multi_gnn.edge_arrays(f)
    assert codes1 == codes0 and len(y1) > len(y0)
    np.testing.assert_array_equal(ea1[: len(y0)], ea0)  # the train/val rows are the prefix, with the same features
    assert (ea1[len(y0):, 2] == -1).all()  # an unseen category is -1, the same default as in the val run


def test_final_graph_refits_the_val_calibrator_and_scores_only_test_rows(monkeypatch):
    pytest.importorskip("sklearn")
    spec = importlib.util.spec_from_file_location("train_gnn", Path(__file__).resolve().parents[1] / "scripts" / "train_gnn.py")
    tg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tg)
    monkeypatch.setenv("FTM_FINAL_RUN", "1")
    f = _frame(with_test=True).sort(["timestamp", "tx_id"])
    v = f.filter(pl.col("split") == "val")
    t = f.filter(pl.col("split") == "test")
    rng = np.random.default_rng(0)
    raw_v = np.clip(0.02 + 0.3 * v["is_laundering"].to_numpy() * rng.random(v.height) + 0.01 * rng.random(v.height), 0, 1)
    cut = (v["timestamp"].min(), v["timestamp"].max())
    monkeypatch.setattr(multi_gnn, "train_and_score", lambda *a, **k: {"raw": raw_v, "tx_id": v["tx_id"].to_numpy(), "history": []})
    # the val run of the same graph: its table is what the final run reads back
    stored, _ = tg.run_graph(f.filter(pl.col("split") != "test"), cut, "see_everything", None, 1.0, 1, Path("."), "abc1234", 0, None)
    monkeypatch.setattr(multi_gnn, "train_and_score", lambda *a, **k: {"raw": raw_t, "tx_id": t["tx_id"].to_numpy(), "history": []})
    raw_t = np.clip(rng.random(t.height) * 0.3, 0, 1)
    raw_t[:3] = np.nan
    ck = Path("ck.pt")
    ck.write_bytes(b"x")
    try:
        tab, rec = tg.final_graph(f, cut, "see_everything", None, Path("."), "abc1234", 0, ck, 1, "multi_gnn", stored.select("tx_id", "val_part", "score_raw", "label", "p0"))
    finally:
        ck.unlink()
    assert rec["val_p0_max_abs_diff"] < 1e-12  # the refit calibrator reproduces the stored val p0 exactly
    assert set(tab["split"]) == {"test"} and tab["val_part"].null_count() == tab.height and tab["tx_id"].to_list() == t["tx_id"].to_list()
    assert tab["p0"].null_count() == 3 and tab["p0"].drop_nulls().is_between(0, 1).all()  # NaN (unsampled seed) stays null
    assert rec["calibration"] in {"platt", "isotonic"} and rec["test_rows"] == t.height
