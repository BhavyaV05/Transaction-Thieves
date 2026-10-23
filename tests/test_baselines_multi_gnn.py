"""T1 Multi-GNN our-split driver: no test rows into the graph, p0 table schema for model=multi_gnn. Synthetic, no GPU."""

import importlib.util
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ftm.baselines import multi_gnn
from tests.test_baselines_gfp import _frame


def test_edge_arrays_refuse_test_rows_and_keep_int64_ids():
    f = _frame(with_test=True).sort(["timestamp", "tx_id"])
    with pytest.raises(ValueError, match="test"):
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
