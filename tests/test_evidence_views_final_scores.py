"""The final-run evidence view joins the stored train/val edge scores (Float64) with the --final test scores (Float32 as written by
the scorers); 8 Nov 2026: every g5 LLM job failed with `SchemaError: type Float32 is incompatible with expected type Float64`."""

from __future__ import annotations

import os

import polars as pl

from ftm.retrieval import evidence_views as ev


def _write(root, ds, v2: bool):
    d = root / "data" / "retrieval" / ds
    (d / "final").mkdir(parents=True)
    if v2:
        pl.DataFrame({"tx_id": [1, 2], "bank": [7, 7], "score": [0.1, 0.9]}).write_parquet(d / "edge_scores_home_v2.parquet")
        pl.DataFrame({"tx_id": [3], "bank": [7], "score": pl.Series([0.5], dtype=pl.Float32)}).write_parquet(d / "final" / "edge_scores_home_v2_test.parquet")
    else:
        pl.DataFrame({"tx_id": [1, 2], "score": [0.1, 0.9]}).write_parquet(d / "edge_scores.parquet")
        pl.DataFrame({"tx_id": [3], "score": pl.Series([0.5], dtype=pl.Float32)}).write_parquet(d / "final" / "edge_scores_test.parquet")


def test_final_scores_concat_float32_test_with_float64_stored(tmp_path, monkeypatch):
    _write(tmp_path, "dsv2", True)
    _write(tmp_path, "dsv1", False)
    monkeypatch.chdir(tmp_path)
    ev._scores.cache_clear()
    try:
        v2 = ev._scores("dsv2", final=True, version="v2")
        v1 = ev._scores("dsv1", final=True, version="v1")
    finally:
        ev._scores.cache_clear()
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert v2["score"].dtype == pl.Float64 and v2["tx_id"].to_list() == [1, 2, 3]
    assert v1["score"].dtype == pl.Float64 and v1["tx_id"].to_list() == [1, 2, 3]
    assert abs(v2.filter(pl.col("tx_id") == 3)["score"][0] - 0.5) < 1e-6
