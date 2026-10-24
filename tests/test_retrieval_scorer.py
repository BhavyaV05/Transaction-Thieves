"""T1 edge scorer features: past-only aggregates, checked against a brute-force loop. Synthetic data."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from ftm.retrieval import edge_scorer as es


def _frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    t0 = datetime(2022, 9, 1)
    ts = sorted(t0 + timedelta(minutes=int(m)) for m in rng.integers(0, 4000, n))
    return pl.DataFrame({
        "tx_id": np.arange(n), "timestamp": ts, "from_bank": rng.integers(1, 3, n), "to_bank": rng.integers(1, 3, n),
        "src_id": rng.integers(0, 6, n), "dst_id": rng.integers(0, 6, n), "amount_received": rng.uniform(1, 100, n),
        "amount_paid": rng.uniform(1, 100, n), "receiving_currency": rng.choice(["a", "b"], n),
        "payment_currency": rng.choice(["a", "b"], n), "payment_format": rng.choice(["ACH", "Wire"], n),
    })  # fmt: skip


def test_window_aggregates_match_brute_force_and_ignore_the_future():
    f = _frame()
    feats, _ = es.timeline_features(f)
    d = feats.sort("tx_id")
    rows = f.sort("tx_id").to_dicts()
    for i in (5, 20, 59):
        r = rows[i]
        past = [q for q in rows if r["timestamp"] - timedelta(hours=24) <= q["timestamp"] < r["timestamp"]]
        src_out = sum(q["amount_paid"] for q in past if q["src_id"] == r["src_id"])
        dst_in = sum(q["amount_received"] for q in past if q["dst_id"] == r["dst_id"])
        assert np.isclose(d["src_out_amt"][i], np.log1p(src_out), atol=1e-6)
        assert np.isclose(d["dst_in_amt"][i], np.log1p(dst_in), atol=1e-6)
    # deleting every later edge cannot change an earlier edge's features
    cut = f.filter(pl.col("tx_id") < 30)
    d2, _ = es.timeline_features(cut, codes=_)
    a, b = d.filter(pl.col("tx_id") < 30), d2.sort("tx_id")
    for c in es.AGG:
        assert np.allclose(a[c].to_numpy(), b[c].to_numpy(), atol=1e-6), c


def test_chunked_features_equal_the_whole_frame_computation():
    f = _frame(400, seed=3).with_columns((pl.col("timestamp").dt.ordinal_day() - 244).alias("day"))  # days 1-3, ties on minutes
    f = f.with_columns(pl.col("day").cast(pl.Int64))
    assert f["day"].n_unique() >= 3
    full, codes = es.timeline_features(f.drop("day"))
    n_nodes = int(max(f["src_id"].max(), f["dst_id"].max())) + 1
    parts = [x for _, x in es.chunked_timeline_features(f.lazy(), codes, n_nodes)]
    got = pl.concat(parts).sort("tx_id")
    assert got["tx_id"].to_list() == full["tx_id"].to_list()
    for c in es.FEATURES2:
        assert np.allclose(got[c].to_numpy(), full[c].to_numpy(), atol=1e-6), c
