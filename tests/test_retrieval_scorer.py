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


def test_score_final_scores_test_edges_and_reproduces_the_val_scores(monkeypatch, tmp_path):
    import pytest

    torch = pytest.importorskip("torch")
    from ftm import guards
    from ftm.data import io

    f = _frame(500, seed=5)
    f = f.with_columns((pl.col("timestamp").dt.ordinal_day() - 244).cast(pl.Int64).alias("day"))
    f = f.with_columns(pl.when(pl.col("day") <= 0).then(pl.lit("train")).when(pl.col("day") == 1).then(pl.lit("val")).otherwise(pl.lit("test")).alias("split"))
    assert set(f["split"]) == {"train", "val", "test"}
    tv = f.filter(pl.col("split") != "test")
    feats, codes = es.timeline_features(tv.drop("day", "split"))
    # a random but fixed network: the checkpoint of a "val run"
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(len(es.FEATURES2), 128), torch.nn.ReLU(), torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1))
    X = feats.select(es.FEATURES2).to_numpy().astype("f4")
    X = np.sign(X) * np.log1p(np.abs(X))
    mu, sd = X.mean(0), X.std(0) + 1e-6
    sig = lambda Z: 1 / (1 + np.exp(-Z.astype("f8")))  # noqa: E731
    with torch.no_grad():
        stored = sig(net(torch.tensor(((X - mu) / sd).astype("f4"))).squeeze(1).numpy())
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "data" / "runs").mkdir(parents=True)
    (tmp_path / "data" / "retrieval" / "toy").mkdir(parents=True)
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "features": es.FEATURES2, "codes": codes, "best_epoch": 0}, "models/edge_scorer_toy.pt")
    split_of = dict(zip(f["tx_id"], f["split"], strict=True))
    pl.DataFrame({"tx_id": feats["tx_id"], "score": stored, "split": [split_of[t] for t in feats["tx_id"]]}).write_parquet("data/retrieval/toy/edge_scores.parquet")
    monkeypatch.setattr(io, "load_config", lambda name: {"parquet_dir": "pq"})
    monkeypatch.setattr(io, "scan_transactions", lambda *a, **k: f.lazy())
    monkeypatch.delenv("FTM_FINAL_RUN", raising=False)
    with pytest.raises(guards.TestSplitLocked):
        es.score_final("toy")
    monkeypatch.setenv("FTM_FINAL_RUN", "1")
    rec = es.score_final("toy")
    assert rec["max_abs_val_score_diff"] < 1e-5 and rec["test_edges"] == (f["split"] == "test").sum()
    got = pl.read_parquet("data/retrieval/toy/final/edge_scores_test.parquet")
    assert set(got["split"]) == {"test"} and got["score"].is_between(0, 1).all()
    # reference: the whole-frame features of the full timeline with the same network
    full, _ = es.timeline_features(f.drop("day", "split"), codes)
    Xf = full.select(es.FEATURES2).to_numpy().astype("f4")
    Xf = np.sign(Xf) * np.log1p(np.abs(Xf))
    with torch.no_grad():
        ref = sig(net(torch.tensor(((Xf - mu) / sd).astype("f4"))).squeeze(1).numpy())
    want = dict(zip(full["tx_id"].to_list(), ref, strict=True))
    np.testing.assert_allclose(got["score"].to_numpy(), [want[t] for t in got["tx_id"].to_list()], atol=1e-5)
