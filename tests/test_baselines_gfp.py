"""T1: p0 schema, test lock of the scoring path, calibration, GFP leakage rule. Synthetic data only."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ftm import guards
from ftm.baselines import gfp_gbt as g


def _frame(n=6000, seed=0, with_test=False):
    rng = np.random.default_rng(seed)
    t0 = datetime(2022, 9, 1)
    ts = np.sort(rng.integers(0, 6 * 1440, n))  # minutes over 6 days
    split = np.where(ts < 2 * 1440, "train", np.where(ts < 4 * 1440, "val", "test"))
    if not with_test:
        split = np.where(split == "test", "val", split)
    amt = rng.lognormal(6, 1.5, n)
    y = (rng.random(n) < 0.02).astype(int)
    amt = np.where(y == 1, amt * 5, amt)  # something learnable
    return pl.DataFrame({
        "tx_id": np.arange(n), "timestamp": [t0 + timedelta(minutes=int(m)) for m in ts],
        "from_bank": rng.integers(1, 4, n), "to_bank": rng.integers(1, 4, n),
        "src_id": rng.integers(0, 3000, n), "dst_id": rng.integers(0, 3000, n),
        "amount_received": amt, "amount_paid": amt, "receiving_currency": ["US Dollar"] * n,
        "payment_currency": ["US Dollar"] * n, "payment_format": rng.choice(["ACH", "Wire"], n),
        "is_laundering": y.astype(bool), "split": split,
    })  # fmt: skip


def test_scoring_path_refuses_test_rows(monkeypatch):
    monkeypatch.delenv("FTM_FINAL_RUN", raising=False)
    f = _frame(with_test=True)
    with pytest.raises(guards.TestSplitLocked):
        g.build_features(f, g.category_codes(f))
    with pytest.raises(guards.TestSplitLocked):
        g.score_graph(f, g.category_codes(f), (f["timestamp"].min(), f["timestamp"].max()))


def test_metrics_and_calibrators():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.1).astype(int)
    raw = np.clip(0.1 + 0.5 * y * rng.random(5000) + 0.2 * rng.random(5000), 0, 1)
    cal, name = g.fit_calibrator(raw, y)
    assert name == "isotonic" and 0 <= cal(raw).min() and cal(raw).max() <= 1
    cal2, name2 = g.fit_calibrator(raw[:200], np.r_[np.ones(3, int), np.zeros(197, int)])
    assert name2 == "platt" and np.all((cal2(raw) >= 0) & (cal2(raw) <= 1))
    m = g.metrics(y, cal(raw))
    assert m["brier"] < y.mean() * (1 - y.mean()) and m["pr_auc"] > 0.3 and m["f1"] > 0.3


def test_val_parts_cuts():
    t0 = datetime(2022, 9, 10)
    ts = pl.Series([t0 + timedelta(hours=h) for h in (0, 20, 35, 40, 60, 71)])
    assert g.val_parts(ts, t0, t0 + timedelta(hours=72)).tolist() == ["fit", "fit", "fit", "cal", "eval", "eval"]


def test_gfp_features_are_past_only_across_batches():
    pytest.importorskip("snapml")
    f = _frame(n=1500)
    codes = g.category_codes(f)
    X1, _, s1 = g.build_features(f, codes, threads=1)
    # change only the LAST 300 rows (their endpoints); features of rows that are in earlier batches stay
    f2 = f.sort(["timestamp", "tx_id"]).with_columns(
        pl.when(pl.int_range(pl.len()) >= 1200).then((pl.col("src_id") + 1) % 3000).otherwise(pl.col("src_id")).alias("src_id")
    )
    X2, _, s2 = g.build_features(f2, codes, threads=1)
    cut = 1200 - g.BATCH  # batches restart at each split, so only rows >= 128 before the change can share its batch
    assert s1["tx_id"].to_list() == s2["tx_id"].to_list()
    np.testing.assert_array_equal(X1[:cut, len(g.BASIC):], X2[:cut, len(g.BASIC):])


def test_score_graph_and_p0_schema():
    pytest.importorskip("snapml")
    pytest.importorskip("lightgbm")
    pytest.importorskip("sklearn")
    f = _frame(n=9000)
    codes = g.category_codes(f)
    val = f.filter(pl.col("split") == "val")["timestamp"]
    res = g.score_graph(f, codes, (val.min(), val.max()), grid=g.LGBM_GRID[:1], threads=1)
    assert res["meta"]["status"] == "ok"
    t = g.p0_table(res, "one_bank", 7, "abc1234")
    assert t.columns == ["tx_id", "bank", "mode", "model", "split", "val_part", "p0", "score_raw", "label", "seed", "commit"]
    assert set(t["split"].unique()) == {"train", "val"} and set(t["mode"].unique()) == {"one_bank"}
    assert t["p0"].is_between(0, 1).all() and t["p0"].null_count() == 0
    assert set(t.filter(pl.col("split") == "train")["val_part"].unique().to_list()) == {None}
    assert set(t.filter(pl.col("split") == "val")["val_part"].unique().to_list()) == {"fit", "cal", "eval"}


def test_edge_scorer_features_are_past_only():
    from ftm.retrieval.edge_scorer import FEATURES, edge_features

    f = _frame(n=500).filter(pl.col("split") == "train")
    feats, codes = edge_features(f)
    assert feats.columns[0] == "tx_id" and set(FEATURES) <= set(feats.columns)
    s = f.sort(["timestamp", "tx_id"])
    # the 1st transaction of any sender has 0 prior sends counted before it: cum_count starts at 1 (itself)
    first = s.group_by("src_id").first().select("tx_id")["tx_id"]
    assert (feats.filter(pl.col("tx_id").is_in(first))["src_prior_out"] == 1).all()
    assert feats["src_prior_in"].min() >= 0 and feats["dst_prior_out"].min() >= 0
    # appending later rows never changes earlier rows' features
    feats2, _ = edge_features(s.head(300), codes)
    ids = feats2["tx_id"]
    a = feats.filter(pl.col("tx_id").is_in(ids)).sort("tx_id").select(FEATURES)
    b = feats2.sort("tx_id").select(FEATURES)
    assert a.equals(b)
