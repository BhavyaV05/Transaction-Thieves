"""Edge scorer v2 (finding 18): features from the home bank's own view, day-block folds, per-bank prize lookup. Synthetic data."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ftm.retrieval import edge_scorer as es
from ftm.retrieval import edge_scorer_home as eh
from ftm.retrieval import evidence_views as ev


def _frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    t0 = datetime(2022, 9, 1)
    ts = sorted(t0 + timedelta(minutes=int(m)) for m in rng.integers(0, 4000, n))
    return pl.DataFrame({
        "tx_id": np.arange(n), "timestamp": ts, "from_bank": rng.integers(1, 4, n), "to_bank": rng.integers(1, 4, n),
        "src_id": rng.integers(0, 8, n), "dst_id": rng.integers(0, 8, n), "amount_received": rng.uniform(1, 100, n),
        "amount_paid": rng.uniform(1, 100, n), "receiving_currency": rng.choice(["a", "b"], n),
        "payment_currency": rng.choice(["a", "b"], n), "payment_format": rng.choice(["ACH", "Wire"], n),
    })  # fmt: skip


def test_home_features_use_only_the_view_and_flag_the_side():
    f = _frame()
    view = f.filter((pl.col("from_bank") == 1) | (pl.col("to_bank") == 1))
    feats, codes = eh.home_features(view, 1)
    assert feats.columns[1:] == eh.FEATURES_HOME and feats.height == view.height
    glob, _ = es.timeline_features(f, codes)
    g = glob.filter(pl.col("tx_id").is_in(view["tx_id"].to_list())).sort("tx_id")
    # the global (v1) aggregates see foreign flows of the same accounts, the home-view ones do not
    assert not np.allclose(g["src_out_amt"].to_numpy(), feats["src_out_amt"].to_numpy())
    # removing every edge outside the view cannot change a view feature (nothing else is read); the other-bank rows are not even passed
    again, _ = eh.home_features(view.sort("tx_id", descending=True), 1, codes)
    assert np.allclose(again.sort("tx_id")[eh.FEATURES_HOME].to_numpy(), feats.sort("tx_id")[eh.FEATURES_HOME].to_numpy())
    side = view.sort("tx_id").select(((pl.col("from_bank") == 1).cast(int) + 2 * (pl.col("to_bank") == 1).cast(int)).alias("s"))["s"].to_numpy()
    assert (feats.sort("tx_id")["home_side"].to_numpy() == side).all() and set(side) <= {1, 2, 3}


def test_day_folds_are_contiguous_blocks():
    d = np.arange(1, 13)
    f = eh.day_folds(d, 1, 9)
    assert f.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, -1, -1, -1]


def test_fit_net_runs_and_orders_signal():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 5)).astype("f4")
    y = (X[:, 0] + 0.3 * rng.normal(size=4000) > 1.2).astype(int)
    net, ep, hist = eh.fit_net(X, y, 80, 0.5, 0, select=(X, y))
    s = eh._predict(net, X)
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y, s) > 0.9 and 0 <= ep < 80 and len(hist) == 80


def test_prizes_look_up_the_home_banks_score_only():
    cand = pl.DataFrame({"tx_id": [1, 2, 3], "amount_paid": [1.0, 2.0, 3.0]})
    sc = pl.DataFrame({"tx_id": [1, 2, 2, 3], "bank": [5, 5, 6, 6], "score": [0.1, 0.2, 0.9, 0.3]})
    assert ev.prizes(cand, sc, 5).tolist() == [0.1, 0.2, 0.0]
    assert ev.prizes(cand, sc, 6).tolist() == [0.0, 0.9, 0.3]
    with pytest.raises(ValueError):
        ev.prizes(cand, sc)
    assert ev.prizes(cand, sc.drop("bank").unique("tx_id"), None).shape == (3,)


def test_scorer_version_switch(monkeypatch):
    monkeypatch.delenv("FTM_EDGE_SCORER", raising=False)
    assert ev.scorer_version() == "v1" and ev.scorer_version("v2") == "v2"
    monkeypatch.setenv("FTM_EDGE_SCORER", "v2")
    assert ev.scorer_version() == "v2"
    with pytest.raises(ValueError):
        ev.scorer_version("v3")
