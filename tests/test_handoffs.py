"""T1 hand-offs: `direction` / `counterparty_bank` in alert contexts, and `top_features` (SHAP) per alert."""

from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ftm.banks.binding import AlertBinding, alert_context
from ftm.data import alias
from ftm.intake.alerts import add_direction
from ftm.retrieval import views
from ftm.retrieval.views import alert_summary, direction_keys, top_k_by_amount_view

T0 = datetime(2022, 9, 5, 12, 0)
CASES = [(70, 220, 70), (220, 70, 70), (70, 70, 70), (1, 2, 70)]  # from_bank, to_bank, home (the last is not at home)


def test_direction_keys_match_alert_context_and_alert_set_columns():
    df = pl.DataFrame({"from_bank": [c[0] for c in CASES], "to_bank": [c[1] for c in CASES], "bank": [c[2] for c in CASES]})
    cols = add_direction(df)
    assert "from_bank" not in cols.columns
    for i, (fb, tb, home) in enumerate(CASES):
        ctx = alert_context(AlertBinding("A", home, T0, 1, 2, 1.0, "USD", sender_bank=fb, receiver_bank=tb), 0.5)
        k = direction_keys(fb, tb, home)
        assert k == {"direction": ctx.summary["direction"], "counterparty_bank": ctx.summary["counterparty_bank"]}
        assert (cols["direction"][i], cols["counterparty_bank"][i]) == (k["direction"], k["counterparty_bank"])
    assert direction_keys(70, 70, 70) == {"direction": "out", "counterparty_bank": None}


def _toy_view(path: Path, bank: int):
    rows = [(i, datetime(2022, 9, 5, 10 + i // 4, 15 * (i % 4)), 1, 2, bank, 220, 10.0 + i) for i in range(8)]
    df = pl.DataFrame(rows, schema=["tx_id", "timestamp", "src_id", "dst_id", "from_bank", "to_bank", "amount_paid"], orient="row")
    n = df.height
    df = df.with_columns(
        pl.lit("US Dollar").alias("payment_currency"), pl.lit("ACH").alias("payment_format"), pl.lit("x").alias("from_account"),
        pl.lit("y").alias("to_account"), pl.lit([False] * n).alias("is_laundering"), pl.lit("val").alias("split"),
        pl.col("timestamp").dt.day().cast(pl.Int32).alias("day"),
    )  # fmt: skip
    path.mkdir(parents=True)
    df.write_parquet(path / f"bank={bank}.parquet")


def test_summary_gets_direction_and_top_features_without_editing_the_caller(tmp_path):
    ds = tmp_path / "data"
    _toy_view(ds / "views" / "toy", 70)
    (ds / "alerts" / "toy").mkdir(parents=True)
    item = {"feature": "payment_format=ACH", "value": 1.0, "contribution": 1.23456789}
    pl.DataFrame({"alert_id": ["V000000"], "tx_id": [7], "bank": [70], "top_features": [[item] * 5]}).write_parquet(
        ds / "alerts" / "toy" / "top_features_val.parquet")
    aliases = alias.make_alias_table([1, 2], seed=0)
    row = {"tx_id": 7, "timestamp": datetime(2022, 9, 5, 11, 45), "src_id": 1, "dst_id": 2, "from_bank": 70, "to_bank": 220,
           "amount_paid": 17.0, "payment_currency": "US Dollar", "payment_format": "ACH"}  # fmt: skip
    assert "direction" not in alert_summary(dict(row), aliases)  # no home bank known: unchanged behaviour
    top_k_by_amount_view(ds / "views" / "toy", 70, row, aliases, k=5)  # the call T3's data_alerts makes first
    s = alert_summary(row, aliases)  # ... and then this one
    assert s["direction"] == "out" and s["counterparty_bank"] == 220
    assert len(s["top_features"]) == 5 and s["top_features"][0]["contribution"] == 1.2346
    assert "src_id" not in s and s["sender"].startswith("A")
    views._top_features.cache_clear()


# ---- A2: LightGBM SHAP on synthetic data ----


def test_top_contributions_sorted_complete_and_readable():
    lgb = pytest.importorskip("lightgbm")
    from ftm.baselines import gfp_gbt as g

    rng = np.random.default_rng(0)
    n = 3000
    names = ["amount_received", "amount_paid", "receiving_currency", "payment_currency", "payment_format", "same_bank",
             "hour", "gfp_14", "gfp_22", "out"]  # fmt: skip
    X = rng.normal(size=(n, len(names))).astype("f4")
    X[:, 4] = rng.integers(0, 2, n)
    y = ((X[:, 7] > 0.5) & (X[:, 4] == 1)).astype(int)
    m = lgb.LGBMClassifier(n_estimators=40, verbose=-1, random_state=0).fit(X, y)
    codes = {"receiving_currency": {"US Dollar": 0}, "payment_currency": {"US Dollar": 0}, "payment_format": {"ACH": 0, "Wire": 1}}
    out = g.top_contributions(m, X[:50], np.arange(50), names, codes, k=5)
    assert out.height == 50 and out.schema["top_features"].inner.fields[0].name == "feature"
    full = m.booster_.predict(X[:50], pred_contrib=True)
    for i, items in enumerate(out["top_features"].to_list()):
        assert len(items) == 5
        mags = [abs(t["contribution"]) for t in items]
        assert mags == sorted(mags, reverse=True)
        assert max(mags) == pytest.approx(np.abs(full[i, :-1]).max())
    seen = {t["feature"] for items in out["top_features"].to_list() for t in items}
    assert "sender_out_distinct_counterparties_24h" in seen or "payment_format=Wire" in seen or "payment_format=ACH" in seen
    assert not any(f.startswith("gfp_") for f in seen)
    assert g.readable_names(["hour", "out", "gfp_14", "gfp_65"]) == [
        "hour_of_day", "home_bank_is_sender", "sender_out_distinct_counterparties_24h", "receiver_in_amount_kurtosis_24h"]


def test_gfp_labels_follow_the_snapml_layout():
    """Designed toy graphs: the column that reacts to a fan-out / fan-in / parallel edges / cycle is the one named so."""
    pytest.importorskip("snapml")
    from ftm.baselines import gfp_gbt as g

    def run(edges):
        return g._gfp(1).transform(np.array(edges, dtype="f8"))[:, g.N_ECHO:]

    fan_out = run([(i, 1, 2 + i, i + 1, 10) for i in range(5)])
    assert g.gfp_label(2) == "fan_out_size2_24h" and fan_out[0, 2] == 1 and fan_out[0, 0] == 0
    fan_in = run([(i, 2 + i, 1, i + 1, 10) for i in range(5)])
    assert g.gfp_label(0) == "fan_in_size2_24h" and fan_in[0, 0] == 1 and fan_in[0, 2] == 0
    deg = run([(i, 1, 2, i + 1, 10) for i in range(4)])
    assert g.gfp_label(4) == "degree_in_size2_24h" and deg[0, 4] == 1 and deg[0, 0] == 0
    cyc = run([(0, 1, 2, 1, 10), (1, 2, 3, 2, 10), (2, 3, 1, 3, 10)])
    assert g.gfp_label(11) == "temporal_cycle_size3plus_24h" and cyc[0, 11] == 1
    # vertex blocks: edge 0 = 1 -> 2; vertex 1 sends 3 (100, 200, 400), vertex 2 receives 3 (100, 50, 70)
    e = [(0, 1, 2, 10, 100), (1, 1, 3, 20, 200), (2, 1, 4, 30, 400), (3, 7, 1, 5, 1000), (4, 8, 2, 6, 50), (5, 9, 2, 7, 70)]
    v = run(e)[0]
    by = {g.gfp_label(i): v[i] for i in range(66)}
    assert by["sender_out_distinct_counterparties_24h"] == 3 and by["sender_out_amount_sum_24h"] == 700
    assert by["sender_in_amount_sum_24h"] == 1000 and by["receiver_in_amount_sum_24h"] == 220
    assert by["receiver_in_tx_count_24h"] == 3 and by["receiver_out_tx_count_24h"] == 0


# ---- real alert sets (slow; skipped when they have not been built) ----

ALERTS = Path(__file__).resolve().parents[1] / "data" / "alerts" / "hi_small"


@pytest.mark.slow
@pytest.mark.skipif(not (ALERTS / "top_features_val.parquet").exists(), reason="alert sets with hand-offs not built")
def test_real_alert_sets_have_direction_and_top_features():
    for split in ("train", "val"):
        a = pl.read_parquet(ALERTS / f"{split}.parquet")
        t = pl.read_parquet(ALERTS / f"top_features_{split}.parquet")
        assert set(a["direction"].unique()) <= {"out", "in"}
        assert t.height == a.height and t["top_features"].null_count() == 0
        assert (t["top_features"].list.len() == 5).all()
        assert t["alert_id"].to_list() == a["alert_id"].to_list()
