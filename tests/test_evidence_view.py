"""T3 week 2: top-40-by-amount evidence stub (hand-off #11) on a toy view written in the ftm.data layout."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from ftm.data import alias
from ftm.guards import TimeGuardError
from ftm.retrieval.views import (
    alert_summary,
    format_trail,
    select_top_k_by_amount,
    top_k_by_amount_view,
)

T0 = datetime(2022, 9, 5, 12, 0)
BANK = 70
SENDER, RECEIVER, OTHER = 1, 2, 99


def toy_view() -> pl.DataFrame:
    rows = []
    # 60 edges into RECEIVER over the 72 h before T0, amounts 100..6000
    for i in range(60):
        rows.append((i, T0 - timedelta(hours=71) + timedelta(hours=i), 100 + i, RECEIVER, 220, BANK, 100.0 * (i + 1)))
    rows.append((100, T0, SENDER, RECEIVER, BANK, BANK, 50.0, ))  # the alert: small amount, at as_of
    rows.append((101, T0 - timedelta(hours=80), 300, RECEIVER, 220, BANK, 1e6))  # outside the window
    rows.append((102, T0 + timedelta(hours=1), RECEIVER, 400, BANK, 1588, 1e6))  # after as_of
    rows.append((103, T0 - timedelta(hours=1), OTHER, 500, BANK, BANK, 1e6))  # does not touch the alert
    df = pl.DataFrame(rows, schema=["tx_id", "timestamp", "src_id", "dst_id", "from_bank", "to_bank", "amount_paid"],
                      orient="row")  # fmt: skip
    n = df.height
    return df.with_columns(
        pl.lit("US Dollar").alias("payment_currency"), pl.lit("ACH").alias("payment_format"),
        pl.lit("x").alias("from_account"), pl.lit("y").alias("to_account"), pl.lit([False] * n).alias("is_laundering"),
        pl.lit("val").alias("split"), pl.col("timestamp").dt.day().cast(pl.Int32).alias("day"),
    )  # fmt: skip


def test_top_k_selection_window_and_order():
    view = toy_view().filter(pl.col("timestamp") <= T0)  # what ftm.data.io gives us
    top = select_top_k_by_amount(view, [SENDER, RECEIVER], T0, k=40, alert_tx_id=100)
    assert top.height == 40
    ids = set(top["tx_id"].to_list())
    assert 100 in ids  # the alert is always kept even though its amount is small
    assert not ids & {101, 102, 103}
    assert top["timestamp"].is_sorted()
    assert set(range(21, 60)) <= ids  # the 39 largest in-window edges


def test_rows_after_as_of_never_selected_even_from_an_unguarded_frame():
    top = select_top_k_by_amount(toy_view(), [SENDER, RECEIVER], T0, k=100)  # caller forgot ftm.data.io
    assert 102 not in set(top["tx_id"].to_list())
    assert top["timestamp"].max() <= T0
    with pytest.raises(TimeGuardError):
        select_top_k_by_amount(toy_view(), [SENDER, RECEIVER], None)


def test_format_trail_refuses_raw_ids_and_labels():
    view = toy_view().filter(pl.col("timestamp") <= T0)
    top = select_top_k_by_amount(view, [SENDER, RECEIVER], T0, k=5)
    with pytest.raises(ValueError):
        format_trail(top)


def test_full_pipeline_through_ftm_data_io(tmp_path):
    views = tmp_path / "views"
    views.mkdir()
    toy_view().write_parquet(views / f"bank={BANK}.parquet")
    nodes = [SENDER, RECEIVER, OTHER, 300, 400, 500, *range(100, 160)]
    aliases = alias.make_alias_table(nodes, seed=0)
    alert = {"tx_id": 100, "timestamp": T0, "src_id": SENDER, "dst_id": RECEIVER, "from_bank": BANK, "to_bank": BANK,
             "amount_paid": 50.0, "payment_currency": "US Dollar", "payment_format": "ACH"}  # fmt: skip
    rows, text = top_k_by_amount_view(views, BANK, alert, aliases, k=10)
    assert rows.height == 10 and {"src_alias", "dst_alias"} <= set(rows.columns)
    assert not {"src_id", "dst_id", "from_account", "to_account"} & set(rows.columns)
    assert text.count("\n") == 9 and "<- ALERT" in text and "(incoming)" in text
    s = alert_summary(alert, aliases)
    assert s["sender"].startswith("A") and s["sender"] in text and "src_id" not in s
