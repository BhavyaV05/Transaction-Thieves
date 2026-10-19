"""views: the N largest banks and one Parquet view per bank.

Owner: T1 (Abhishek).  Spec: plan.md 5.2 step 4, tracks.md 3.3.

View(b) = transactions with from_bank == b or to_bank == b (all splits; readers filter).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def bank_table(trans: pl.LazyFrame, n_banks: int) -> pl.DataFrame:
    """rank, bank, n_tx, n_laundering for the n largest banks (sender or receiver; ties by bank id)."""
    sides = pl.concat(
        [
            trans.select(pl.col("from_bank").alias("bank"), "is_laundering"),
            trans.filter(pl.col("from_bank") != pl.col("to_bank")).select(
                pl.col("to_bank").alias("bank"), "is_laundering"
            ),
        ]
    )
    return (
        sides.group_by("bank")
        .agg(pl.len().cast(pl.Int64).alias("n_tx"), pl.col("is_laundering").sum().cast(pl.Int64).alias("n_laundering"))
        .sort(["n_tx", "bank"], descending=[True, False])
        .head(n_banks)
        .with_row_index("rank", offset=1)
        .with_columns(pl.col("rank").cast(pl.Int64))
        .collect(engine="streaming")
    )


def view_path(views_dir: str | Path, bank: int) -> Path:
    return Path(views_dir) / f"bank={bank}.parquet"


def write_views(trans: pl.LazyFrame, banks: pl.DataFrame, views_dir: str | Path) -> None:
    """One streaming pass per bank (ponytail: N scans; one partitioned pass is not possible because
    a transaction belongs to two views). Writes banks.parquet next to the views."""
    Path(views_dir).mkdir(parents=True, exist_ok=True)
    for bank in banks["bank"]:
        trans.filter((pl.col("from_bank") == bank) | (pl.col("to_bank") == bank)).sink_parquet(
            view_path(views_dir, bank), engine="streaming"
        )
    banks.write_parquet(Path(views_dir) / "banks.parquet")
