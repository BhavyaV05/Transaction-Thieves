"""convert: AMLworld CSVs -> typed Parquet (transactions + accounts), with stable tx_id and node ids.

Owner: T1 (Abhishek).  Spec: plan.md 5.2 step 1, tracks.md 3.3.

Streams through Polars lazy scans and `sink_parquet`, so the same code runs on HI-Medium (32M rows).
Bank ids are parsed as integers: the raw files write the same bank as `010` and `0010`.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

# Raw Trans.csv has two columns both called "Account", so the header row is skipped and renamed.
RAW_COLS = [
    "timestamp", "from_bank", "from_account", "to_bank", "to_account",
    "amount_received", "receiving_currency", "amount_paid", "payment_currency",
    "payment_format", "is_laundering",
]  # fmt: skip
TS_FORMAT = "%Y/%m/%d %H:%M"
RAW_SCHEMA = {c: pl.String for c in RAW_COLS}


def scan_raw_trans(path: str | Path) -> pl.LazyFrame:
    """Raw Trans.csv (or a header-less block of pattern lines) as a typed lazy frame, no tx_id yet."""
    return pl.scan_csv(
        path, has_header=False, skip_rows=1, new_columns=RAW_COLS, schema=RAW_SCHEMA
    ).pipe(_type_raw)


def _type_raw(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        pl.col("timestamp").str.to_datetime(TS_FORMAT),
        pl.col("from_bank").cast(pl.Int64),
        pl.col("to_bank").cast(pl.Int64),
        pl.col("amount_received").cast(pl.Float64),
        pl.col("amount_paid").cast(pl.Float64),
        pl.col("is_laundering").cast(pl.Int8).cast(pl.Boolean),
    )


def build_accounts(accounts_csv: str | Path, trans: pl.LazyFrame) -> pl.DataFrame:
    """node_id, bank, account, entity_id, entity_name. node_id = rank of (bank, account).

    Accounts that appear in transactions but not in accounts.csv are appended with null entity fields.
    """
    acc = pl.read_csv(accounts_csv, infer_schema=False).select(
        pl.col("Bank ID").cast(pl.Int64).alias("bank"),
        pl.col("Account Number").alias("account"),
        pl.col("Entity ID").alias("entity_id"),
        pl.col("Entity Name").alias("entity_name"),
    )
    if acc.select(["bank", "account"]).is_duplicated().any():
        raise ValueError("accounts.csv has duplicate (bank, account) keys")
    seen = pl.concat(
        [
            trans.select(pl.col("from_bank").alias("bank"), pl.col("from_account").alias("account")),
            trans.select(pl.col("to_bank").alias("bank"), pl.col("to_account").alias("account")),
        ]
    ).unique().collect(engine="streaming")
    return (
        pl.concat([acc, seen], how="diagonal")
        .unique(subset=["bank", "account"], keep="first", maintain_order=True)
        .sort(["bank", "account"])
        .with_row_index("node_id")
        .with_columns(pl.col("node_id").cast(pl.Int64))
        .select("node_id", "bank", "account", "entity_id", "entity_name")
    )


def split_expr(split_days: dict) -> pl.Expr:
    """`split` label from the 1-indexed `day`. split_days = {train: [lo, hi], val: [...], test: [...]}."""
    expr = pl.when(pl.lit(False)).then(pl.lit(None, dtype=pl.String))
    for name in ("train", "val", "test"):
        lo, hi = split_days[name]
        expr = expr.when(pl.col("day").is_between(lo, hi)).then(pl.lit(name))
    return expr.otherwise(pl.lit(None, dtype=pl.String)).alias("split")


def build_transactions(
    trans_csv: str | Path, accounts: pl.DataFrame, split_days: dict, out: str | Path
) -> None:
    """Write transactions.parquet. tx_id = 0-based row index in the raw CSV (the file is fixed)."""
    lf = scan_raw_trans(trans_csv).with_row_index("tx_id").with_columns(pl.col("tx_id").cast(pl.Int64))
    nodes = accounts.lazy().select("node_id", "bank", "account")
    first_day = lf.select(pl.col("timestamp").dt.date().min()).collect(engine="streaming").item()
    lf = (
        lf.join(
            nodes.rename({"node_id": "src_id", "bank": "from_bank", "account": "from_account"}),
            on=["from_bank", "from_account"],
            how="left",
            maintain_order="left",
        )
        .join(
            nodes.rename({"node_id": "dst_id", "bank": "to_bank", "account": "to_account"}),
            on=["to_bank", "to_account"],
            how="left",
            maintain_order="left",
        )
        .with_columns(
            ((pl.col("timestamp").dt.date() - first_day).dt.total_days() + 1)
            .cast(pl.Int32)
            .alias("day")
        )
        .with_columns(split_expr(split_days))
        .select(
            "tx_id", "timestamp", "day", "from_bank", "from_account", "to_bank", "to_account",
            "src_id", "dst_id", "amount_received", "receiving_currency", "amount_paid",
            "payment_currency", "payment_format", "is_laundering", "split",
        )  # fmt: skip
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    lf.sink_parquet(out, engine="streaming")
