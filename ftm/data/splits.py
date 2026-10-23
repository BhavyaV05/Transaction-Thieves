"""splits: temporal split rule and the held-out-cycles switch (E12).

Owner: T1 (Abhishek).  Spec: plan.md 5.2 steps 3 and 7, tracks.md 3.3.

The split label itself is written by convert.build_transactions from `split_days` in the data config.
Loaders that return test rows must call ftm.guards.require_final_run() (see ftm/data/io.py).
"""

from __future__ import annotations

import polars as pl


def validate_split_days(split_days: dict | None) -> dict:
    """Check train < val < test, contiguous, 1-indexed inclusive day ranges."""
    if not split_days:
        raise ValueError("split_days is not set in the data config")
    prev_hi = 0
    for name in ("train", "val", "test"):
        lo, hi = split_days[name]
        if lo != prev_hi + 1 or hi < lo:
            raise ValueError(f"split_days[{name}]={[lo, hi]} must start at day {prev_hi + 1}")
        prev_hi = hi
    return split_days


def test_part(cfg: dict, day: str = "day") -> pl.Expr:
    """Expression: "bulk" / "tail" for test days, null for train and val days. Pure function of `day`.

    Uses `test_bulk_days` and `test_tail_days` of the data config (decision 4: the headline test is the
    bulk, the sparse tail is reported separately). It adds no column to the Parquet and changes neither
    `split` nor `tx_id`: `df.with_columns(test_part(cfg).alias("test_part"))`, then `.filter(is_tail)`.
    """
    sd = validate_split_days(cfg["split_days"])
    (b0, b1), (t0, t1) = cfg["test_bulk_days"], cfg["test_tail_days"]
    if (b0, t1) != tuple(sd["test"]) or t0 != b1 + 1 or b1 < b0:
        raise ValueError(f"test_bulk_days {[b0, b1]} + test_tail_days {[t0, t1]} must tile split_days.test {sd['test']}")
    d = pl.col(day)
    return pl.when(d.is_between(b0, b1)).then(pl.lit("bulk")).when(d.is_between(t0, t1)).then(pl.lit("tail")).otherwise(None)


def drop_cycle_attempts(frame, patterns: pl.LazyFrame | pl.DataFrame):
    """Remove transactions of cycle attempts from the *train* rows (held-out-cycles variant, E12).

    Val/test rows are untouched, so cycle alerts can still be evaluated. `patterns` needs
    `tx_id` and `typology`. Works on a DataFrame or LazyFrame of transactions.
    """
    cycles = patterns.lazy().filter(pl.col("typology") == "cycle").select("tx_id")
    drop = pl.col("tx_id").is_in(cycles.collect()["tx_id"].implode()) & (pl.col("split") == "train")
    return frame.filter(~drop)
