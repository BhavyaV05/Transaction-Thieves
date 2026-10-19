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


def drop_cycle_attempts(frame, patterns: pl.LazyFrame | pl.DataFrame):
    """Remove transactions of cycle attempts from the *train* rows (held-out-cycles variant, E12).

    Val/test rows are untouched, so cycle alerts can still be evaluated. `patterns` needs
    `tx_id` and `typology`. Works on a DataFrame or LazyFrame of transactions.
    """
    cycles = patterns.lazy().filter(pl.col("typology") == "cycle").select("tx_id")
    drop = pl.col("tx_id").is_in(cycles.collect()["tx_id"].implode()) & (pl.col("split") == "train")
    return frame.filter(~drop)
