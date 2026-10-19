"""C2 alert sets (hand-off #6, draft): drop rules, test-count projection, locked external subset.

Owner: T1 (Abhishek).  Spec: plan.md 5.2 step 3 and 6.3, tracks.md 3.6.  Schema: ftm/intake/README.md.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ftm import guards


def add_drop_flags(alerts: pl.DataFrame, spans_split: pl.DataFrame, split_starts: dict, window_hours: float = 72):
    """Add drop_spans_split, drop_window and keep (plan.md 5.2 step 3).

    alerts: needs attempt_id (null if the transaction is in no attempt), timestamp, split.
    spans_split: attempt_id, spans_split (bool). split_starts: split name -> first timestamp of that split.
    An alert is dropped if its laundering attempt spans a split boundary, or if its evidence window
    [timestamp - window_hours, timestamp] starts before the split starts.
    """
    starts = pl.DataFrame({"split": list(split_starts), "split_start": list(split_starts.values())})
    out = (
        alerts.join(spans_split, on="attempt_id", how="left")
        .join(starts, on="split", how="left")
        .with_columns(
            pl.col("spans_split").fill_null(False).alias("drop_spans_split"),
            ((pl.col("timestamp") - pl.duration(hours=window_hours)) < pl.col("split_start")).alias("drop_window"),
        )
        .drop("spans_split", "split_start")
    )
    return out.with_columns((~pl.col("drop_spans_split") & ~pl.col("drop_window")).alias("keep"))


def project_test_count(per_day_rate: float, test_bulk_days: float) -> float:
    """PROJECTION (not a measurement): alerts per day observed on val x number of bulk test days.

    Uses only val rates and the calendar; no test row is touched.
    """
    return per_day_rate * test_bulk_days


def stratified_subset(alerts: pl.DataFrame, n: int = 1000, seed: int = 0) -> pl.DataFrame:
    """The frozen external-comparison subset: `n` TEST alerts stratified by label x typology x bank.

    Needs test alerts and labels, so it is locked: it raises TestSplitLocked unless FTM_FINAL_RUN=1.
    Allocation is proportional to stratum size with largest-remainder rounding and at least one alert
    per non-empty stratum while the budget allows; draws are seeded.
    """
    guards.require_final_run("the stratified external 1,000-alert subset (built from test alerts)")
    keys = ["label", "typology", "bank"]
    g = alerts.with_columns(pl.col("typology").fill_null("none")).group_by(keys).len()
    size = g["len"].to_numpy().astype("f8")
    raw = size / size.sum() * n
    alloc = np.floor(raw).astype(int)
    alloc = np.maximum(alloc, 1)
    # trim / top up to exactly n, by remainder
    order = np.argsort(-(raw - np.floor(raw)))
    i = 0
    while alloc.sum() < n and i < 10 * len(alloc):
        alloc[order[i % len(alloc)]] += 1
        i += 1
    while alloc.sum() > n:
        j = int(np.argmax(alloc))
        alloc[j] -= 1
    alloc = np.minimum(alloc, size.astype(int))
    take = g.with_columns(pl.Series("take", alloc))
    work = alerts.with_columns(pl.col("typology").fill_null("none")).join(take.select(keys + ["take"]), on=keys)
    work = work.sort("alert_id").with_columns(
        pl.int_range(pl.len()).shuffle(seed=seed).over(keys).alias("_r")
    )
    # shuffle(seed) over a group is deterministic; rank within group picks the first `take` rows
    work = work.with_columns(pl.col("_r").rank("ordinal").over(keys).alias("_rank"))
    return work.filter(pl.col("_rank") <= pl.col("take")).drop("_r", "_rank", "take")
