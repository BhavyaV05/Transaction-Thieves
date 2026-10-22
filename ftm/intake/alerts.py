"""C2 alert sets (hand-off #6, draft): drop rules, test-count projection, locked external subset.

Owner: T1 (Abhishek).  Spec: plan.md 5.2 step 3 and 6.3, tracks.md 3.6.  Schema: ftm/intake/README.md.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ftm import guards


def add_span_flag(alerts: pl.DataFrame, spans_split: pl.DataFrame) -> pl.DataFrame:
    """Add the boolean flag `spans_split` (plan.md 5.2 step 3, decision 3 of docs/t4_decisions_week2.md).

    alerts: needs attempt_id (null if the transaction is in no attempt). spans_split: attempt_id, spans_split.
    Flag only, for the sensitivity analysis: the headline alert set is never filtered by it. The 72 h
    evidence-window rule was dropped (it only removes alerts whose evidence is *earlier* data, which is allowed).
    """
    return alerts.join(spans_split, on="attempt_id", how="left").with_columns(pl.col("spans_split").fill_null(False))


def project_test_count(per_day_rate: float, test_bulk_days: float) -> float:
    """PROJECTION (not a measurement): alerts per day observed on val x number of bulk test days.

    Uses only val rates and the calendar; no test row is touched.
    """
    return per_day_rate * test_bulk_days


STRATA = {
    "bank_p0_decile": ["bank", "p0_decile"],  # label-free: known at alert time (decision 1)
    "label_typology_bank": ["label", "typology", "bank"],  # needs labels; kept for synthetic/val checks
}


def stratified_subset(alerts: pl.DataFrame, n: int = 1000, seed: int = 0, strata: str = "bank_p0_decile") -> pl.DataFrame:
    """The frozen external-comparison subset: `n` TEST alerts, stratified.

    strata="bank_p0_decile" (default, decision 1 of docs/t4_decisions_week2.md): home bank x decile of p0
    (deciles of p0 over the alerts passed in, ties broken by alert_id). Uses no label. Needs `bank`, `p0`, `alert_id`.
    strata="label_typology_bank": the earlier rule, needs `label`, `typology`, `bank`.
    Locked: it builds the subset from test alerts, so it raises TestSplitLocked unless FTM_FINAL_RUN=1.
    Allocation is proportional to stratum size with largest-remainder rounding and at least one alert
    per non-empty stratum while the budget allows; draws are seeded.
    """
    guards.require_final_run("the stratified external 1,000-alert subset (built from test alerts)")
    keys = STRATA[strata]
    work = alerts.sort("alert_id")
    if strata == "bank_p0_decile":
        work = work.with_columns((pl.col("p0").rank("ordinal") * 10 // (pl.len() + 1)).cast(pl.Int32).alias("p0_decile"))
    else:
        work = work.with_columns(pl.col("typology").fill_null("none"))
    g = work.group_by(keys).len()
    size = g["len"].to_numpy().astype("f8")
    raw = size / size.sum() * n
    alloc = np.maximum(np.floor(raw).astype(int), 1)
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
    work = work.join(take.select(keys + ["take"]), on=keys)
    # shuffle(seed) over a group is deterministic; rank within group picks the first `take` rows
    work = work.with_columns(pl.int_range(pl.len()).shuffle(seed=seed).over(keys).alias("_r"))
    work = work.with_columns(pl.col("_r").rank("ordinal").over(keys).alias("_rank"))
    return work.filter(pl.col("_rank") <= pl.col("take")).drop("_r", "_rank", "take")
