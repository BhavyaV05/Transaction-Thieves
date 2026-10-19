"""patterns: parse `*_Patterns.txt` and map every pattern line to a `tx_id`.

Owner: T1 (Abhishek).  Spec: plan.md 5.2 step 2, tracks.md 3.3.

File format: blocks of
    BEGIN LAUNDERING ATTEMPT - <TYPOLOGY>[:  <detail>]
    <transaction lines in Trans.csv format, no header>
    END LAUNDERING ATTEMPT - <TYPOLOGY>
`attempt_id` is the 0-based ordinal of the block in the file.
"""

from __future__ import annotations

import io
from pathlib import Path

import polars as pl

from ftm.data.convert import RAW_COLS, RAW_SCHEMA, _type_raw

TYPOLOGIES = (
    "fan_in", "fan_out", "cycle", "scatter_gather", "gather_scatter", "bipartite", "stack", "random",
)  # fmt: skip
KEY = [c for c in RAW_COLS if c != "is_laundering"]
_BEGIN = "BEGIN LAUNDERING ATTEMPT - "


def _typology(header: str) -> str:
    name = header.removeprefix(_BEGIN).split(":")[0].strip().lower().replace("-", "_")
    if name not in TYPOLOGIES:
        raise ValueError(f"unknown typology in pattern file: {header!r}")
    return name


def parse_patterns(path: str | Path) -> pl.DataFrame:
    """One row per pattern line: attempt_id, typology, line_no (global order) + the Trans.csv columns."""
    lines, attempt_id, typology = [], -1, None
    for raw in Path(path).read_text().splitlines():
        raw = raw.strip()
        if raw.startswith(_BEGIN):
            attempt_id, typology = attempt_id + 1, _typology(raw)
        elif raw.startswith("END LAUNDERING ATTEMPT"):
            typology = None
        elif raw:
            if typology is None:
                raise ValueError(f"transaction line outside a block: {raw!r}")
            lines.append((attempt_id, typology, raw))
    csv = "\n".join(line for _, _, line in lines)
    tx = _type_raw(pl.read_csv(io.StringIO(csv), has_header=False, new_columns=RAW_COLS, schema=RAW_SCHEMA).lazy())
    meta = pl.DataFrame(
        {"attempt_id": [a for a, _, _ in lines], "typology": [t for _, t, _ in lines]},
        schema={"attempt_id": pl.Int64, "typology": pl.String},
    )
    return pl.concat([meta, tx.collect()], how="horizontal").with_row_index("line_no")


def match_patterns(pat: pl.DataFrame, trans: pl.LazyFrame) -> tuple[pl.DataFrame, dict]:
    """Map pattern lines to transactions on all columns except the label.

    Identical transactions can occur more than once. Lines with the same key are assigned in
    file order to candidate transactions ordered (is_laundering desc, tx_id asc), one transaction
    per line. Returns (matched: attempt_id, typology, tx_id, line_no; stats).
    """
    cand = (
        trans.select("tx_id", "is_laundering", *KEY)
        .join(pat.lazy().select(KEY).unique(), on=KEY, how="semi")
        .sort(["is_laundering", "tx_id"], descending=[True, False])
        .with_columns(pl.int_range(pl.len(), dtype=pl.UInt32).over(KEY).alias("rank"))
        .collect(engine="streaming")
    )
    n_cand = cand.group_by(KEY).agg(pl.len().alias("n_cand"))
    pat = (
        pat.with_columns(pl.col("line_no").rank("ordinal").over(KEY).sub(1).cast(pl.UInt32).alias("rank"))
        .join(n_cand, on=KEY, how="left")
    )
    matched = pat.join(
        cand.select("tx_id", "rank", *KEY), on=[*KEY, "rank"], how="inner"
    ).select("attempt_id", "typology", "tx_id", "line_no")
    stats = {
        "n_pattern_lines": pat.height,
        "n_attempts": int(pat["attempt_id"].n_unique()),
        "n_unmatched_lines": pat.height - matched.height,
        "n_lines_matching_multiple_tx": int((pat["n_cand"].fill_null(0) > 1).sum()),
    }
    return matched.sort("line_no"), stats


def build_attempts(matched: pl.DataFrame, pat: pl.DataFrame, trans: pl.LazyFrame) -> pl.DataFrame:
    """attempt_id, typology, n_lines, n_tx, first/last timestamp, splits (sorted list), spans_split."""
    info = trans.select("tx_id", "split").join(
        matched.lazy().select("tx_id", "attempt_id"), on="tx_id"
    ).group_by("attempt_id").agg(pl.col("split").unique().sort().alias("splits")).collect()
    return (
        pat.group_by("attempt_id", maintain_order=True)
        .agg(
            pl.col("typology").first(),
            pl.len().alias("n_lines"),
            pl.col("timestamp").min().alias("first_timestamp"),
            pl.col("timestamp").max().alias("last_timestamp"),
        )
        .join(matched.group_by("attempt_id").agg(pl.len().alias("n_tx")), on="attempt_id", how="left")
        .join(info, on="attempt_id", how="left")
        .with_columns(
            pl.col("n_tx").fill_null(0),
            pl.col("splits").fill_null([]),
        )
        .with_columns((pl.col("splits").list.len() > 1).alias("spans_split"))
        .select("attempt_id", "typology", "n_lines", "n_tx", "first_timestamp", "last_timestamp",
                "splits", "spans_split")  # fmt: skip
        .sort("attempt_id")
    )
