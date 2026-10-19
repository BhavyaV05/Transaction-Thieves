"""alias: opaque, per-seed aliases for account ids, for anything an LLM reads.

Owner: T1 (Abhishek).  Spec: plan.md 5.2 step 5, invariant 4.

SERVER-SIDE ONLY. The mapping files `data/parquet/<ds>/aliases/seed=<s>.parquet` (node_id -> alias)
must never be sent to a model, a partner-bank simulator or a log that an LLM reads; only the
aliased frames are. Aliases are a seeded random permutation of the node ids, so neither the
alias string nor its order reveals the account number or the bank.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

ID_COLS = ("src_id", "dst_id")
RAW_COLS = ("from_account", "to_account")  # dropped by apply_aliases: they would undo the alias


def make_alias_table(node_ids, seed: int) -> pl.DataFrame:
    """Deterministic for (node_ids, seed): node_id -> alias string like `A00f3a1c`."""
    ids = np.sort(np.asarray(node_ids, dtype=np.int64))
    perm = np.random.default_rng(seed).permutation(len(ids))
    return pl.DataFrame({"node_id": ids, "alias": [f"A{p:07x}" for p in perm]})


def alias_path(parquet_dir: str | Path, seed: int) -> Path:
    return Path(parquet_dir) / "aliases" / f"seed={seed}.parquet"


def write_aliases(parquet_dir: str | Path, seed: int) -> Path:
    nodes = pl.read_parquet(Path(parquet_dir) / "accounts.parquet", columns=["node_id"])
    path = alias_path(parquet_dir, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_alias_table(nodes["node_id"].to_numpy(), seed).write_parquet(path)
    return path


def read_aliases(parquet_dir: str | Path, seed: int) -> pl.DataFrame:
    return pl.read_parquet(alias_path(parquet_dir, seed))


def apply_aliases(frame, aliases: pl.DataFrame):
    """Replace src_id/dst_id by src_alias/dst_alias and drop raw account numbers.

    Works on a DataFrame or LazyFrame. Columns that are absent are skipped.
    """
    cols = frame.collect_schema().names()
    amap = aliases.lazy() if isinstance(frame, pl.LazyFrame) else aliases
    for c in ID_COLS:
        if c in cols:
            frame = frame.join(
                amap.rename({"node_id": c, "alias": c.replace("_id", "_alias")}),
                on=c, how="left", maintain_order="left",
            ).drop(c)  # fmt: skip
    return frame.drop([c for c in RAW_COLS if c in cols])
