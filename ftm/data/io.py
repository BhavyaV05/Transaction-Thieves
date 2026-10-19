"""io: the single entry point for reading transaction, view and label Parquet inside ftm/.

Owner: T1 (Abhishek).  Spec: plan.md 4.3 and 5.2 steps 3 and 6, tracks.md 3.3.

Rules enforced here (tests/test_static_checks.py checks nobody else reads transaction Parquet):
  - every read names `split=` and/or `as_of=`; there is no "read everything" call;
  - `as_of` goes through ftm.guards.enforce_as_of (timestamp <= as_of);
  - asking for the test split calls ftm.guards.require_final_run(); with `as_of` only, test rows
    are removed unless FTM_FINAL_RUN=1, so a late `as_of` cannot leak the test days;
  - `heldout_cycles=True` drops cycle-attempt rows from the train split (E12).
All scans return LazyFrames; call `.collect()`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import yaml

from ftm import guards
from ftm.data.splits import drop_cycle_attempts

SPLITS = ("train", "val", "test")
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "data"


def load_config(name_or_path: str | Path) -> dict:
    """`hi_small` -> configs/data/hi_small.yaml, or a path to a yaml file."""
    p = Path(name_or_path)
    if not p.suffix:
        p = _CONFIG_DIR / f"{name_or_path}.yaml"
    return yaml.safe_load(p.read_text())


def _splits(split) -> list[str]:
    out = [] if split is None else [split] if isinstance(split, str) else list(split)
    bad = [s for s in out if s not in SPLITS]
    if bad:
        raise ValueError(f"unknown split {bad}; expected one of {SPLITS}")
    return out


def _guarded(lf: pl.LazyFrame, split, as_of, what: str) -> pl.LazyFrame:
    splits = _splits(split)
    if not splits and as_of is None:
        raise ValueError(f"reading {what} needs split= or as_of=")
    if "test" in splits:
        guards.require_final_run(f"the test split of {what}")
    if splits:
        lf = lf.filter(pl.col("split").is_in(splits))
    elif not guards.final_run_enabled():
        lf = lf.filter(pl.col("split") != "test")
    if as_of is not None:
        as_of = datetime.fromisoformat(as_of) if isinstance(as_of, str) else as_of
        lf = guards.enforce_as_of(lf, as_of)
    return lf


def scan_transactions(
    parquet_dir: str | Path, *, split=None, as_of=None, columns=None, heldout_cycles: bool = False
) -> pl.LazyFrame:
    """Transactions of `parquet_dir/transactions.parquet`, guarded. See module docstring."""
    parquet_dir = Path(parquet_dir)
    lf = _guarded(pl.scan_parquet(parquet_dir / "transactions.parquet"), split, as_of, "transactions")
    if heldout_cycles:
        lf = drop_cycle_attempts(lf, pl.scan_parquet(parquet_dir / "patterns.parquet"))
    return lf.select(columns) if columns else lf


def scan_view(
    views_dir: str | Path, bank: int, *, split=None, as_of=None, columns=None,
    heldout_cycles: bool = False, parquet_dir: str | Path | None = None,
) -> pl.LazyFrame:  # fmt: skip
    """View(bank) = rows with from_bank == bank or to_bank == bank, guarded like scan_transactions.

    `heldout_cycles` needs `parquet_dir` (where patterns.parquet lives).
    """
    lf = _guarded(pl.scan_parquet(Path(views_dir) / f"bank={bank}.parquet"), split, as_of, f"view {bank}")
    if heldout_cycles:
        lf = drop_cycle_attempts(lf, pl.scan_parquet(Path(parquet_dir) / "patterns.parquet"))
    return lf.select(columns) if columns else lf


def scan_patterns(parquet_dir: str | Path, *, split) -> pl.LazyFrame:
    """tx_id, attempt_id, typology, split of laundering transactions that belong to an attempt.

    These are labels, so the test split is locked like the transactions.
    """
    return _guarded(pl.scan_parquet(Path(parquet_dir) / "patterns.parquet"), split, None, "patterns")


def scan_attempts(parquet_dir: str | Path, *, split) -> pl.LazyFrame:
    """Attempts that have at least one transaction in `split` (an attempt may appear in several)."""
    splits = _splits(split)
    if not splits:
        raise ValueError("reading attempts needs split=")
    if "test" in splits:
        guards.require_final_run("the test split of attempts")
    lf = pl.scan_parquet(Path(parquet_dir) / "attempts.parquet")
    return lf.filter(pl.col("splits").list.eval(pl.element().is_in(splits)).list.any())


def read_accounts(parquet_dir: str | Path) -> pl.DataFrame:
    """node_id, bank, account, entity_id, entity_name. No labels or timestamps, so unguarded."""
    return pl.read_parquet(Path(parquet_dir) / "accounts.parquet")


def read_banks(views_dir: str | Path) -> pl.DataFrame:
    """rank, bank, n_tx, n_laundering of the view banks (whole-dataset counts; for planning only)."""
    return pl.read_parquet(Path(views_dir) / "banks.parquet")
