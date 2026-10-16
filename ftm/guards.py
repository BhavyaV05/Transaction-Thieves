"""Safety guards shared by every track. Do not bypass these.

1. Test-split lock (plan.md 4.3): the test split loads only when FTM_FINAL_RUN=1, which is set at G5 only.
2. Time guard (plan.md 5.2 step 6): every tool query takes `as_of` and sees only timestamp <= as_of.

Owner: T4 (test-split guard, CI), T1 (time guard on data). Changes need review from both.
"""

from __future__ import annotations

import os
from typing import Any

FINAL_RUN_ENV = "FTM_FINAL_RUN"


class TestSplitLocked(RuntimeError):
    """Raised when the test split is requested outside the final run."""

    __test__ = False  # not a pytest test class


class TimeGuardError(ValueError):
    """Raised when a query would see data after `as_of`."""


def final_run_enabled() -> bool:
    return os.environ.get(FINAL_RUN_ENV) == "1"


def require_final_run(what: str = "the test split") -> None:
    """Call at the top of every function that reads test data or test labels."""
    if not final_run_enabled():
        raise TestSplitLocked(
            f"Refusing to load {what}: the test split is locked until G5. "
            f"Set {FINAL_RUN_ENV}=1 only in the agreed final-run script."
        )


def assert_not_future(max_timestamp: Any, as_of: Any) -> None:
    """Raise if a result contains a timestamp after `as_of`. Works on any comparable values."""
    if max_timestamp is not None and max_timestamp > as_of:
        raise TimeGuardError(f"data at {max_timestamp!r} is after as_of={as_of!r}")


def enforce_as_of(frame: Any, as_of: Any, column: str = "timestamp") -> Any:
    """Filter a Polars DataFrame/LazyFrame to timestamp <= as_of. Every tool must call this."""
    import polars as pl

    if as_of is None:
        raise TimeGuardError("as_of is required")
    return frame.filter(pl.col(column) <= as_of)
