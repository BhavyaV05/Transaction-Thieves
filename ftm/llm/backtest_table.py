"""Adapter: T2's assembled back-test table (`scripts/backtest.py --assemble`) -> our LRTable and RealisticAcceptance.

Owner: T3 (Akshat).  Spec: plan.md 6.8 (realistic vs exact back-testing), tracks.md 4.6 (hand-off #9).

T2's `lr_table.jsonl` rows are exactly what `LRTable.from_rows` reads, but a row does not say which mode it was made
in. The mode lives in the cache entries next to it (`<dir>/*.json`, key "mode": "realistic" | "exact"), and the
cache directory name is the namespace that hashes the mode with the alert set and seed. So the loader takes the
assembled directory (or the `lr_table.jsonl` inside it), reads the mode from the entries and refuses a table whose
mode cannot be established or is not the one asked for. Distillation and the main results need `realistic`: exact mode
ignores the noise and accepts questions that are useless once it is added.

This module imports no ftm.backtest code (static check: nothing in ftm/llm may), it only reads T2's files.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ftm.llm.investigation import LRTable

TABLE_NAME = "lr_table.jsonl"
PLANNER_EPS_LEVELS = (0.25, 0.5, 1.0, 2.0)  # ftm.planners.llm_prompt.EPS_LEVELS, configs/compliance/gate.yaml


class BacktestTableError(ValueError):
    pass


@dataclass(frozen=True)
class BacktestTable:
    table: LRTable
    mode: str  # realistic | exact
    path: Path
    namespace: str
    rows: int
    accepted_by_eps: dict[float, int]
    eligible_by_eps: dict[float, int]

    def missing_eps(self, levels=PLANNER_EPS_LEVELS) -> list[float]:
        """Planner eps levels with no row at all: a question asked at such a level can never be accepted."""
        return [e for e in levels if not any(math.isclose(e, k, rel_tol=0, abs_tol=1e-9) for k in self.eligible_by_eps)]

    def summary(self) -> dict:
        return {"path": str(self.path), "namespace": self.namespace, "mode": self.mode, "rows": self.rows,
                "accepted_by_eps": {f"{k:g}": v for k, v in sorted(self.accepted_by_eps.items())},
                "eligible_by_eps": {f"{k:g}": v for k, v in sorted(self.eligible_by_eps.items())},
                "eps_levels_without_rows": self.missing_eps()}  # fmt: skip


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _cache_mode(d: Path) -> str | None:
    modes = set()
    for p in d.glob("*.json"):
        if len(p.stem) != 20 or any(c not in "0123456789abcdef" for c in p.stem):
            continue  # entries are <20 hex digits>.json (T2); report.json and the like are not entries
        m = json.loads(p.read_text(encoding="utf-8")).get("mode")
        if m is not None:
            modes.add(m)
    if len(modes) > 1:
        raise BacktestTableError(f"{d}: cache entries of several modes {sorted(modes)}")
    return modes.pop() if modes else None


def load_backtest(path: str | Path, mode: str | None = None, require: str | None = "realistic") -> BacktestTable:
    """Load an assembled table. `path` is the assembled directory or the `lr_table.jsonl` in it.

    The mode is read from the cache entries beside the table. `mode` (e.g. a CLI flag) is only a cross-check, or
    the stated mode when the entries are not there (a table copied away from its cache). `require` names the mode
    the caller can work with (default realistic); pass None to accept any (reports only)."""
    p = Path(path)
    d, f = (p, p / TABLE_NAME) if p.is_dir() else (p.parent, p)
    if not f.exists():
        raise BacktestTableError(f"{f} not found (run scripts/backtest.py ... --assemble)")
    found = _cache_mode(d)
    if found is not None and mode is not None and found != mode:
        raise BacktestTableError(f"{d} was made in {found!r} mode, not {mode!r}")
    got = found or mode
    if got is None:
        raise BacktestTableError(f"cannot establish the mode of {f}: no cache entries beside it and no mode given")
    if require is not None and got != require:
        raise BacktestTableError(f"{f} is a {got!r}-mode table; {require!r} is required (exact mode ignores the noise)")
    rows = _read_rows(f)
    acc: Counter = Counter()
    eli: Counter = Counter()
    for r in rows:
        eli[float(r["eps"])] += 1
        acc[float(r["eps"])] += bool(r["accepted"])
    return BacktestTable(LRTable.from_rows(rows), got, f, d.name, len(rows), dict(acc), dict(eli))


class TableAcceptance:
    """`RealisticAcceptance` (ftm.planners.llm_distill) over a realistic-mode back-test table: a question passes iff
    its (canonical program, eps) row is accepted by e-BH. Anything not tested, ineligible or at an eps level with no
    rows (T2 did not back-test it) is not accepted."""

    def __init__(self, bt: BacktestTable):
        if bt.mode != "realistic":
            raise BacktestTableError(f"acceptance needs a realistic-mode table, not {bt.mode!r}")
        self.bt = bt

    def __call__(self, canonical: str, eps: float) -> bool:
        e = self.bt.table.get(canonical, eps)
        return e is not None and e.accepted
