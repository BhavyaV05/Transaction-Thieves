"""LLM question batches for back-testing (hand-off #8, T3 -> T2, weekly from week 4).

Owner: T3 (Akshat).  Spec: tracks.md 2.1 row 8 and 5.4.

One JSONL row per composed question (selected library questions are already back-tested). T2 dedups on
`program_canonical` + `eps`, back-tests, and returns the back-test table (hand-off #9). Rows hold the question
and planner metadata only: no rationale text, no labels, no account aliases (the DSL has slots, not accounts).
Questions from test alerts are not exported outside the final run (they go through the online e-value rule).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ftm import guards

BATCH_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BatchRow:
    program_canonical: str
    kind: str
    family: str | None
    eps: float
    bank: str
    alert_id: str
    split: str
    step: int
    mode: str
    typology_guess: str | None
    model: str
    run_id: str
    seed: int
    schema_version: int = BATCH_SCHEMA_VERSION


class QuestionBatchWriter:
    def __init__(self, path: str | Path, run_id: str, model: str, seed: int = 0):
        self.path = Path(path)
        self.run_id, self.model, self.seed = run_id, model, seed
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n = 0

    def add(self, *, canonical: str, kind: str, family: str | None, eps: float, bank, alert_id: str, split: str,
            step: int, mode: str, typology_guess: str | None) -> None:  # fmt: skip
        if split == "test":
            guards.require_final_run("exporting questions from test alerts")
        row = BatchRow(canonical, kind, family, float(eps), str(bank), alert_id, split, step, mode, typology_guess,
                       self.model, self.run_id, self.seed)  # fmt: skip
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(row)) + "\n")
        self.n += 1


def read_batch(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def unique_questions(rows: list[dict]) -> list[dict]:
    """One row per (program_canonical, eps), keeping the first, with a `n_alerts` count. What T2 back-tests."""
    seen: dict[tuple, dict] = {}
    for r in rows:
        k = (r["program_canonical"], r["eps"])
        if k not in seen:
            seen[k] = {**r, "n_alerts": 0}
        seen[k]["n_alerts"] += 1
    return list(seen.values())
