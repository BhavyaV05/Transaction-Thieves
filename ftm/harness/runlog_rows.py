"""Verdicts to run-log rows (ftm/eval/runlog.py schema) and Parquet. The orchestrator's log writer (C15).

Owner: T2 (Bhavya); the schema is T4's (`RUN_LOG_COLUMNS`, draft v0, hand-off #2).

One "step" row per question attempt (including gate refusals) and one "verdict" row per alert. The label goes in only
for val/test rows and only here, after the investigation is over.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from ftm.eval.runlog import RUN_LOG_COLUMNS
from ftm.types import Reply, Verdict

_DTYPES = {"str": pl.String, "int": pl.Int64, "float": pl.Float64}


def verdict_rows(v: Verdict, base: dict, label: int | None = None) -> list[dict]:
    """`base`: run_id, commit, machine, system, variant, seed, split, alert_id."""
    rows = []
    for e in v.questions_log:
        r = e.response
        kind = None if r is None else "reply" if isinstance(r, Reply) else type(r).__name__.lower()
        rows.append({**base, "event": "step", "step": e.step, "bank": e.bank, "program_canonical": e.program,
                     "eps": e.eps, "gate": e.gate_decision, "response_kind": kind,
                     "answer_value": json.dumps(r.value) if isinstance(r, Reply) else None,
                     "latency_days": getattr(r, "latency_days", None) if isinstance(r, Reply)
                     else getattr(r, "after_days", None),
                     "posterior": e.posterior_after})  # fmt: skip
    rows.append({**base, "event": "verdict", "probability": v.probability, "label": label, "tokens_in": v.tokens_in,
                 "tokens_out": v.tokens_out, "llm_calls": v.llm_calls, "gpu_seconds": v.gpu_seconds})  # fmt: skip
    return rows


def to_frame(rows: list[dict]) -> pl.DataFrame:
    schema = {c: _DTYPES[t] for c, t in RUN_LOG_COLUMNS.items()}
    return pl.DataFrame([{c: r.get(c) for c in RUN_LOG_COLUMNS} for r in rows], schema=schema)


def write_runlog(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_frame(rows).write_parquet(path)
    return path
