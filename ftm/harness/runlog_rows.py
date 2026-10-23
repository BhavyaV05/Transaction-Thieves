"""Verdicts to run-log rows (ftm/eval/runlog.py schema, v1) and Parquet. The orchestrator's log writer (C15).

Owner: T2 (Bhavya); the schema is T4's (`RUN_LOG_COLUMNS`, v1, hand-off #2).

One "step" row per question attempt (including gate refusals) and one "verdict" row per alert. The label goes in only
for val/test rows and only here, after the investigation is over. The v1 fields that need more than the verdict
(`p0`, `home_bank`, `eps_spent`, `sim_days`, `stop_reason`; `family`, `kind`, `log_lr`) are filled when the caller passes
`ctx`, `facts` and `pool`; without them they stay null, as in v0.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from ftm.eval.runlog import RUN_LOG_COLUMNS
from ftm.types import AlertContext, Reply, Verdict

_DTYPES = {"str": pl.String, "int": pl.Int64, "float": pl.Float64}


def verdict_rows(v: Verdict, base: dict, label: int | None = None, *, ctx: AlertContext | None = None, pool=None,
                 facts: dict | None = None) -> list[dict]:  # fmt: skip
    """`base`: run_id, commit, machine, system, variant, seed, split, alert_id.

    `ctx`: the alert (home_bank, p0). `pool`: the back-tested `QuestionPool`; an answer to a question in it gets its
    family and the log LR the fusion used (`log_lr` stays null for a question without an accepted LR).
    `facts`: `eps_spent`, `sim_days` and `stop_reason` of the investigation (from the toolbox and the planner)."""
    cands = {(c.program, c.eps): c for c in pool} if pool is not None else {}
    rows = []
    for e in v.questions_log:
        r = e.response
        kind = None if r is None else "reply" if isinstance(r, Reply) else type(r).__name__.lower()
        c = cands.get((e.program, e.eps))
        rows.append({**base, "event": "step", "step": e.step, "bank": e.bank, "program_canonical": e.program,
                     "eps": e.eps, "gate": e.gate_decision, "response_kind": kind,
                     "answer_value": json.dumps(r.value) if isinstance(r, Reply) else None,
                     "latency_days": getattr(r, "latency_days", None) if isinstance(r, Reply)
                     else getattr(r, "after_days", None),
                     "posterior": e.posterior_after,
                     "family": c.family if c else None, "kind": r.kind if isinstance(r, Reply) else None,
                     "log_lr": c.log_lr(r.value) if c and isinstance(r, Reply) else None,
                     "note_present": int(r.note is not None) if isinstance(r, Reply) else None})  # fmt: skip
    facts = facts or {}
    rows.append({**base, "event": "verdict", "probability": v.probability, "label": label, "tokens_in": v.tokens_in,
                 "tokens_out": v.tokens_out, "llm_calls": v.llm_calls, "gpu_seconds": v.gpu_seconds,
                 "home_bank": ctx.home_bank if ctx else None, "p0": ctx.p0 if ctx else None,
                 "eps_spent": facts.get("eps_spent"), "sim_days": facts.get("sim_days"),
                 "stop_reason": facts.get("stop_reason"),
                 "trials": v.trials if v.trials is not None else 1})  # fmt: skip
    return rows


def to_frame(rows: list[dict]) -> pl.DataFrame:
    schema = {c: _DTYPES[t] for c, t in RUN_LOG_COLUMNS.items()}
    return pl.DataFrame([{c: r.get(c) for c in RUN_LOG_COLUMNS} for r in rows], schema=schema)


def write_runlog(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_frame(rows).write_parquet(path)
    return path
