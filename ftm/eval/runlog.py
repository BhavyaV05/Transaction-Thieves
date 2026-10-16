"""Run-log schema: one table that every machine writes and the eval code reads.

DRAFT v0. T2 (orchestrator logs) and T4 (eval) must agree on this in week 1 (hand-off #2 in tracks.md).
One row per event; `event` is "step" (a question and its answer) or "verdict" (end of an investigation).
Stored as Parquet under data/runs/<run_id>/.
"""

from __future__ import annotations

RUN_LOG_COLUMNS: dict[str, str] = {
    "run_id": "str",  # unique per script invocation
    "commit": "str",  # git commit hash (also logged in MLflow)
    "machine": "str",  # t1-3070ti | t2-3050ti | t3-5070ti | t4-cpu; timing counts only from the reference machine
    "system": "str",  # e.g. ours_compose, react, torres_native, torres_ported, enumerator
    "variant": "str",  # ablation switch label
    "seed": "int",
    "split": "str",  # train | val | test
    "alert_id": "str",
    "event": "str",  # step | verdict
    "step": "int",
    "bank": "str",
    "program_canonical": "str",
    "eps": "float",
    "gate": "str",  # allowed | blocked | over_budget
    "response_kind": "str",  # reply | decline | timeout
    "answer_value": "str",  # JSON-encoded typed value
    "latency_days": "float",
    "posterior": "float",
    "probability": "float",  # verdict rows: the system's own end-to-end probability
    "label": "int",  # ground truth, filled by the eval code for val/test only
    "tokens_in": "int",
    "tokens_out": "int",
    "llm_calls": "int",
    "gpu_seconds": "float",
}

EVENTS = ("step", "verdict")
