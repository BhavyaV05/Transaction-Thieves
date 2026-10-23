"""Run-log schema: one table that every machine writes and the eval code reads.

v1 (T4 proposal, 2026-10-23; hand-off #2 with T2). v0 columns are unchanged, so v0 writers keep working: missing
columns are simply null. One row per event; `event` is "step" (a question and its answer) or "verdict" (end of an
investigation). Stored as Parquet under data/runs/<run_id>/ (or outputs/runs/<run_id>/).
Types are only str | int | float (writers map them with one dict); booleans are int 0/1.

Who fills what: the run script fills the run/alert columns; the harness (T2) the step columns, from its own question
log; the eval code fills `label` (val only, test at G5) and never trusts a label written by a run.
"""

from __future__ import annotations

RUN_LOG_VERSION = 1

RUN_LOG_COLUMNS: dict[str, str] = {
    # --- run (same on every row of a run) ---
    "run_id": "str",  # unique per script invocation
    "commit": "str",  # git commit hash (also logged in MLflow)
    "machine": "str",  # t1-3070ti | t2-3050ti | t3-5070ti | t4-cpu; timing counts only from the reference machine
    "system": "str",  # e.g. ours_compose, react, torres_native, torres_ported, enumerator
    "variant": "str",  # ablation switch label
    "seed": "int",
    "split": "str",  # train | val | test
    # --- alert ---
    "alert_id": "str",
    "event": "str",  # step | verdict
    "step": "int",
    # --- step rows ---
    "bank": "str",
    "program_canonical": "str",
    "eps": "float",
    "gate": "str",  # allowed | blocked | over_budget
    "response_kind": "str",  # reply | decline | timeout
    "answer_value": "str",  # JSON-encoded typed value
    "latency_days": "float",
    "posterior": "float",
    # --- verdict rows ---
    "probability": "float",  # the system's own end-to-end probability; null = no verdict (scored as p0)
    "label": "int",  # ground truth, filled by the eval code for val/test only
    "tokens_in": "int",
    "tokens_out": "int",
    "llm_calls": "int",
    "gpu_seconds": "float",
    # ===== v1 additions =====
    # run / model (so a result can be traced to the exact serving set-up, plan.md 12.3)
    "log_version": "int",  # RUN_LOG_VERSION
    "model": "str",  # served model id, e.g. qwen3-8b@<revision>; null for non-LLM systems
    "thinking": "int",  # 0/1, thinking mode (plan.md 6.11)
    "mode": "str",  # native | ported (Group C), ported for the rest
    # alert (verdict rows)
    "home_bank": "str",
    "val_part": "str",  # fit | cal | eval for val alerts (T1); null otherwise
    "p0": "float",  # one-bank score the investigation started from
    "stop_reason": "str",  # budget | fast_clear | cannot_cross | finish | calls | time | error (ftm.decision)
    "eps_spent": "float",  # total eps charged for the alert
    "sim_days": "float",  # simulated time to the verdict
    "trials": "int",  # Reflexion trials, LATS expansions, ...; 1 for single-pass systems
    # step rows
    "family": "str",  # question family (back-test / fusion weight key); null for free-text tools
    "kind": "str",  # bool | count | kary | set_count
    "rule_id": "str",  # legal basis the gate used
    "gate_reason": "str",  # why the gate or accountant refused (null if allowed)
    "log_lr": "float",  # log LR the fusion used for this answer; null = no accepted back-test LR (not used)
    "note_present": "int",  # 0/1: the reply carried a partner note
    "injection_id": "str",  # E15: id of the planted injection (ftm.safety.injection), null otherwise
}

EVENTS = ("step", "verdict")
V0_COLUMNS = tuple(list(RUN_LOG_COLUMNS)[: list(RUN_LOG_COLUMNS).index("log_version")])
REQUIRED = ("run_id", "system", "split", "alert_id", "event")


def validate(columns, events) -> list[str]:
    """Problems with a run log (column names, event values). Empty list = fine. Extra columns are allowed."""
    problems = [f"missing required column {c!r}" for c in REQUIRED if c not in columns]
    bad = sorted({e for e in events if e not in EVENTS})
    if bad:
        problems.append(f"unknown event values {bad}")
    return problems
