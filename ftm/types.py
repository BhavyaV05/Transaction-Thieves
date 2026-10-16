"""Shared data types: the contract between tracks.

CONTRACT. These types are used by T1-T4. Until the week-4 harness freeze they change by agreement of the
owners involved; after the freeze, by agreement of all four (tracks.md section 8).
Owner: T2 (Bhavya). Draft v0, to be confirmed in week 1 with T3/T4.

Invariants these types support (plan.md section 3):
  - a partner bank returns a single noisy value, never rows;
  - LLM text never enters a decision, so `Reply.note` is kept apart from `Reply.value`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# A question is a DSL program in text form (grammar: ftm/dsl, owner T2).
DslProgram = str

AnswerKind = Literal["bool", "count", "kary", "set_count"]


@dataclass(frozen=True)
class AlertContext:
    """One alert as an investigator sees it. Account IDs are random aliases (plan.md invariant 4)."""

    alert_id: str
    home_bank: str
    as_of: str  # ISO timestamp; tools never see data after this
    split: Literal["train", "val", "test"]
    p0: float  # calibrated one-bank probability
    summary: dict = field(default_factory=dict)  # aliased fields only


@dataclass(frozen=True)
class Reply:
    """A partner bank's answer: one DP-noised value. `note` is untrusted text and must never reach fusion."""

    bank: str
    kind: AnswerKind
    value: float | int | bool | tuple
    eps_spent: float
    latency_days: float
    note: str | None = None


@dataclass(frozen=True)
class Decline:
    bank: str
    reason: str = ""


@dataclass(frozen=True)
class Timeout:
    bank: str
    after_days: float = 5.0


BankResponse = Reply | Decline | Timeout


@dataclass(frozen=True)
class Limits:
    """Budgets applied identically to every system (fairness protocol, plan.md 7.2)."""

    eps_per_alert: float = 2.0
    max_llm_calls: int = 20
    max_tokens: int = 6000  # per-prompt cap; chosen so it fits every serving machine (tracks.md 1.2)
    max_sim_time_days: float = 5.0


@dataclass
class QuestionLogEntry:
    step: int
    bank: str
    program: DslProgram
    eps: float
    response: BankResponse | None = None
    gate_decision: Literal["allowed", "blocked", "over_budget"] = "allowed"
    posterior_after: float | None = None


@dataclass
class Verdict:
    """What an investigator returns. `probability` is its own end-to-end answer (None if it gives none);
    the questions-only score is computed from `questions_log` through fusion by the eval code."""

    alert_id: str
    probability: float | None
    questions_log: list[QuestionLogEntry] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    llm_calls: int = 0
    gpu_seconds: float = 0.0
