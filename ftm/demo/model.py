"""The data the demo renders: one `Demo` = one alert + one `ArmRun` per system (plus the p0-only baseline).

Owner: T3 (Akshat).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class DemoStep:
    """One step of an investigation. Every field except `n` and `kind` may be None = not recorded (shown as such)."""

    n: int
    kind: str = "ask"  # ask | rejected (the planner's proposal never went out) | stop
    thought: str | None = None
    action: str | None = None  # select | compose | choice 3 | stop ...
    program: str | None = None
    gloss: str | None = None
    bank: str | None = None
    bank_why: str | None = None
    eps: float | None = None
    gate: str | None = None  # allowed | blocked | over_budget
    rule_id: str | None = None
    rule_text: str | None = None
    rule_derived: bool = False  # True: re-derived from the gate config, not logged by the run
    gate_reason: str | None = None
    eps_spent: float | None = None
    eps_left: float | None = None
    outcome: str | None = None  # reply | decline | timeout | blocked | over_budget
    answer_value: object = None
    answer_kind: str | None = None
    latency_days: float | None = None
    bin_label: str | None = None
    lr: float | None = None
    used: bool | None = None  # did the LR move the posterior
    post_before: float | None = None
    post_after: float | None = None
    note: str | None = None


@dataclass
class ArmRun:
    name: str  # ours_compose | react_adapted | p0_only
    label: str
    p0: float
    final_p: float
    steps: list[DemoStep] = field(default_factory=list)
    eps_budget: float | None = None
    eps_spent: float | None = None
    llm_calls: int | None = None
    stop_reason: str | None = None
    model: str | None = None
    mock: bool = False
    notes: list[str] = field(default_factory=list)  # what the source did not record (graceful degradation)

    @property
    def n_asked(self) -> int:
        return sum(1 for s in self.steps if s.kind == "ask" and s.gate == "allowed")


@dataclass
class AlertView:
    alert_id: str
    split: str
    home_bank: str | None
    p0: float
    summary: dict = field(default_factory=dict)  # aliased fields, no evidence text
    evidence: str | None = None
    evidence_view: str | None = None
    label: int | None = None  # train / val only; shown only with --reveal-label
    notes: list[str] = field(default_factory=list)


@dataclass
class Demo:
    alert: AlertView
    arms: list[ArmRun]
    threshold: float = 0.5  # escalate at or above
    clear_below: float | None = None  # None: the deployed decision is two-way (escalate at the threshold, else clear); a value adds a review band
    threshold_source: str | None = None  # the decision config the threshold came from (shown on the decision panel)


def zone(p: float, threshold: float, clear_below: float | None = None) -> str:
    """auto-clear | human review | escalate. Without `clear_below` there is no review band."""
    if p >= threshold:
        return "escalate"
    return "auto-clear" if clear_below is None or p < clear_below else "human review"


def odds(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return p / (1 - p)


def lr_effect(lr: float | None) -> str:
    if lr is None:
        return "no back-tested LR"
    if math.isclose(lr, 1.0, abs_tol=0.02):
        return "no effect"
    return "raises the odds" if lr > 1 else "lowers the odds"
