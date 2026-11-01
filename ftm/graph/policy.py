"""The planner-policy interface of the system graph (ftm.graph.pipeline).

Owner: T3 (Akshat).  Spec: LangGraph rewrite (wave 2).

A policy only PROPOSES: it sees a `PlannerView` and returns Asks (a question to put to a bank) or a Stop. Everything else (the bank
routing, the gate and accountant, the fusion that sets the probability, the stop rules) belongs to the pipeline, so every adapted
arm shares one harness path and its verdict is the fused posterior, never an LLM number.

Optional hooks (the pipeline calls them when the policy has them; the interface below is all a policy must implement):
  observe(results, scratch)  after the asks went out: `results` is one `AskResult` per Ask, in order (see pipeline.py);
  finish(state, scratch)     once, at the end: may return a probability that replaces the fused posterior (only ours, in the
                             llm_decides / no_backtest ablations; adapted arms never do);
  `scratch["_state"]`        the live AlertState (alert, session, tools, limits, budget), for policies that need more than the view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ftm.llm.budget import BudgetedClient


@dataclass(frozen=True)
class MenuItem:  # one routed, back-tested question
    qid: str
    canonical: str
    eps: float
    bank: str
    family: str
    lr_bins: list[tuple[str, float]]  # (answer bin label, LR)


@dataclass(frozen=True)
class Ask:  # bank is filled by the pipeline from routing, never by the policy
    canonical: str
    eps: float
    rationale: str = ""


@dataclass(frozen=True)
class Stop:
    reason: str = ""


Action = Ask | Stop


@dataclass
class PlannerView:
    alert_summary: dict
    p0: float
    posterior: float
    threshold: float | None
    evidence: str
    rules: list[dict]
    strategies: list[dict]
    menu: list[MenuItem]
    answers: list[dict]  # answers so far (program, bank, value, lr or None)
    eps_left: float
    calls_left: int
    step: int
    allow_compose: bool  # adapted arms: False (menu only) unless the policy is ours/compose


class PlannerPolicy(Protocol):
    name: str

    def propose(self, view: PlannerView, llm: BudgetedClient, scratch: dict) -> list[Action]:
        """One element = a sequential step; several Asks = one wave (the pipeline enforces the wave rules: the eps fit
        together, no two questions of one family)."""
        ...
