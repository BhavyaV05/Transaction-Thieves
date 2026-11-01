"""State of the per-alert system graph (ftm.graph.pipeline). One AlertState per alert; nodes return partial updates.

Owner: T3 (Akshat).  Spec: LangGraph rewrite (wave 2), docs: rewrite graph spec.
"""

from __future__ import annotations

from typing import Any, TypedDict

from ftm.graph.policy import Action, PlannerView
from ftm.llm.budget import BudgetedClient
from ftm.llm.investigation import Investigation
from ftm.types import AlertContext, Limits, QuestionLogEntry, Verdict


class AlertState(TypedDict, total=False):
    alert: AlertContext
    limits: Limits
    tools: Any  # the ToolBox of this alert (one toolbox session per worker, see runner.py)
    budget: BudgetedClient  # per alert: Limits.max_llm_calls / max_tokens bind here
    session: Investigation  # the harness-side state: eps left, clock, question log, fusion
    view: PlannerView
    pending: list[Action]  # what the policy proposed this step
    results: list  # one AskResult per Ask of this step (pipeline.AskResult)
    log: list[QuestionLogEntry]
    posterior: float
    stop_reason: str | None
    verdict: Verdict
    scratch: dict  # per-policy private state (the pipeline adds "_state": this dict, for policies that need more than the view)
    step: int
    failures: int  # steps in a row without an allowed answer
    # prepared once
    evidence: str
    rules: list[dict]
    strategies: list[dict]
    menu_raw: list[dict]  # routed menu items as dicts ({id, program, family, eps, ..., bank}), the prompt of our planner shows these
    menu: list
    refused: dict  # (canonical, bank) the gate / accountant refused -> why
