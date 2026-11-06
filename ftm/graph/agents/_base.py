"""Shared plumbing of the PORTED agent graphs: each published agent loop as a LangGraph StateGraph.

A ported graph reproduces the loop of ftm/external/<agent>.py node by node (same prompts, parsing, stopping, wrap-up,
verdict -> probability): the node bodies are the loop bodies, the conditional edges are the loop conditions. The old
class stays as the base class (prompts, parsers and helpers are shared, not copied) and its loop stays reachable as
`investigate_legacy`, so tests can run both on the same mock LLM and compare.

Per-alert objects (session, budget, executor) live in the graph state; the compiled graph itself holds none, so one
instance serves concurrent alerts.

Owner: T3 (Akshat).
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph  # noqa: F401 - re-exported for the agent modules

RECURSION_LIMIT = 10_000  # loops here are bounded by the call caps; langgraph's default of 25 would cut them


class AgentState(TypedDict, total=False):
    """Union of every ported agent's fields (a TypedDict, so each graph uses the keys it needs)."""

    alert: Any
    tools: Any
    limits: Any
    session: Any
    budget: Any
    ex: Any
    messages: list
    step: int
    halt: bool  # a BudgetExceeded / PromptTooLong ended the loop
    stop: Any  # agent-specific stop reason (None = keep going)
    verdict: Any
    scratch: dict  # agent-specific private fields


def run_graph(graph, state: dict) -> dict:
    return graph.invoke(state, {"recursion_limit": RECURSION_LIMIT})


def new_graph() -> StateGraph:
    return StateGraph(AgentState)
