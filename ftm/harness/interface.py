"""The shared agent harness interface (C16). EVERY investigator, ours or external, implements `Investigator`.

CONTRACT. Frozen at week 4 (harness v1). Owner: T2 (Bhavya). Consumers: T3 (planner, Group D), T4 (gate, Group C).
`ask_bank` always goes through the compliance gate (T4, ftm/compliance) and the privacy accountant
(T2, ftm/privacy), so no system can exceed budgets. See plan.md 6.16.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ftm.types import AlertContext, BankResponse, DslProgram, Limits, Verdict


@runtime_checkable
class ToolBox(Protocol):
    """Tools an investigator may call. Which ones exist depends on the mode (native vs ported)."""

    def local_graph(self, query: str, as_of: str) -> list[dict]:
        """Rows from the home bank's own view, aliased. Must apply the as_of time guard."""
        ...

    def ask_bank(self, bank: str, program: DslProgram, eps: float) -> BankResponse:
        """Ported mode only. Gate + accountant sit in front of this."""
        ...

    def lookup_rules(self, query: str) -> list[dict]:
        """Rule snippets with rule IDs and jurisdiction (T4, regulatory RAG)."""
        ...

    def search_library(self, shape: dict) -> list[dict]:
        """Strategy library lookup by situation shape. Only for systems that use memory (T3)."""
        ...


@runtime_checkable
class Investigator(Protocol):
    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict: ...


class ScriptedToolBox:
    """Placeholder mock so T3/T4 can develop before harness v0 (week 3). T2 replaces this with the real
    mock mode. Returns the scripted responses in order and records the calls."""

    def __init__(self, responses: list[BankResponse] | None = None):
        self._responses = list(responses or [])
        self.calls: list[tuple[str, DslProgram, float]] = []

    def local_graph(self, query: str, as_of: str) -> list[dict]:
        return []

    def ask_bank(self, bank: str, program: DslProgram, eps: float) -> BankResponse:
        self.calls.append((bank, program, eps))
        if not self._responses:
            raise RuntimeError("ScriptedToolBox ran out of scripted responses")
        return self._responses.pop(0)

    def lookup_rules(self, query: str) -> list[dict]:
        return []

    def search_library(self, shape: dict) -> list[dict]:
        return []
