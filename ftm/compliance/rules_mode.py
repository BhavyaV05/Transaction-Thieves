"""How the rules reach a system: the `--rules` switch of the RAG bench (docs/t3_rag_bench.md).

Owner: T3 (Akshat).

  none   no rules at all (the WITHOUT arm)
  fixed  the one fixed sentence for every alert (`FIXED_RULES_QUERY`, the planner's original query, audit finding 15), top k
  alert  `rules_query(alert)`, top k (the default)
  whole  the whole corpus (`all_rules()` of the toolbox: every rule and red flag of the home jurisdiction), cut to the prompt
         cap by the shared trimmer (`ftm.planners.llm_prompt.build_messages_with_estimate`), which records how many fit

The graph pipeline (planner and the adapted Group D arms) calls `fetch_rules` in its prepare node. Systems that call
`tools.lookup_rules(<their own query>)` themselves (Group D loops, Torres, Co-Investigator) get `RulesModeToolBox`, which applies the
same four modes to their tool call.
"""

from __future__ import annotations

from ftm.compliance.query import FIXED_RULES_QUERY, rules_query

RULES_MODES = ("none", "fixed", "alert", "whole")
RULES_K = 5


def check_mode(mode: str) -> str:
    if mode not in RULES_MODES:
        raise ValueError(f"rules mode {mode!r}: choose one of {RULES_MODES}")
    return mode


def whole_rules(tools) -> list[dict]:
    fn = getattr(tools, "all_rules", None)
    if fn is None:
        raise RuntimeError(f"--rules whole needs a toolbox with all_rules() (the harness toolbox has one); got {type(tools).__name__}")
    return list(fn())


def fetch_rules(tools, mode: str, alert, gate=None, k: int = RULES_K) -> list[dict]:
    """The raw rule documents a system is offered for `alert` under `mode`."""
    check_mode(mode)
    if mode == "none":
        return []
    if mode == "whole":
        return whole_rules(tools)
    query = FIXED_RULES_QUERY if mode == "fixed" else rules_query(alert, gate)
    return list(tools.lookup_rules(query))[:k]


class RulesModeToolBox:
    """Applies a rules mode to a system's own `lookup_rules(query)` calls. `alert` mode is a pass-through (the system's query)."""

    def __init__(self, tools, mode: str):
        self._tools, self.rules_mode = tools, check_mode(mode)

    def __getattr__(self, name):
        return getattr(self._tools, name)

    def lookup_rules(self, query: str) -> list[dict]:
        if self.rules_mode == "none":
            return []
        if self.rules_mode == "whole":
            return whole_rules(self._tools)
        return self._tools.lookup_rules(FIXED_RULES_QUERY if self.rules_mode == "fixed" else query)
