"""Registration of the agent graphs for scripts/run_llm_agent.py.

PORTED: `ported_group_d()` / `ported_group_c()` -> {system: Investigator class}, the StateGraph versions by default; FTM_LEGACY_AGENT_LOOPS=1 returns the
old hand-written loops of ftm/external (kept, parity-tested against the graphs, for A/B checks).
ADAPTED: `adapted_system(...)` -> a GraphInvestigator running the agent's PlannerPolicy inside the system graph (ftm.graph.pipeline).

Owner: T3 (Akshat).
"""

from __future__ import annotations

import os

from .adapted import ADAPTED_SYSTEMS, make_policy  # noqa: F401 - ADAPTED_SYSTEMS is the registry's public list

_GROUP_D = ("react", "reflexion", "rewoo", "lats", "llmcompiler", "dspy_react", "debate", "uot", "bed_llm", "mediq")


def _classes(module: str, legacy: bool) -> dict:
    import importlib

    names = {"react": ("react", "React"), "reflexion": ("reflexion", "Reflexion"), "rewoo": ("rewoo", "Rewoo"), "lats": ("lats", "Lats"),
             "llmcompiler": ("llmcompiler", "LlmCompiler"), "dspy_react": ("dspy_react", "DspyReact"),
             "debate": ("debate_autogen", "Debate"), "uot": ("uot", "Uot"), "bed_llm": ("bed_llm", "BedLlm"), "mediq": ("mediq", "MediQ")}
    mod, cls = names[module]
    return getattr(importlib.import_module(("ftm.external." if legacy else "ftm.graph.agents.") + mod), cls)


def legacy_loops() -> bool:
    return os.environ.get("FTM_LEGACY_AGENT_LOOPS") == "1"


def ported_group_d(legacy: bool | None = None) -> dict:
    legacy = legacy_loops() if legacy is None else legacy
    return {name: _classes(name, legacy) for name in _GROUP_D}


_GROUP_C = {"sharma": ("sharma", "Sharma"), "torres": ("torres", "Torres"), "co_investigator": ("co_investigator", "CoInvestigator"),
            "finfre_rag": ("finfre_rag", "FinfreRag")}


def ported_group_c(legacy: bool | None = None) -> dict:
    """{agent: class} for the Group C agents (each has native and ported modes, chosen by the `mode` argument)."""
    import importlib

    legacy = legacy_loops() if legacy is None else legacy
    return {name: getattr(importlib.import_module(("ftm.external." if legacy else "ftm.graph.agents.") + mod), cls)
            for name, (mod, cls) in _GROUP_C.items()}


def adapted_system(system: str, llm, *, menu=None, question_filter=None, fusion=None, seed: int = 0, extra: dict | None = None,
                   use_library: bool = False, program_path=None, threshold: float | None = None,
                   stop_cannot_cross: bool = False, rules_mode: str = "alert"):  # fmt: skip
    """`react_adapted` ... -> GraphInvestigator(policy). The menu is the accepted back-tested questions (as for `ours`); no composing.
    `threshold` (the decision config's) is shown to the policy and, with `stop_cannot_cross`, ends the alert once the remaining eps cannot cross it."""
    from ftm.graph.pipeline import GraphInvestigator

    kw = {"program_path": program_path} if system == "dspy_react_adapted" else {}
    policy = make_policy(system, seed=seed, extra=extra, **kw)
    return GraphInvestigator(policy, fusion, llm, questions=menu, question_filter=question_filter, use_library=use_library, allow_compose=False,
                             threshold=threshold, stop_cannot_cross=stop_cannot_cross, rules_mode=rules_mode)


def adapted_dev_responder(messages: list[dict], kw: dict) -> str:
    """DEV ONLY MockLLM responder for smoke runs of the adapted arms: ask option 1 once, then stop; other schemas get the smallest valid object."""
    import json

    from ftm.llm.jsonout import example_instance

    schema = kw.get("json_schema") or {}
    user = messages[-1]["content"]
    first = "(no question asked yet)" in user
    t = schema.get("title")
    if t == "adapted_choice":
        return json.dumps({"thought": "dev", "choice": 1 if first and schema["properties"]["choice"]["maximum"] >= 1 else 0})
    if t == "adapted_plan":
        return json.dumps({"plan": [{"why": "dev", "choice": 1}]})
    if t == "adapted_dag":
        return json.dumps({"tasks": [{"id": 1, "choice": 1, "deps": []}]})
    if t == "adapted_join":
        return json.dumps({"decision": "FINISH", "note": "dev"})
    if t == "adapted_eval":
        return json.dumps({"verdict": "CONFIDENT", "reason": "dev"})
    if t == "adapted_mediq_state":
        return json.dumps({"probability": 0.5, "confidence": 0.4 if first else 0.9})
    return json.dumps(example_instance(schema)) if schema else "MOCK"
