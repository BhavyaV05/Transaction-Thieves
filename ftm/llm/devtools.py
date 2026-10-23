"""DEV ONLY: a random-answer ToolBox so LLM loops run end to end before T2's harness and banks exist.

Owner: T3 (Akshat).  Replaced by the harness mock mode (T2, harness v0 week 3). Never used for results:
answers are random draws by answer type, not executions of the question on a ledger, and there is no DP noise,
gate or accountant. `scripts/run_llm_agent.py --toolbox dev` marks every run it makes as `variant=devtools`.
"""

from __future__ import annotations

import json
import random
import re

from ftm.dsl.parser import parse
from ftm.dsl.spec import PRIMS
from ftm.planners.llm_dsl import DslAdapter, V0Dsl
from ftm.types import BankResponse, Decline, Reply, Timeout

DEV_RULES = [
    {"rule_id": "US-314b-1", "text": "Section 314(b): financial institutions may share information with each other "
                                      "to identify and report possible money laundering.", "jurisdiction": "US"},
    {"rule_id": "SG-COSMIC-1", "text": "COSMIC: participant banks may request risk information on a customer "
                                        "when there are red flags of financial crime.", "jurisdiction": "SG"},
]  # fmt: skip


class DevRandomToolBox:
    def __init__(self, seed: int = 0, p_decline: float = 0.05, p_timeout: float = 0.02, dsl: DslAdapter | None = None):
        self.rng = random.Random(seed)
        self.p_decline, self.p_timeout = p_decline, p_timeout
        self.dsl = dsl or V0Dsl()
        self.calls: list[tuple] = []

    def local_graph(self, query: str, as_of: str) -> list[dict]:
        return []

    def ask_bank(self, bank, program: str, eps: float) -> BankResponse:
        self.calls.append((bank, program, eps))
        u = self.rng.random()
        if u < self.p_decline:
            return Decline(bank=str(bank), reason="dev decline")
        if u < self.p_decline + self.p_timeout:
            return Timeout(bank=str(bank))
        check = self.dsl.validate(program)
        kind = check.kind or "bool"
        if kind == "bool":
            value = self.rng.random() < 0.5
        elif kind == "kary":
            try:
                banks = list(parse(program).arg("banks"))
            except Exception:  # noqa: BLE001 - dev only: another DSL's syntax
                banks = [0]
            value = self.rng.choice(banks)
        else:
            value = max(0, round(self.rng.gauss(3, 2)))
        return Reply(bank=str(bank), kind=kind, value=value, eps_spent=eps,
                     latency_days=min(4.0, self.rng.expovariate(2.0)))  # fmt: skip

    def lookup_rules(self, query: str) -> list[dict]:
        return list(DEV_RULES)

    def search_library(self, shape: dict) -> list[dict]:
        return []


def dev_responder(messages: list[dict], kw: dict, n_questions: int = 2) -> str:
    """DEV ONLY MockLLM responder: asks `n_questions` example questions, then stops / finishes.

    Structured requests (our planner) get planner JSON built from the schema's enums; free-text requests (Group D
    text loops) get a ReAct-style Thought/Action.
    """
    examples = [p.example for p in PRIMS.values() if not p.restricted]
    schema = kw.get("json_schema")
    if schema:
        user = messages[-1]["content"]
        asked = user.split("## Answers so far", 1)[-1].split("## Output", 1)[0].count(", eps ")
        props = schema["properties"]
        banks = props["bank"]["anyOf"][0]["enum"] if "anyOf" in props["bank"] else []
        eps = [e for e in props["eps"]["enum"] if e > 0]
        out = {k: None for k in props}
        if "hypothesis" in props:
            out["hypothesis"] = {"typology": "fan_in", "rationale": "dev responder"}
        if "belief" in props:
            out["belief"] = 0.5
        out["eps"] = 0.0
        if asked < n_questions and banks and eps and "compose" in props["action"]["enum"]:
            out.update(action="compose", program=examples[asked % len(examples)], bank=banks[0], eps=eps[0])
        else:
            out["action"] = "stop"
        return json.dumps(out)
    system = messages[0]["content"]
    if "variable #E" in system:  # ReWOO planner
        bank = _first_bank(messages)
        return "\n".join(f"Plan: check {i}\n#E{i + 1} = ask_bank[{bank} | 0.5 | {examples[i]}]" for i in range(n_questions))
    if "as finish[p]" in system or system.startswith("You see an anti-money-laundering alert, the tasks"):  # solvers
        return "finish[0.4]"
    if "utmost parallelizability" in system:  # LLMCompiler planner
        bank = _first_bank(messages)
        return "\n".join(f"{i + 1}. ask_bank[{bank} | 0.5 | {examples[i]}]" for i in range(n_questions)) + f"\n{n_questions + 1}. join()"
    if "[[ ## completed ## ]]" in "".join(m["content"] for m in messages):  # DSPy ChatAdapter
        if "next_tool_name" not in system:  # the final extraction step
            return "[[ ## reasoning ## ]]\ndev\n\n[[ ## probability ## ]]\n0.4\n\n[[ ## completed ## ]]"
        return "[[ ## next_thought ## ]]\ndev\n\n[[ ## next_tool_name ## ]]\nfinish\n\n[[ ## next_tool_args ## ]]\n{}\n\n[[ ## completed ## ]]"
    if system.startswith("You are checking"):  # Reflexion evaluator
        return "CONFIDENT. The answers agree."
    asked = sum(1 for m in messages if m["role"] == "user" and m["content"].startswith("Observation"))
    if asked < n_questions:
        return f"I should check the receiver.\nAction: ask_bank[{_first_bank(messages)} | 0.5 | {examples[asked]}]"
    return "The answers are mixed.\nAction: finish[0.4]"


def _first_bank(messages: list[dict]) -> str:
    m = re.search(r"Partner banks: ([0-9]+)", messages[0]["content"])
    return m.group(1) if m else "0"
