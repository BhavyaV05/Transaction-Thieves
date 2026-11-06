"""Sharma et al. as a StateGraph, ported modes (native and ported, agent and governed variants). Parity with ftm/external/sharma.py is tested.

Owner: T3 (Akshat).

    init -> step -(more?)-> step ... -> verdict
`step` is one iteration of the official bounded tool loop (one LLM call, one tool call). The old class is the base class: prompts,
evidence tools, `_ask`, the governed policy and the verdict map are shared, and its loop stays reachable as `investigate_legacy`.
"""

from __future__ import annotations

import json

from ftm.external.sharma import (
    EVIDENCE_TOOLS,
    MAX_ITERATIONS,
    NUDGE,
    OUTPUT,
    PORTED_ADDENDUM,
    SYSTEM_PROMPT,
    VERDICTS,
    _schema,
    governed,
)
from ftm.external.sharma import Sharma as _Legacy
from ftm.harness.interface import ToolBox
from ftm.llm.budget import BudgetedClient, BudgetExceeded, PromptTooLong
from ftm.llm.investigation import Investigation
from ftm.llm.jsonout import JsonOutputError, parse_and_validate
from ftm.types import AlertContext, Limits, Verdict

from ._base import END, START, new_graph, run_graph


def _more(s) -> str:
    """The old `for step in range(1, N + 1)` plus the `break`s."""
    return "verdict" if s["stop"] or s["step"] >= s["scratch"]["n_max"] else "step"


class Sharma(_Legacy):
    def graph(self):
        if getattr(self, "_graph", None) is None:
            g = new_graph()
            for name, fn in (("init", self._init), ("step", self._step), ("verdict", self._verdict)):
                g.add_node(name, fn)
            g.add_edge(START, "init")
            g.add_conditional_edges("init", _more, ["step", "verdict"])
            g.add_conditional_edges("step", _more, ["step", "verdict"])
            g.add_edge("verdict", END)
            self._graph = g.compile()
        return self._graph

    def _init(self, s) -> dict:
        alert, limits = s["alert"], s["limits"]
        session = Investigation(alert, s["tools"], limits, fusion=self.fusion)
        budget = BudgetedClient(self.llm, limits, tag=f"{self.name}:{alert.alert_id}")
        banks = list(alert.summary.get("partner_banks", [])) if self.mode == "ported" else []
        tool_names = [*EVIDENCE_TOOLS, *(["ask_bank"] if self.mode == "ported" else []), "submit_verdict"]
        system = SYSTEM_PROMPT + (PORTED_ADDENDUM.format(dsl=self.dsl.reference_text()) if banks else "")
        system += "\n\n" + OUTPUT.format(tools=json.dumps(tool_names))
        messages = [{"role": "system", "content": system}, {"role": "user", "content": f"Investigate case: {alert.alert_id}"}]
        return {"session": session, "budget": budget, "messages": messages, "step": 0, "stop": False,
                "scratch": {"banks": banks, "tool_names": tool_names, "trace": [], "final": None,
                            "n_max": min(MAX_ITERATIONS, limits.max_llm_calls)}}  # fmt: skip

    def _step(self, s) -> dict:
        sc, session, budget, alert = s["scratch"], s["session"], s["budget"], s["alert"]
        step, messages, trace = s["step"] + 1, list(s["messages"]), list(sc["trace"])
        schema = _schema(sc["tool_names"], sc["banks"], session.eps_left)
        try:
            resp = budget.chat(messages, max_tokens=self.max_tokens, temperature=self.temperature, json_schema=schema,
                               seed=self.seed, extra=self.extra)  # fmt: skip
        except (BudgetExceeded, PromptTooLong):
            return {"step": step, "stop": True}
        try:
            obj = parse_and_validate(resp.text, schema)
        except JsonOutputError as e:
            messages += [{"role": "assistant", "content": resp.text}, {"role": "user", "content": NUDGE}]
            trace.append({"step": step, "type": "text", "content": resp.text[:500], "error": str(e)})
            return {"step": step, "messages": messages, "scratch": {**sc, "trace": trace}}
        tool = obj["tool"]
        messages.append({"role": "assistant", "content": json.dumps(obj)})
        if tool == "submit_verdict":
            if obj.get("verdict") not in VERDICTS:
                messages.append({"role": "user", "content": "submit_verdict needs a verdict and a rationale."})
                return {"step": step, "messages": messages}
            final = {"verdict": obj["verdict"], "rationale": obj.get("rationale") or ""}
            trace.append({"step": step, "type": "verdict", "content": final})
            return {"step": step, "messages": messages, "stop": True, "scratch": {**sc, "trace": trace, "final": final}}
        result = self._ask(obj, session, sc["banks"], step) if tool == "ask_bank" else self._evidence(tool, alert)
        trace.append({"step": step, "type": "tool_call", "name": tool, "result": result})
        messages.append({"role": "user", "content": f"Tool result ({tool}): " + (result if isinstance(result, str) else json.dumps(result))})
        return {"step": step, "messages": messages, "scratch": {**sc, "trace": trace}}

    def _verdict(self, s) -> dict:
        sc, alert = s["scratch"], s["alert"]
        self.last_trace = sc["trace"]
        final, probability = sc["final"], None
        if final is not None:
            v = governed(final["verdict"], alert.p0) if self.variant == "governed" else final["verdict"]
            final["final"] = v
            probability = self.verdict_map.probability(v, alert.p0)
        self.last_verdict = final
        return {"verdict": s["session"].verdict(probability, s["budget"].usage)}

    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict:
        return run_graph(self.graph(), {"alert": alert, "tools": tools, "limits": limits})["verdict"]

    investigate_legacy = _Legacy.investigate
