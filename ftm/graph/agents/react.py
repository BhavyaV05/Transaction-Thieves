"""ReAct (Yao et al. 2023) as a StateGraph, ported mode. Parity with ftm/external/react.py is tested on the mock LLM.

    init -> act -(more?)-> act ... -> wrap_up -> verdict
`act` is one Thought/Action/Observation turn (one LLM call); the edge after it is the old `while` condition.
The node functions take the agent so Reflexion reuses them.
"""

from __future__ import annotations

from ftm.external.react import React as _Legacy
from ftm.external.text_agent import parse_tool_call
from ftm.harness.interface import ToolBox
from ftm.llm.budget import BudgetExceeded, PromptTooLong
from ftm.types import AlertContext, Limits, Verdict

from ._base import END, START, new_graph, run_graph


def more(s) -> bool:
    """The old episode `while`: not finished, calls left, simulated time left, and the last call did not fail."""
    return not s.get("halt") and s["ex"].finished is None and s["budget"].calls_left > 0 and not s["session"].out_of_time


def act(agent, s) -> dict:
    """One turn: the step counter moves before the call (as in the old loop), a capped/over-long call ends the episode."""
    step, messages, ex, budget = s["step"] + 1, s["messages"], s["ex"], s["budget"]
    try:
        resp = budget.chat(messages, max_tokens=agent.max_tokens_per_step, temperature=agent.temperature,
                           stop=["\nObservation", "Observation:"], seed=agent.seed, extra=agent.extra)  # fmt: skip
    except (BudgetExceeded, PromptTooLong):
        return {"step": step, "halt": True}
    thought_action = resp.text.strip()
    obs = ex.run(parse_tool_call(thought_action), step)
    messages = [*messages, {"role": "assistant", "content": thought_action}]
    if ex.finished is None:
        messages.append({"role": "user", "content": f"Observation {step}: {obs}\nThought {step + 1}:"})
    return {"step": step, "messages": messages}


def wrap_up(agent, s) -> dict:
    agent.wrap_up(s["messages"], s["ex"], s["budget"], s["step"])
    return {}


class React(_Legacy):
    def graph(self):
        if getattr(self, "_graph", None) is None:
            g = new_graph()
            g.add_node("init", self._init)
            g.add_node("act", lambda s: act(self, s))
            g.add_node("wrap_up", lambda s: wrap_up(self, s))
            g.add_node("verdict", self._verdict)
            g.add_edge(START, "init")
            g.add_conditional_edges("init", lambda s: "act" if more(s) else "wrap_up", ["act", "wrap_up"])
            g.add_conditional_edges("act", lambda s: "act" if more(s) else "wrap_up", ["act", "wrap_up"])
            g.add_edge("wrap_up", "verdict")
            g.add_edge("verdict", END)
            self._graph = g.compile()
        return self._graph

    def _init(self, s) -> dict:
        session, budget, ex = self.start(s["alert"], s["tools"], s["limits"], self.name)
        return {"session": session, "budget": budget, "ex": ex, "messages": self.first_messages(s["alert"], ex), "step": 0}

    def _verdict(self, s) -> dict:
        self.last_trace = s["ex"].log
        return {"verdict": s["session"].verdict(s["ex"].finished, s["budget"].usage)}

    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict:
        return run_graph(self.graph(), {"alert": alert, "tools": tools, "limits": limits})["verdict"]

    investigate_legacy = _Legacy.investigate
