"""Reflexion (Shinn et al. 2023) as a StateGraph, ported mode. Parity with ftm/external/reflexion.py is tested.

    init -> trial -> act* -> wrap_up -> (finished? room?) -> evaluate -> reflect -> trial ... -> verdict
The inner Thought/Action/Observation loop is ReAct's `act` node.

Owner: T3 (Akshat).
"""

from __future__ import annotations

from ftm.external.reflexion import EVALUATE, REFLECT, render
from ftm.external.reflexion import Reflexion as _Legacy
from ftm.harness.interface import ToolBox
from ftm.llm.investigation import describe
from ftm.planners.llm_prompt import EPS_LEVELS
from ftm.types import AlertContext, Limits, Verdict

from ._base import END, START, new_graph, run_graph
from .react import act, more, wrap_up


class Reflexion(_Legacy):
    def graph(self):
        if getattr(self, "_graph", None) is None:
            g = new_graph()
            g.add_node("init", self._init)
            g.add_node("trial", self._trial)
            g.add_node("act", lambda s: act(self, s))
            g.add_node("wrap_up", lambda s: wrap_up(self, s))
            g.add_node("decide", self._decide)
            g.add_node("evaluate", self._evaluate)
            g.add_node("reflect", self._reflect)
            g.add_node("verdict", self._verdict)
            g.add_edge(START, "init")
            g.add_edge("init", "trial")
            g.add_conditional_edges("trial", lambda s: "act" if more(s) else "wrap_up", ["act", "wrap_up"])
            g.add_conditional_edges("act", lambda s: "act" if more(s) else "wrap_up", ["act", "wrap_up"])
            g.add_edge("wrap_up", "decide")
            g.add_conditional_edges("decide", lambda s: "evaluate" if s["scratch"]["retry_check"] else "verdict", ["evaluate", "verdict"])
            g.add_conditional_edges("evaluate", lambda s: "reflect" if s["scratch"]["retry"] else "verdict", ["reflect", "verdict"])
            g.add_conditional_edges("reflect", lambda s: "trial" if s["scratch"]["lesson_ok"] else "verdict", ["trial", "verdict"])
            g.add_edge("verdict", END)
            self._graph = g.compile()
        return self._graph

    def _init(self, s) -> dict:
        session, budget, ex = self.start(s["alert"], s["tools"], s["limits"], self.name)
        return {"session": session, "budget": budget, "ex": ex, "step": 0,
                "scratch": {"memory": [], "final": None, "trial": 0, "retry_check": False, "retry": False, "lesson_ok": False}}  # fmt: skip

    def _trial(self, s) -> dict:
        ex, session, sc = s["ex"], s["session"], s["scratch"]
        ex.finished = None
        trial = sc["trial"] + 1
        preface = ""
        if sc["memory"]:
            preface += "Reflections from your earlier attempts on this alert:\n" + "\n".join(f"- {m}" for m in sc["memory"]) + "\n\n"
        if session.log:
            preface += ("Answers already obtained (real; asking again is refused):\n"
                        + "\n".join(f"- {e.program} to bank {e.bank}: {describe(e)}" for e in session.log) + "\n\n")  # fmt: skip
        return {"messages": self.first_messages(s["alert"], ex, preface), "halt": False, "scratch": {**sc, "trial": trial}}

    def _decide(self, s) -> dict:
        ex, session, budget, sc = s["ex"], s["session"], s["budget"], s["scratch"]
        if ex.finished is None:  # calls or time ran out inside the trial
            return {"scratch": {**sc, "retry_check": False}}
        out_of_room = (budget.calls_left < self.min_calls_for_retry or session.out_of_time
                       or session.eps_left + 1e-9 < min(EPS_LEVELS))  # fmt: skip
        again = not (sc["trial"] == self.max_trials or out_of_room)
        return {"scratch": {**sc, "final": ex.finished, "retry_check": again}}

    def _evaluate(self, s) -> dict:
        judgement = self._ask(s["budget"], EVALUATE, render(s["messages"]), 48)
        return {"scratch": {**s["scratch"], "retry": judgement is not None and judgement.upper().startswith("RETRY")}}

    def _reflect(self, s) -> dict:
        sc = s["scratch"]
        lesson = self._ask(s["budget"], REFLECT, render(s["messages"]), self.reflection_tokens)
        if lesson is None:
            return {"scratch": {**sc, "lesson_ok": False}}
        return {"scratch": {**sc, "lesson_ok": True, "memory": [*sc["memory"], lesson.replace("\n", " ")[:600]]}}

    def _verdict(self, s) -> dict:
        sc = s["scratch"]
        self.last_trace = s["ex"].log + [{"reflections": sc["memory"]}]
        v = s["session"].verdict(sc["final"], s["budget"].usage)
        v.trials = sc["trial"]
        return {"verdict": v}

    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict:
        return run_graph(self.graph(), {"alert": alert, "tools": tools, "limits": limits})["verdict"]

    investigate_legacy = _Legacy.investigate
