"""LLMCompiler (Kim et al. 2024) as a StateGraph, ported mode. Parity with ftm/external/llmcompiler.py is tested.

    plan -> wave* (independent asks sent together) -> join -> (replan -> plan | verdict)

Owner: T3 (Akshat).
"""

from __future__ import annotations

import re

from ftm.external.llmcompiler import JOINER, PLANNER, _replans, parse_tasks, substitute, waves
from ftm.external.llmcompiler import LlmCompiler as _Legacy
from ftm.external.rewoo import solver_probability
from ftm.external.text_agent import ToolCall, ToolExecutor, case_text, tools_doc
from ftm.harness.interface import ToolBox
from ftm.llm.budget import BudgetedClient, BudgetExceeded, PromptTooLong
from ftm.llm.investigation import Investigation
from ftm.types import AlertContext, Limits, Verdict

from ._base import END, START, new_graph, run_graph


class LlmCompiler(_Legacy):
    def graph(self):
        if getattr(self, "_graph", None) is None:
            g = new_graph()
            for name, fn in (("init", self._init), ("plan", self._plan), ("wave", self._wave), ("join", self._join),
                             ("verdict", self._verdict)):  # fmt: skip
                g.add_node(name, fn)
            g.add_edge(START, "init")
            g.add_edge("init", "plan")
            g.add_conditional_edges("plan", lambda s: "verdict" if s.get("halt") else self._next(s), ["wave", "join", "verdict"])
            g.add_conditional_edges("wave", self._next, ["wave", "join"])
            g.add_conditional_edges("join", lambda s: "plan" if s["scratch"]["replan"] and not s.get("halt") else "verdict", ["plan", "verdict"])
            g.add_edge("verdict", END)
            self._graph = g.compile()
        return self._graph

    @staticmethod
    def _next(s) -> str:
        sc = s["scratch"]
        return "wave" if sc["w"] < len(sc["waves"]) else "join"

    def _init(self, s) -> dict:
        alert, limits = s["alert"], s["limits"]
        session = Investigation(alert, s["tools"], limits, fusion=self.fusion)
        budget = BudgetedClient(self.llm, limits, tag=f"llmcompiler:{alert.alert_id}")
        banks = list(alert.summary.get("partner_banks", []))
        ex = ToolExecutor(alert, s["tools"], session, self.dsl, banks)
        case = case_text(alert, str(alert.summary.get("evidence", "")))
        return {"session": session, "budget": budget, "ex": ex, "step": 0, "halt": False,
                "scratch": {"case": case, "banks": banks, "history": [], "p": None, "trace": [], "rnd": -1, "replan": False,
                            "tasks": {}, "waves": [], "w": 0, "results": {}}}  # fmt: skip

    def _plan(self, s) -> dict:
        sc, session = s["scratch"], s["session"]
        rnd = sc["rnd"] + 1
        history = sc["history"]
        prior = ("\n\nResults so far:\n" + "\n".join(history)) if history else ""
        try:
            plan = self._chat(s["budget"], f"{PLANNER}\n\n{tools_doc(self.dsl, sc['banks'], session.eps_left)}", sc["case"] + prior,
                              self.plan_tokens)  # fmt: skip
        except (BudgetExceeded, PromptTooLong):
            return {"halt": True}
        tasks = parse_tasks(plan)
        return {"scratch": {**sc, "rnd": rnd, "plan": plan, "tasks": tasks, "waves": waves(tasks), "w": 0, "results": {}}}

    def _wave(self, s) -> dict:
        sc, ex, step = s["scratch"], s["ex"], s["step"]
        wave, tasks, results = sc["waves"][sc["w"]], sc["tasks"], dict(sc["results"])
        # tasks of one wave never reference each other ($k only points to earlier waves), so every argument is
        # substituted before the wave starts
        calls = []
        for i in wave:
            step += 1
            assert not any(int(k) in wave for k in re.findall(r"\$(\d+)", tasks[i].arg))
            calls.append((ToolCall(tasks[i].tool, substitute(tasks[i].arg, results)), step))
        history = list(sc["history"])
        for i, (call, _), obs in zip(wave, calls, ex.run_wave(calls), strict=True):
            results[i] = obs
            history.append(f"{call.tool}[{call.arg}] -> {obs}")
        return {"step": step, "scratch": {**sc, "w": sc["w"] + 1, "results": results, "history": history}}

    def _join(self, s) -> dict:
        sc = s["scratch"]
        trace = [*sc["trace"], {"round": sc["rnd"], "plan": sc["plan"], "waves": waves(sc["tasks"])}]
        try:
            join = self._chat(s["budget"], JOINER, sc["case"] + "\n\nResults:\n" + ("\n".join(sc["history"]) or "(none)"), self.join_tokens)
        except (BudgetExceeded, PromptTooLong):
            return {"halt": True, "scratch": {**sc, "trace": trace}}
        trace[-1]["join"] = join
        if not _replans(join) or sc["rnd"] == self.max_replans:
            return {"scratch": {**sc, "trace": trace, "p": solver_probability(join), "replan": False}}
        return {"scratch": {**sc, "trace": trace, "replan": True}}

    def _verdict(self, s) -> dict:
        sc = s["scratch"]
        self.last_trace = sc["trace"] + s["ex"].log
        return {"verdict": s["session"].verdict(sc["p"], s["budget"].usage)}

    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict:
        return run_graph(self.graph(), {"alert": alert, "tools": tools, "limits": limits})["verdict"]

    investigate_legacy = _Legacy.investigate
