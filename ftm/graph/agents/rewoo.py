"""ReWOO (Xu et al. 2023) as a StateGraph, ported mode. Parity with ftm/external/rewoo.py is tested.

    plan (1 LLM call) -> work* (one tool per visit, no LLM) -> solve (1 LLM call) -> verdict

Owner: T3 (Akshat).
"""

from __future__ import annotations

import re

from ftm.external.rewoo import PLANNER, SOLVER, parse_plan, solver_probability
from ftm.external.rewoo import Rewoo as _Legacy
from ftm.external.text_agent import ToolCall, ToolExecutor, case_text, tools_doc
from ftm.harness.interface import ToolBox
from ftm.llm.budget import BudgetedClient
from ftm.llm.investigation import Investigation
from ftm.types import AlertContext, Limits, Verdict

from ._base import END, START, new_graph, run_graph


class Rewoo(_Legacy):
    def graph(self):
        if getattr(self, "_graph", None) is None:
            g = new_graph()
            for name, fn in (("plan", self._plan), ("work", self._work), ("solve", self._solve), ("verdict", self._verdict)):
                g.add_node(name, fn)
            g.add_edge(START, "plan")
            g.add_conditional_edges("plan", lambda s: "verdict" if s["stop"] == "no_plan" else self._next(s), ["work", "solve", "verdict"])
            g.add_conditional_edges("work", self._next, ["work", "solve"])
            g.add_edge("solve", "verdict")
            g.add_edge("verdict", END)
            self._graph = g.compile()
        return self._graph

    @staticmethod
    def _next(s) -> str:
        sc = s["scratch"]
        return "work" if sc["i"] < len(sc["steps"]) and not s["session"].out_of_time else "solve"

    def _plan(self, s) -> dict:
        alert, limits = s["alert"], s["limits"]
        session = Investigation(alert, s["tools"], limits, fusion=self.fusion)
        budget = BudgetedClient(self.llm, limits, tag=f"rewoo:{alert.alert_id}")
        banks = list(alert.summary.get("partner_banks", []))
        ex = ToolExecutor(alert, s["tools"], session, self.dsl, banks)
        task = case_text(alert, str(alert.summary.get("evidence", "")))
        plan_text = self._chat(budget, f"{PLANNER}\n\n{tools_doc(self.dsl, banks, session.eps_left)}", task, self.plan_tokens)
        out = {"session": session, "budget": budget, "ex": ex, "stop": "no_plan" if plan_text is None else None}
        steps = [] if plan_text is None else parse_plan(plan_text)[: self.max_steps]
        return {**out, "scratch": {"task": task, "plan_text": plan_text, "steps": steps, "i": 0, "evidence": {}, "worked": []}}

    def _work(self, s) -> dict:  # Worker: no LLM calls
        sc, ex = s["scratch"], s["ex"]
        reason, n, call = sc["steps"][sc["i"]]
        arg = re.sub(r"#E(\d+)", lambda m: sc["evidence"].get(int(m.group(1)), m.group(0)), call.arg)
        obs = ex.run(ToolCall(call.tool, arg), sc["i"] + 1)
        sc["evidence"][n] = obs
        sc["worked"].append(f"Plan: {reason}\n#E{n} = {call.tool}[{arg}]\nEvidence: {obs}")
        return {"scratch": {**sc, "i": sc["i"] + 1}}

    def _solve(self, s) -> dict:
        sc, ex = s["scratch"], s["ex"]
        worked = sc["worked"]
        solve_in = f"{sc['task']}\n\n" + ("\n\n".join(worked) if worked else "(the plan had no valid steps)") + "\n\nNow begin to resolve the alert."
        answer = self._chat(s["budget"], SOLVER, solve_in, self.solve_tokens)
        self.last_trace = [{"plan": sc["plan_text"], "n_steps": len(sc["steps"])}, *ex.log, {"solver": answer}]
        return {"scratch": {**sc, "p": solver_probability(answer)}}

    def _verdict(self, s) -> dict:
        p = None if s["stop"] == "no_plan" else s["scratch"]["p"]
        return {"verdict": s["session"].verdict(p, s["budget"].usage)}

    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict:
        return run_graph(self.graph(), {"alert": alert, "tools": tools, "limits": limits})["verdict"]

    investigate_legacy = _Legacy.investigate
