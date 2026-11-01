"""Multi-agent debate (Du et al. 2024) as a StateGraph, ported mode. Parity with ftm/external/debate_autogen.py is tested.

    init -> round -> turn* (one analyst per visit) -> questions -> round ... -> verdict
"""

from __future__ import annotations

from ftm.external.debate_autogen import Debate as _Legacy
from ftm.external.debate_autogen import aggregate, turn_schema
from ftm.harness.interface import ToolBox
from ftm.llm.budget import BudgetedClient, BudgetExceeded, PromptTooLong
from ftm.llm.investigation import Investigation
from ftm.llm.jsonout import JsonOutputError, parse_and_validate
from ftm.planners.llm_prompt import eps_levels
from ftm.types import AlertContext, Limits, Verdict

from ._base import END, START, new_graph, run_graph


class Debate(_Legacy):
    def graph(self):
        if getattr(self, "_graph", None) is None:
            g = new_graph()
            for name, fn in (("init", self._init), ("round", self._round), ("turn", self._turn), ("questions", self._questions),
                             ("verdict", self._verdict)):  # fmt: skip
                g.add_node(name, fn)
            g.add_edge(START, "init")
            g.add_edge("init", "round")
            g.add_edge("round", "turn")
            g.add_conditional_edges("turn", lambda s: "turn" if s["scratch"]["k"] < self.n_agents and not s["scratch"]["break"] else "questions",
                                    ["turn", "questions"])  # fmt: skip
            g.add_conditional_edges("questions", lambda s: "verdict" if s["stop"] else "round", ["round", "verdict"])
            g.add_edge("verdict", END)
            self._graph = g.compile()
        return self._graph

    def _init(self, s) -> dict:
        alert, limits = s["alert"], s["limits"]
        session = Investigation(alert, s["tools"], limits, fusion=self.fusion)
        budget = BudgetedClient(self.llm, limits, tag=f"{self.name}:{alert.alert_id}")
        sc = {"banks": list(alert.summary.get("partner_banks", [])), "last": [None] * self.n_agents, "trace": [], "notices": [],
              "r": -1, "k": 0, "break": False, "stop_budget": False, "prev": None, "proposals": []}  # fmt: skip
        return {"session": session, "budget": budget, "step": 0, "stop": False, "scratch": sc}

    def _round(self, s) -> dict:
        sc = s["scratch"]
        # synchronous: every agent sees the same snapshot of the previous round
        return {"scratch": {**sc, "r": sc["r"] + 1, "k": 0, "break": False, "stop_budget": False, "prev": list(sc["last"]), "proposals": []}}

    def _turn(self, s) -> dict:
        sc, session, budget, alert = s["scratch"], s["session"], s["budget"], s["alert"]
        r, k, banks, trace = sc["r"], sc["k"], sc["banks"], list(sc["trace"])
        can_ask = r < self.rounds  # a question in the last round would have no reader
        schema = turn_schema(banks, session.eps_left, can_ask)
        messages = [{"role": "system", "content": self.system_prompt(k, banks)},
                    {"role": "user", "content": self.user_prompt(alert, r, k, session, sc["notices"], sc["prev"], can_ask)}]  # fmt: skip
        new = {**sc, "k": k + 1, "trace": trace}
        try:
            resp = budget.chat(messages, max_tokens=self.max_tokens, temperature=self.temperature,
                               json_schema=schema, seed=self.seed + k, extra=self.extra)  # fmt: skip
        except (BudgetExceeded, PromptTooLong) as e:
            trace.append({"round": r, "agent": k, "error": type(e).__name__})
            stop = isinstance(e, BudgetExceeded)
            return {"scratch": {**new, "stop_budget": stop, "break": stop}}
        try:
            obj = parse_and_validate(resp.text, schema)
        except JsonOutputError as e:
            trace.append({"round": r, "agent": k, "error": str(e), "text": resp.text[:300]})
            return {"scratch": new}
        last = list(sc["last"])
        last[k] = {"probability": float(obj["probability"]), "rationale": obj["rationale"]}
        trace.append({"round": r, "agent": k, **last[k], "question": obj["question"]})
        proposals = [*sc["proposals"], (k, obj["question"])] if obj["question"] is not None else sc["proposals"]
        return {"scratch": {**new, "last": last, "proposals": proposals}}

    def _questions(self, s) -> dict:
        sc, session, budget = s["scratch"], s["session"], s["budget"]
        if sc["stop_budget"] or budget.calls_left <= 0:  # nobody would read the answers
            return {"stop": True}
        trace, step = list(sc["trace"]), s["step"]
        for k, q in sc["proposals"]:  # agent order, until the budget runs out
            if not eps_levels(session.eps_left) or session.out_of_time:
                break
            step += 1
            trace.append({"round": sc["r"], "type": "question", **self._ask(q, session, k, step, sc["notices"])})
        return {"step": step, "stop": sc["r"] >= self.rounds, "scratch": {**sc, "trace": trace}}

    def _verdict(self, s) -> dict:
        sc = s["scratch"]
        self.last_trace = sc["trace"]
        return {"verdict": s["session"].verdict(aggregate([a["probability"] if a else None for a in sc["last"]]), s["budget"].usage)}

    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict:
        return run_graph(self.graph(), {"alert": alert, "tools": tools, "limits": limits})["verdict"]

    investigate_legacy = _Legacy.investigate
