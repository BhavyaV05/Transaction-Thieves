"""MediQ (Li et al. 2024) as a StateGraph, ported mode (stretch). Parity with ftm/external/mediq.py is tested.

    init -> state (abstention step) -> question -> state ... -> verdict

Owner: T3 (Akshat).
"""

from __future__ import annotations

from ftm.external._infoseek import ask_question, call_json
from ftm.external.mediq import P_FLOOR, STATE_SCHEMA, _question_schema
from ftm.external.mediq import MediQ as _Legacy
from ftm.harness.interface import ToolBox
from ftm.llm.budget import BudgetedClient
from ftm.llm.investigation import Investigation, outcome
from ftm.planners.llm_prompt import eps_levels
from ftm.types import AlertContext, Limits, Verdict

from ._base import END, START, new_graph, run_graph


class MediQ(_Legacy):
    def graph(self):
        if getattr(self, "_graph", None) is None:
            g = new_graph()
            for name, fn in (("init", self._init), ("state", self._state), ("question", self._question), ("verdict", self._verdict)):
                g.add_node(name, fn)
            g.add_edge(START, "init")
            g.add_edge("init", "state")
            g.add_conditional_edges("state", lambda s: "verdict" if s["scratch"]["done"] else ("question" if s["scratch"]["go"] else "state"),
                                    ["question", "state", "verdict"])  # fmt: skip
            g.add_conditional_edges("question", lambda s: "state", ["state"])
            g.add_edge("verdict", END)
            self._graph = g.compile()
        return self._graph

    def _init(self, s) -> dict:
        alert, limits = s["alert"], s["limits"]
        session = Investigation(alert, s["tools"], limits, fusion=self.fusion)
        budget = BudgetedClient(self.llm, limits, tag=f"{self.name}:{alert.alert_id}")
        return {"session": session, "budget": budget, "step": 0,
                "scratch": {"banks": list(alert.summary.get("partner_banks", [])), "p": None, "conf": 0.0, "trace": [], "stop": "questions",
                            "hint": "", "asked_n": 0, "done": False, "go": False, "rec": None}}  # fmt: skip

    def _state(self, s) -> dict:
        sc, session, budget, alert = s["scratch"], s["session"], s["budget"], s["alert"]
        step = s["step"] + 1
        new = {**sc, "done": True, "go": False}
        if budget.calls_left <= 0:
            return {"step": step, "scratch": {**new, "stop": "calls_spent"}}
        # abstention step: how sure is the expert of the probability it has?
        rs = call_json(budget, self._state_messages(alert, session), STATE_SCHEMA, max_tokens=self.max_tokens,
                       temperature=self.temperature, seed=self.seed, extra=self.extra)  # fmt: skip
        if rs.out_of_budget:
            return {"step": step, "scratch": {**new, "stop": "calls_spent"}}
        trace = list(sc["trace"])
        new["trace"] = trace
        if rs.obj is None:
            trace.append({"step": step, "skipped": "malformed state", "error": rs.error})
            if step > self.max_questions + 2:  # every state call failed: give up rather than loop
                return {"step": step, "scratch": {**new, "stop": "malformed"}}
            return {"step": step, "scratch": {**new, "done": False}}
        p = min(max(float(rs.obj["probability"]), P_FLOOR), 1 - P_FLOOR)
        conf = float(rs.obj["confidence"])
        rec = {"step": step, "probability": p, "confidence": conf}
        trace.append(rec)
        new.update(p=p, conf=conf, rec=rec)
        if conf >= self.threshold:
            return {"step": step, "scratch": {**new, "stop": "confident"}}
        if sc["asked_n"] >= self.max_questions:
            return {"step": step, "scratch": {**new, "stop": "questions"}}
        if not sc["banks"] or not eps_levels(session.eps_left):
            return {"step": step, "scratch": {**new, "stop": "no_banks" if not sc["banks"] else "eps_spent"}}
        if session.out_of_time:
            return {"step": step, "scratch": {**new, "stop": "time_spent"}}
        if budget.calls_left <= 1:  # the question needs a call and the next state check another
            return {"step": step, "scratch": {**new, "stop": "calls_spent"}}
        return {"step": step, "scratch": {**new, "done": False, "go": True}}

    def _question(self, s) -> dict:  # not confident: ask a follow-up
        sc, session, rec, step = s["scratch"], s["session"], s["scratch"]["rec"], s["step"]
        banks = sc["banks"]
        rq = call_json(s["budget"], self._question_messages(s["alert"], session, banks, sc["p"], sc["conf"], sc["hint"]),
                       _question_schema(banks, session.eps_left), max_tokens=self.max_tokens,
                       temperature=self.temperature, seed=self.seed, extra=self.extra)  # fmt: skip
        if rq.out_of_budget:
            return {"scratch": {**sc, "stop": "calls_spent", "done": True, "go": False}}
        if rq.obj is None:
            rec["skipped"] = f"malformed question: {rq.error}"
            return {"scratch": {**sc, "asked_n": sc["asked_n"] + 1}}  # a failed turn still counts, so a broken model cannot loop
        q = rq.obj
        asked, why = ask_question(session, self.dsl, banks, q["bank"], q["eps"], q["question"], step)
        new = {**sc, "asked_n": sc["asked_n"] + 1}
        if asked is None:
            rec["refused"] = why
            return {"scratch": {**new, "hint": f"\nYour last question was refused: {why}"}}
        new["hint"] = ""
        rec["asked"] = asked.entry.program
        rec["outcome"] = outcome(asked.entry)
        if asked.cell is None:
            new["hint"] = f"\nThe last question ({asked.entry.program}) got no usable answer: {rec['outcome']}."
        else:
            rec["cell"] = asked.cell
        return {"scratch": new}

    def _verdict(self, s) -> dict:
        sc = s["scratch"]
        self.last_trace, self.last_stop = sc["trace"], sc["stop"]
        return {"verdict": s["session"].verdict(sc["p"], s["budget"].usage)}

    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict:
        return run_graph(self.graph(), {"alert": alert, "tools": tools, "limits": limits})["verdict"]

    investigate_legacy = _Legacy.investigate
