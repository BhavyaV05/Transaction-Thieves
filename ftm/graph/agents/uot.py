"""Uncertainty of Thoughts (Hu et al. 2024) as a StateGraph, ported mode. Parity with ftm/external/uot.py is tested.

    init -> gate -> propose -> act -> gate ... -> verdict
`gate` holds the loop head (step cap, no banks / eps / time / calls), `propose` the candidate call and EIG scoring
(stop rules: no candidates, low EIG), `act` the question and the belief update.
"""

from __future__ import annotations

from ftm.external._infoseek import ask_question, binary_belief_update, call_json
from ftm.external.uot import MAX_CANDIDATES, Uot as _Legacy, _outcome, _schema
from ftm.harness.interface import ToolBox
from ftm.llm.budget import BudgetedClient
from ftm.llm.investigation import Investigation
from ftm.planners.llm_prompt import eps_levels
from ftm.types import AlertContext, Limits, Verdict

from ._base import END, START, new_graph, run_graph


class Uot(_Legacy):
    def graph(self):
        if getattr(self, "_graph", None) is None:
            g = new_graph()
            for name, fn in (("init", self._init), ("gate", self._gate), ("propose", self._propose), ("act", self._act),
                             ("verdict", self._verdict)):  # fmt: skip
                g.add_node(name, fn)
            g.add_edge(START, "init")
            g.add_edge("init", "gate")
            g.add_conditional_edges("gate", lambda s: "verdict" if s["scratch"]["done"] else "propose", ["propose", "verdict"])
            g.add_conditional_edges("propose", lambda s: "act" if s["scratch"]["best"] else ("verdict" if s["scratch"]["done"] else "gate"),
                                    ["act", "gate", "verdict"])  # fmt: skip
            g.add_edge("act", "gate")
            g.add_edge("verdict", END)
            self._graph = g.compile()
        return self._graph

    def _init(self, s) -> dict:
        alert, limits = s["alert"], s["limits"]
        session = Investigation(alert, s["tools"], limits, fusion=self.fusion)
        budget = BudgetedClient(self.llm, limits, tag=f"{self.name}:{alert.alert_id}")
        banks = list(alert.summary.get("partner_banks", []))
        belief = min(max(float(alert.p0), 1e-6), 1 - 1e-6)
        return {"session": session, "budget": budget, "step": 0,
                "scratch": {"banks": banks, "belief": belief, "trace": [], "stop": "steps", "hint": "", "done": False,
                            "best": None, "rec": None}}  # fmt: skip

    def _gate(self, s) -> dict:
        sc, session = s["scratch"], s["session"]
        step = s["step"] + 1
        new = {**sc, "best": None, "done": True}
        if step > self.max_steps:
            return {"step": step, "scratch": new}
        if not sc["banks"] or not eps_levels(session.eps_left):
            new["stop"] = "no_banks" if not sc["banks"] else "eps_spent"
        elif session.out_of_time:
            new["stop"] = "time_spent"
        elif s["budget"].calls_left <= 0:
            new["stop"] = "calls_spent"
        else:
            new["done"] = False
        return {"step": step, "scratch": new}

    def _propose(self, s) -> dict:
        sc, session, step = s["scratch"], s["session"], s["step"]
        banks, belief, trace = sc["banks"], sc["belief"], list(sc["trace"])
        res = call_json(s["budget"], self._messages(s["alert"], session, belief, banks, sc["hint"]), _schema(banks, session.eps_left),
                        max_tokens=self.max_tokens, temperature=self.temperature, seed=self.seed, extra=self.extra)  # fmt: skip
        new = {**sc, "trace": trace, "best": None}
        if res.out_of_budget:
            return {"scratch": {**new, "stop": "calls_spent", "done": True}}
        if res.obj is None:
            trace.append({"step": step, "skipped": "malformed", "error": res.error})
            return {"scratch": new}
        offered = res.obj["candidates"][:MAX_CANDIDATES]
        if not offered:
            trace.append({"step": step, "belief": belief, "candidates": []})
            return {"scratch": {**new, "stop": "no_candidates", "done": True}}
        scored, rejected = [], []
        for cand in offered:
            c, why = self._score(belief, cand, session, banks)
            (scored if c else rejected).append(c or why)
        rec = {"step": step, "belief_before": belief, "candidates": [
            {k: c[k] for k in ("bank", "eps", "canonical", "eig")} for c in scored], "rejected": rejected}  # fmt: skip
        trace.append(rec)
        if not scored:  # nothing usable: tell the model why and let it try again (bounded by the step cap)
            return {"scratch": {**new, "hint": "\nYour last candidates were not usable: " + "; ".join(rejected)[:600]}}
        best = max(scored, key=lambda c: c["eig"])
        if best["eig"] < self.eig_stop:
            rec["stop"] = f"best EIG {best['eig']:.4f} < {self.eig_stop}"
            return {"scratch": {**new, "hint": "", "stop": "low_eig", "done": True}}
        return {"scratch": {**new, "hint": "", "best": best, "rec": rec}}

    def _act(self, s) -> dict:
        sc, session = s["scratch"], s["session"]
        best, rec = sc["best"], sc["rec"]
        asked, why = ask_question(session, self.dsl, sc["banks"], best["bank"], best["eps"], best["question"], s["step"])
        if asked is None:
            rec["refused"] = why
            return {"scratch": {**sc, "best": None, "hint": f"\nYour last question was refused: {why}"}}
        rec["asked"] = best["canonical"]
        rec["outcome"] = _outcome(asked)
        if asked.cell is None:
            return {"scratch": {**sc, "best": None, "hint": f"\nThe last question ({best['canonical']}) got no usable answer: {rec['outcome']}. Ask something else."}}
        i = best["cells"].index(asked.cell)
        belief = binary_belief_update(sc["belief"], best["p_launder"][i], best["p_clean"][i])
        rec.update(cell=asked.cell, belief_after=belief)
        return {"scratch": {**sc, "best": None, "belief": belief}}

    def _verdict(self, s) -> dict:
        sc = s["scratch"]
        self.last_trace, self.last_belief, self.last_stop = sc["trace"], sc["belief"], sc["stop"]
        return {"verdict": s["session"].verdict(sc["belief"], s["budget"].usage)}

    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict:
        return run_graph(self.graph(), {"alert": alert, "tools": tools, "limits": limits})["verdict"]

    investigate_legacy = _Legacy.investigate
