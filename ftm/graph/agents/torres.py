"""Torres et al. as a StateGraph, ported modes (native and ported). Parity with ftm/external/torres.py is tested.

Owner: T3 (Akshat).

    init (evidence bundle) -> [questions] -> triage -(problems?)-> repair ... -> settle -> counterfactual ... -> verdict
Bundle, verifier, calls, prompts and the disposition -> probability map are the old class's (shared, not copied).
"""

from __future__ import annotations

import json

from ftm.external.torres import (
    _CITE,
    COUNTERFACTUAL_BUDGET,
    REPAIR_BUDGET,
    SYSTEM,
    _output_schema,
    disposition_probability,
    verify,
)
from ftm.external.torres import Torres as _Legacy
from ftm.harness.interface import ToolBox
from ftm.llm.budget import BudgetedClient
from ftm.llm.investigation import Investigation
from ftm.types import AlertContext, Limits, Verdict

from ._base import END, START, new_graph, run_graph


def _problems_of(out, items) -> list[str]:
    return ["invalid JSON"] if "_invalid" in out else verify(out, items)


class Torres(_Legacy):
    def graph(self):
        if getattr(self, "_graph", None) is None:
            g = new_graph()
            for name, fn in (("init", self._init), ("questions", self._questions_node), ("triage", self._triage_node),
                             ("repair", self._repair), ("settle", self._settle), ("counterfactual", self._counterfactual),
                             ("verdict", self._verdict)):  # fmt: skip
                g.add_node(name, fn)
            g.add_edge(START, "init")
            g.add_conditional_edges("init", lambda s: "questions" if self.mode == "ported" else "triage", ["questions", "triage"])
            g.add_edge("questions", "triage")
            g.add_conditional_edges("triage", self._repair_more, ["repair", "settle"])
            g.add_conditional_edges("repair", self._repair_more, ["repair", "settle"])
            g.add_conditional_edges("settle", lambda s: "counterfactual" if s["scratch"]["out"] is not None else "verdict",
                                    ["counterfactual", "verdict"])  # fmt: skip
            g.add_conditional_edges("counterfactual", lambda s: "verdict" if s["scratch"]["cf_done"] else "counterfactual",
                                    ["counterfactual", "verdict"])  # fmt: skip
            g.add_edge("verdict", END)
            self._graph = g.compile()
        return self._graph

    @staticmethod
    def _repair_more(s) -> str:
        sc = s["scratch"]
        return "repair" if sc["problems"] and not sc["repair_stop"] and sc["repairs"] < REPAIR_BUDGET and sc["raw"] is not None else "settle"

    def _init(self, s) -> dict:
        alert, limits = s["alert"], s["limits"]
        session = Investigation(alert, s["tools"], limits, fusion=self.fusion)
        budget = BudgetedClient(self.llm, limits, tag=f"{self.name}:{alert.alert_id}")
        items = self.bundle(alert, s["tools"])
        return {"session": session, "budget": budget, "scratch": {"items": items}}

    def _questions_node(self, s) -> dict:
        sc = s["scratch"]
        self._questions(s["budget"], s["session"], s["alert"], sc["items"])  # appends partner_answer items in place
        return {}

    def _triage_node(self, s) -> dict:
        """The first triage call (the head of the old _triage)."""
        sc, alert, budget, items = s["scratch"], s["alert"], s["budget"], s["scratch"]["items"]
        schema = _output_schema([i.eid for i in items])
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Alert {alert.alert_id}\nEvidence:\n{self.render(items)}"}]  # fmt: skip
        out = self._call(budget, msgs, schema)
        if out is None:
            return {"scratch": {**sc, "raw": None, "problems": ["no call left"], "repairs": 0, "repair_stop": True}}
        return {"scratch": {**sc, "raw": out, "msgs": msgs, "schema": schema, "problems": _problems_of(out, items), "repairs": 0,
                            "repair_stop": False}}  # fmt: skip

    def _repair(self, s) -> dict:
        """One repair call with the verifier's feedback; a missing reply ends the repairs, keeping the earlier output."""
        sc, budget, items = s["scratch"], s["budget"], s["scratch"]["items"]
        out, msgs = sc["raw"], list(sc["msgs"])
        msgs += [{"role": "assistant", "content": json.dumps(out)},
                 {"role": "user", "content": "Your output was rejected by the verifier: " + "; ".join(sc["problems"])
                  + ". Return a corrected JSON object."}]  # fmt: skip
        fixed = self._call(budget, msgs, sc["schema"])
        if fixed is None:
            return {"scratch": {**sc, "msgs": msgs, "repairs": sc["repairs"] + 1, "repair_stop": True}}
        return {"scratch": {**sc, "msgs": msgs, "raw": fixed, "problems": _problems_of(fixed, items), "repairs": sc["repairs"] + 1}}

    def _settle(self, s) -> dict:
        sc = s["scratch"]
        raw = sc["raw"]
        out = None if raw is None or "_invalid" in raw else raw
        probability = None if out is None else disposition_probability(out["decision"], out["confidence"])
        trace = {"evidence": [i.__dict__ for i in sc["items"]], "output": out, "verifier_problems": sc["problems"]}
        support = [e for e in out.get("supporting_evidence", []) if any(i.eid == e for i in sc["items"])] if out is not None else []
        return {"scratch": {**sc, "out": out, "probability": probability, "trace": trace, "support": support, "edit": 0,
                            "cf": [], "cf_done": False}}  # fmt: skip

    def _counterfactual(self, s) -> dict:
        """One evidence counterfactual (3.4): drop the cited supporting items (edit 0) or the first one, re-run, compare."""
        sc, alert, budget, items, out, support = s["scratch"], s["alert"], s["budget"], s["scratch"]["items"], s["scratch"]["out"], s["scratch"]["support"]
        edit = sc["edit"]
        if edit >= COUNTERFACTUAL_BUDGET or not support:
            return {"scratch": {**sc, "cf_done": True}}
        drop = set(support) if edit == 0 else {support[0]}
        kept = [i for i in items if i.eid not in drop]
        schema = _output_schema([i.eid for i in kept])
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Alert {alert.alert_id}\nEvidence:\n{self.render(kept)}"}]  # fmt: skip
        cf = self._call(budget, msgs, schema)
        if cf is None or "_invalid" in cf:
            return {"scratch": {**sc, "cf_done": True}}
        still_cites = sorted(set(_CITE.findall(cf.get("rationale", ""))) & drop)
        res = {"dropped": sorted(drop), "decision": cf["decision"], "changed": cf["decision"] != out["decision"],
               "decorative": cf["decision"] == out["decision"] and bool(still_cites), "cites_dropped": still_cites}  # fmt: skip
        return {"scratch": {**sc, "cf": [*sc["cf"], res], "edit": edit + 1, "cf_done": edit + 1 >= COUNTERFACTUAL_BUDGET}}

    def _verdict(self, s) -> dict:
        sc = s["scratch"]
        trace = dict(sc["trace"])
        if sc["out"] is not None:
            trace["counterfactuals"] = sc["cf"]
        self.last_trace = trace
        return {"verdict": s["session"].verdict(sc["probability"], s["budget"].usage)}

    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict:
        return run_graph(self.graph(), {"alert": alert, "tools": tools, "limits": limits})["verdict"]

    investigate_legacy = _Legacy.investigate
