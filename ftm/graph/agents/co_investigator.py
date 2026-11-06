"""Co-Investigator as a StateGraph, ported modes (native and ported). Parity with ftm/external/co_investigator.py is tested.

Owner: T3 (Akshat).

    init -> plan -> typology (one per chosen agent) -> [external] -> narrative -> validate -(problems?)-> revise ... -> verdict
Prompts, schemas, the bundle builder, `_validate` and the emit_unvalidated rule are the old class's (shared, not copied).
"""

from __future__ import annotations

from ftm.external.co_investigator import REVISE_BUDGET
from ftm.external.co_investigator import CoInvestigator as _Legacy
from ftm.external.torres import _CITE
from ftm.harness.interface import ToolBox
from ftm.llm.budget import BudgetedClient
from ftm.llm.investigation import Investigation
from ftm.types import AlertContext, Limits, Verdict

from ._base import END, START, new_graph, run_graph


def _ok(draft) -> bool:
    return draft is not None and "_invalid" not in draft


class CoInvestigator(_Legacy):
    def graph(self):
        if getattr(self, "_graph", None) is None:
            g = new_graph()
            for name, fn in (("init", self._init), ("plan", self._plan_node), ("typology", self._typology_node),
                             ("external", self._external_node), ("narrative", self._narrative_node), ("validate", self._validate_node),
                             ("revise", self._revise), ("verdict", self._verdict)):  # fmt: skip
                g.add_node(name, fn)
            g.add_edge(START, "init")
            g.add_edge("init", "plan")
            g.add_conditional_edges("plan", self._after_plan, ["typology", "external", "narrative"])
            g.add_conditional_edges("typology", self._after_plan, ["typology", "external", "narrative"])
            g.add_edge("external", "narrative")
            g.add_conditional_edges("narrative", lambda s: "validate" if _ok(s["scratch"]["draft"]) else "verdict", ["validate", "verdict"])
            g.add_conditional_edges("validate", self._after_validate, ["revise", "verdict"])
            g.add_conditional_edges("revise", self._after_validate, ["revise", "verdict"])
            g.add_edge("verdict", END)
            self._graph = g.compile()
        return self._graph

    def _after_plan(self, s) -> str:
        sc = s["scratch"]
        if sc["next_agent"] < len(sc["chosen"]):
            return "typology"
        return "external" if self.mode == "ported" else "narrative"

    @staticmethod
    def _after_validate(s) -> str:
        sc = s["scratch"]
        return "revise" if sc["problems"] and not sc["revise_stop"] and sc["revisions"] < REVISE_BUDGET else "verdict"

    def _init(self, s) -> dict:
        alert, limits = s["alert"], s["limits"]
        session = Investigation(alert, s["tools"], limits, fusion=self.fusion)
        budget = BudgetedClient(self.llm, limits, tag=f"{self.name}:{alert.alert_id}")
        items = self._bundle_builder.bundle(alert, s["tools"])
        return {"session": session, "budget": budget,
                "scratch": {"items": items, "chosen": [], "next_agent": 0, "findings": [], "draft": None, "problems": ["no draft"],
                            "revisions": 0, "revise_stop": False}}  # fmt: skip

    def _plan_node(self, s) -> dict:
        sc = s["scratch"]
        return {"scratch": {**sc, "chosen": self._plan(s["budget"], s["alert"], sc["items"])}}

    def _typology_node(self, s) -> dict:
        sc = s["scratch"]
        agent = sc["chosen"][sc["next_agent"]]
        f = self._typology(s["budget"], agent, s["alert"], sc["items"])
        return {"scratch": {**sc, "next_agent": sc["next_agent"] + 1, "findings": [*sc["findings"], f] if f else sc["findings"]}}

    def _external_node(self, s) -> dict:
        sc = s["scratch"]
        self._external_intelligence(s["budget"], s["session"], s["alert"], sc["items"])  # appends partner_answer items in place
        return {}

    def _narrative_node(self, s) -> dict:
        sc = s["scratch"]
        return {"scratch": {**sc, "draft": self._narrative(s["budget"], s["alert"], sc["items"], sc["findings"])}}

    def _validate_node(self, s) -> dict:
        sc = s["scratch"]
        return {"scratch": {**sc, "problems": self._validate(s["budget"], s["alert"], sc["items"], sc["draft"])}}

    def _revise(self, s) -> dict:
        sc = s["scratch"]
        revised = self._narrative(s["budget"], s["alert"], sc["items"], sc["findings"], sc["problems"], sc["draft"])
        if not _ok(revised):
            return {"scratch": {**sc, "revisions": sc["revisions"] + 1, "revise_stop": True}}
        return {"scratch": {**sc, "revisions": sc["revisions"] + 1, "draft": revised,
                            "problems": self._validate(s["budget"], s["alert"], sc["items"], revised)}}  # fmt: skip

    def _verdict(self, s) -> dict:
        sc = s["scratch"]
        draft, problems, items = sc["draft"], sc["problems"], sc["items"]
        has_draft = _ok(draft)
        usable = has_draft and not problems
        p_draft = float(draft["confidence"]) if has_draft else None
        probability = p_draft if usable or (self.emit_unvalidated and has_draft) else None
        self.last_views = {"validated": int(usable), "p_draft": p_draft}
        self.last_trace = {"plan": sc["chosen"], "findings": sc["findings"], "draft": draft,
                           "problems": problems, "cited": sorted(set(_CITE.findall((draft or {}).get("narrative", "")))),
                           "evidence": [i.__dict__ for i in items]}  # fmt: skip
        return {"verdict": s["session"].verdict(probability, s["budget"].usage)}

    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict:
        return run_graph(self.graph(), {"alert": alert, "tools": tools, "limits": limits})["verdict"]

    investigate_legacy = _Legacy.investigate
