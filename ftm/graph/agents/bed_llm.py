"""BED-LLM (Choudhury et al. 2026) as a StateGraph, ported mode. Parity with ftm/external/bed_llm.py is tested.

    init -> gate -> hypotheses -> candidates -> choose -> act -> gate ... -> verdict
`hypotheses` and `candidates` are the two LLM calls of a step, `choose` scores the candidates by label EIG and applies
the stop rule, `act` asks the best one and reweights the particle set.

Owner: T3 (Akshat).
"""

from __future__ import annotations

from ftm.external._infoseek import ask_question, bayes_update
from ftm.external.bed_llm import MAX_CANDIDATES, reweight, smoothed_share
from ftm.external.bed_llm import BedLlm as _Legacy
from ftm.harness.interface import ToolBox
from ftm.llm.budget import BudgetedClient
from ftm.llm.investigation import Investigation, outcome
from ftm.planners.llm_prompt import eps_levels
from ftm.types import AlertContext, Limits, Verdict

from ._base import END, START, new_graph, run_graph


class BedLlm(_Legacy):
    def graph(self):
        if getattr(self, "_graph", None) is None:
            g = new_graph()
            for name, fn in (("init", self._init), ("gate", self._gate), ("hypotheses", self._hyp), ("candidates", self._cands),
                             ("choose", self._choose), ("act", self._act), ("verdict", self._verdict)):  # fmt: skip
                g.add_node(name, fn)
            g.add_edge(START, "init")
            g.add_edge("init", "gate")
            nxt = lambda here, then: (lambda s: "verdict" if s["scratch"]["done"] else (then if s["scratch"]["go"] else "gate"))  # noqa: E731
            g.add_conditional_edges("gate", lambda s: "verdict" if s["scratch"]["done"] else "hypotheses", ["hypotheses", "verdict"])
            g.add_conditional_edges("hypotheses", nxt("hypotheses", "candidates"), ["candidates", "gate", "verdict"])
            g.add_conditional_edges("candidates", nxt("candidates", "choose"), ["choose", "gate", "verdict"])
            g.add_conditional_edges("choose", nxt("choose", "act"), ["act", "gate", "verdict"])
            g.add_edge("act", "gate")
            g.add_edge("verdict", END)
            self._graph = g.compile()
        return self._graph

    def _init(self, s) -> dict:
        alert, limits = s["alert"], s["limits"]
        session = Investigation(alert, s["tools"], limits, fusion=self.fusion)
        budget = BudgetedClient(self.llm, limits, tag=f"{self.name}:{alert.alert_id}")
        return {"session": session, "budget": budget, "step": 0,
                "scratch": {"banks": list(alert.summary.get("partner_banks", [])), "hyps": [], "weights": [], "answered": [], "trace": [],
                            "stop": "steps", "hint": "", "done": False, "go": False, "new": None, "rb": None, "rec": None, "best": None}}  # fmt: skip

    def _gate(self, s) -> dict:
        sc, session = s["scratch"], s["session"]
        step = s["step"] + 1
        new = {**sc, "done": True, "go": False}
        if step > self.max_steps:
            return {"step": step, "scratch": new}
        if not sc["banks"] or not eps_levels(session.eps_left):
            new["stop"] = "no_banks" if not sc["banks"] else "eps_spent"
        elif session.out_of_time:
            new["stop"] = "time_spent"
        elif s["budget"].calls_left < 2:  # a step needs both calls
            new["stop"] = "calls_spent"
        else:
            new["done"] = False
        return {"step": step, "scratch": new}

    def _hyp(self, s) -> dict:
        sc, step = s["scratch"], s["step"]
        ra = self._hypotheses(s["budget"], s["alert"], s["session"])
        if ra.out_of_budget:
            return {"scratch": {**sc, "stop": "calls_spent", "done": True}}
        if ra.obj is None:
            return {"scratch": {**sc, "go": False, "trace": [*sc["trace"], {"step": step, "skipped": "malformed hypotheses", "error": ra.error}]}}
        return {"scratch": {**sc, "go": True, "new": ra.obj["hypotheses"][: self.k]}}

    def _cands(self, s) -> dict:
        sc, step = s["scratch"], s["step"]
        rb = self._candidates(s["budget"], s["alert"], s["session"], sc["banks"], sc["new"], sc["answered"], sc["hint"])
        if rb.out_of_budget:
            return {"scratch": {**sc, "stop": "calls_spent", "done": True}}
        if rb.obj is None:
            return {"scratch": {**sc, "go": False, "trace": [*sc["trace"], {"step": step, "skipped": "malformed candidates", "error": rb.error}]}}
        # the particle set of this step: sampled given the history, reweighted by the observed answers
        hyps = sc["new"]
        weights = reweight(len(hyps), rb.obj["past"], [a["idx"] for a in sc["answered"]])
        rec = {"step": step, "hypotheses": [{"label": h["label"], "weight": round(w, 4)} for h, w in zip(hyps, weights, strict=True)]}  # fmt: skip
        return {"scratch": {**sc, "go": True, "hyps": hyps, "weights": weights, "hint": "", "rb": rb.obj, "rec": rec, "trace": [*sc["trace"], rec]}}

    def _choose(self, s) -> dict:
        sc, session = s["scratch"], s["session"]
        rec, hyps, weights = sc["rec"], sc["hyps"], sc["weights"]
        offered = sc["rb"]["candidates"][:MAX_CANDIDATES]
        if not offered:
            return {"scratch": {**sc, "stop": "no_candidates", "done": True}}
        scored, rejected = [], []
        for cand in offered:
            c, why = self._score(cand, session, hyps, weights)
            (scored if c else rejected).append(c or why)
        rec["candidates"] = [{k: c[k] for k in ("bank", "eps", "canonical", "eig")} for c in scored]
        rec["rejected"] = rejected
        if not scored:
            return {"scratch": {**sc, "go": False, "hint": "\nYour last candidates were not usable: " + "; ".join(rejected)[:600]}}
        best = max(scored, key=lambda c: c["eig"])
        if best["eig"] < self.eig_stop:
            rec["stop"] = f"best EIG {best['eig']:.4f} < {self.eig_stop}"
            return {"scratch": {**sc, "stop": "low_eig", "done": True}}
        return {"scratch": {**sc, "best": best}}

    def _act(self, s) -> dict:
        sc, session = s["scratch"], s["session"]
        best, rec = sc["best"], sc["rec"]
        asked, why = ask_question(session, self.dsl, sc["banks"], best["bank"], best["eps"], best["question"], s["step"])
        if asked is None:
            rec["refused"] = why
            return {"scratch": {**sc, "hint": f"\nYour last question was refused: {why}"}}
        rec["asked"] = best["canonical"]
        rec["outcome"] = outcome(asked.entry)
        if asked.cell is None:
            return {"scratch": {**sc, "hint": f"\nThe last question ({best['canonical']}) got no usable answer: {rec['outcome']}. Ask something else."}}
        idx = best["cells"].index(asked.cell)
        weights = bayes_update(sc["weights"], [p[idx] for p in best["preds"]])
        answered = [*sc["answered"], {"canonical": best["canonical"], "bank": best["bank"], "cells": best["cells"], "cell": asked.cell, "idx": idx}]  # fmt: skip
        rec.update(cell=asked.cell, weights_after=[round(w, 4) for w in weights])
        return {"scratch": {**sc, "weights": weights, "answered": answered}}

    def _verdict(self, s) -> dict:
        sc = s["scratch"]
        hyps, weights = sc["hyps"], sc["weights"]
        probability = smoothed_share(weights, [h["label"] for h in hyps], float(s["alert"].p0))
        self.last_trace, self.last_stop = sc["trace"], sc["stop"]
        self.last_hypotheses = [{**h, "weight": w} for h, w in zip(hyps, weights, strict=True)]
        return {"verdict": s["session"].verdict(probability, s["budget"].usage)}

    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict:
        return run_graph(self.graph(), {"alert": alert, "tools": tools, "limits": limits})["verdict"]

    investigate_legacy = _Legacy.investigate
