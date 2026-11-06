"""FinFRE-RAG as a StateGraph, ported modes (native and ported). Parity with ftm/external/finfre_rag.py is tested.

Owner: T3 (Akshat).

    init -> [questions] -> score -(no "Score:" line yet?)-> score ... -> verdict
Exemplar retrieval, the prompt, the score regex and the 1-5 score table are the old class's (shared, not copied).
"""

from __future__ import annotations

from ftm.external.finfre_rag import _SCORE, FRAUD_AT
from ftm.external.finfre_rag import FinfreRag as _Legacy
from ftm.harness.interface import ToolBox
from ftm.llm.budget import BudgetedClient, BudgetExceeded, PromptTooLong
from ftm.llm.investigation import Investigation
from ftm.types import AlertContext, Limits, Verdict

from ._base import END, START, new_graph, run_graph

ATTEMPTS = 2  # the paper requires a "Score:" line; one repair call if it is missing


class FinfreRag(_Legacy):
    def graph(self):
        if getattr(self, "_graph", None) is None:
            g = new_graph()
            for name, fn in (("init", self._init), ("questions", self._questions_node), ("score", self._score), ("verdict", self._verdict)):
                g.add_node(name, fn)
            g.add_edge(START, "init")
            g.add_conditional_edges("init", lambda s: "questions" if self.mode == "ported" else "score", ["questions", "score"])
            g.add_edge("questions", "score")
            g.add_conditional_edges("score", lambda s: "verdict" if s["stop"] or s["step"] >= ATTEMPTS else "score", ["score", "verdict"])
            g.add_edge("verdict", END)
            self._graph = g.compile()
        return self._graph

    def _init(self, s) -> dict:
        alert, limits = s["alert"], s["limits"]
        session = Investigation(alert, s["tools"], limits, fusion=self.fusion)
        budget = BudgetedClient(self.llm, limits, tag=f"{self.name}:{alert.alert_id}")
        return {"session": session, "budget": budget, "step": 0, "stop": False,
                "scratch": {"lines": [], "messages": None, "shot_ids": [], "score": None, "text": ""}}  # fmt: skip

    def _questions_node(self, s) -> dict:
        return {"scratch": {**s["scratch"], "lines": self._ask_questions(s["budget"], s["session"], s["alert"])}}

    def _score(self, s) -> dict:
        sc, budget = s["scratch"], s["budget"]
        msgs, shot_ids = sc["messages"], sc["shot_ids"]
        if msgs is None:  # first attempt: build the few-shot prompt once, after the question round
            msgs, shot_ids = self.prompt(s["alert"], sc["lines"])
        step = s["step"] + 1
        try:
            r = budget.chat(msgs, max_tokens=self.max_tokens, temperature=self.temperature, seed=self.seed, extra=self.extra)
        except (BudgetExceeded, PromptTooLong):
            return {"step": step, "stop": True, "scratch": {**sc, "messages": msgs, "shot_ids": shot_ids}}
        text = r.text
        m = _SCORE.search(text)
        if m:
            return {"step": step, "stop": True, "scratch": {**sc, "messages": msgs, "shot_ids": shot_ids, "text": text, "score": int(m.group(1))}}
        msgs = [*msgs, {"role": "assistant", "content": text[:400]},
                {"role": "user", "content": "You must include 'Score:' followed by a number from 1 to 5."}]  # fmt: skip
        return {"step": step, "scratch": {**sc, "messages": msgs, "shot_ids": shot_ids, "text": text}}

    def _verdict(self, s) -> dict:
        sc = s["scratch"]
        score = sc["score"]
        probability = None if score is None else float(self.score_map[score])
        self.last_trace = {"score": score, "fraud_at_4": None if score is None else score >= FRAUD_AT, "exemplars": sc["shot_ids"],
                           "features": self.pool.selected, "importance_method": self.pool.importance_method,
                           "response": sc["text"][:600], "partner_lines": sc["lines"]}  # fmt: skip
        return {"verdict": s["session"].verdict(probability, s["budget"].usage)}

    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict:
        return run_graph(self.graph(), {"alert": alert, "tools": tools, "limits": limits})["verdict"]

    investigate_legacy = _Legacy.investigate
