"""LATS (Zhou et al. 2024, adapted: no rollouts) as a StateGraph, ported mode. Parity with ftm/external/lats.py is tested.

    expand (n children + value call) -> select* (best child; backtrack to a sibling when it fails) -> expand ... -> verdict
"""

from __future__ import annotations

import heapq

from ftm.external.lats import VALUE, failed, parse_scores
from ftm.external.lats import Lats as _Legacy
from ftm.external.text_agent import parse_tool_call
from ftm.harness.interface import ToolBox
from ftm.llm.budget import BudgetExceeded, PromptTooLong
from ftm.types import AlertContext, Limits, Verdict

from ._base import END, START, new_graph, run_graph


class Lats(_Legacy):
    def graph(self):
        if getattr(self, "_graph", None) is None:
            g = new_graph()
            for name, fn in (("init", self._init), ("expand", self._expand), ("select", self._select), ("verdict", self._verdict)):
                g.add_node(name, fn)
            g.add_edge(START, "init")
            g.add_conditional_edges("init", self._outer, ["expand", "verdict"])
            g.add_conditional_edges("expand", lambda s: "verdict" if s.get("halt") else self._inner(s), ["select", "expand", "verdict"])
            g.add_conditional_edges("select", self._inner, ["select", "expand", "verdict"])
            g.add_edge("verdict", END)
            self._graph = g.compile()
        return self._graph

    def _outer(self, s) -> str:
        return "expand" if (s["ex"].finished is None and not s["session"].out_of_time
                            and s["budget"].calls_left > self.n_children) else "verdict"

    def _inner(self, s) -> str:
        if s["scratch"]["frontier"] and s["ex"].finished is None and not s["scratch"]["moved"]:
            return "select"
        return self._outer(s)

    def _init(self, s) -> dict:
        session, budget, ex = self.start(s["alert"], s["tools"], s["limits"], self.name)
        return {"session": session, "budget": budget, "ex": ex, "messages": self.first_messages(s["alert"], ex), "step": 0,
                "scratch": {"frontier": [], "traj": None, "moved": False}}  # fmt: skip

    def _expand(self, s) -> dict:
        budget, messages, step = s["budget"], s["messages"], s["step"]
        try:
            children = []
            for j in range(self.n_children):  # expansion
                r = budget.chat(messages, max_tokens=self.max_tokens_per_step, temperature=self.child_temperature,
                                stop=["\nObservation", "Observation:"], seed=self.seed + 1000 * step + j,
                                extra=self.extra)  # fmt: skip
                if r.text.strip() and r.text.strip() not in children:
                    children.append(r.text.strip())
            listing = "\n\n".join(f"Candidate {i + 1}:\n{c}" for i, c in enumerate(children))
            v = budget.chat([{"role": "system", "content": VALUE},
                             {"role": "user", "content": "\n".join(m["content"] for m in messages[1:]) + "\n\n" + listing}],
                            max_tokens=512 + 32 * len(children), temperature=0.0, seed=self.seed, extra=self.extra)  # fmt: skip
        except (BudgetExceeded, PromptTooLong):
            return {"halt": True}
        frontier: list[tuple[float, int, str]] = []  # this expansion's siblings: (-score, tiebreak, child text)
        for tick, (c, sc) in enumerate(zip(children, parse_scores(v.text, len(children)), strict=True)):
            heapq.heappush(frontier, (-sc, tick, c))
        return {"scratch": {"frontier": frontier, "traj": messages, "moved": False}}

    def _select(self, s) -> dict:
        sc, ex = s["scratch"], s["ex"]
        frontier = list(sc["frontier"])
        _, _, child = heapq.heappop(frontier)
        step = s["step"] + 1
        obs = ex.run(parse_tool_call(child), step)
        messages = [*sc["traj"], {"role": "assistant", "content": child}]
        if ex.finished is None:
            messages.append({"role": "user", "content": f"Observation {step}: {obs}\nThought {step + 1}:"})
        return {"step": step, "messages": messages, "scratch": {**sc, "frontier": frontier, "moved": not failed(obs)}}

    def _verdict(self, s) -> dict:
        self.last_trace = s["ex"].log
        return {"verdict": s["session"].verdict(s["ex"].finished, s["budget"].usage)}

    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict:
        return run_graph(self.graph(), {"alert": alert, "tools": tools, "limits": limits})["verdict"]

    investigate_legacy = _Legacy.investigate
