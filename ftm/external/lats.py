"""LATS, Language Agent Tree Search (Zhou et al., ICML 2024), ADAPTED REIMPLEMENTATION, ported mode.

Owner: T3 (Akshat).  Tier: Secondary.  Target week: 8.

Official code: github.com/lapisrocks/LanguageAgentTreeSearch (MIT); nothing is copied. LATS runs Monte Carlo tree
search over ReAct trajectories and needs environment RESETS for its simulated rollouts. Here every ask_bank call
spends privacy budget and cannot be undone, so rollouts are impossible without breaking the fairness protocol.
Kept from the paper:
  - expansion: sample `n_children` Thought/Action continuations of the current trajectory (one call each,
    temperature > 0, different seeds);
  - LM value function: one call scores every child 1-10 given the trajectory so far;
  - selection: execute the best child; keep its siblings on a frontier;
  - backtracking: if the executed action fails (invalid, blocked, declined, timed out, repeated), try the next best
    SIBLING instead. The frontier lives for one expansion only: after a successful step it is discarded, because a
    sibling of an earlier step carries an old message context while the Investigation (asked questions, eps spent)
    has moved on, so running it would re-ask or spend eps on stale reasoning. If every sibling fails, the next round
    re-expands from the current trajectory (the last failed step and its observation), bounded by the call caps.
Dropped: simulated rollouts and value back-propagation (they need resets), reflection on failed terminals.
Labelled "LATS (adapted: no rollouts)" in Table 5a. Calls per step = n_children + 1, so the shared call cap binds
quickly ("cap tokens strictly", tracks.md 5.7).
"""

from __future__ import annotations

import heapq
import re

from ftm.harness.interface import ToolBox
from ftm.llm.budget import BudgetExceeded, PromptTooLong
from ftm.types import AlertContext, Limits, Verdict

from .react import React
from .text_agent import ToolCall, parse_tool_call

VALUE = (
    "You are scoring candidate next steps of an anti-money-laundering investigation (trajectory below). For each "
    "candidate, judge how useful it is for reaching a correct, well-supported probability with little privacy "
    "budget. Reply with one line per candidate: 'Candidate <i>: <score 1-10>'."
)
_SCORE = re.compile(r"Candidate\s*(\d+)\s*:\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_FAILED = ("Invalid", "Unknown bank", "eps must be", "declined", "did not answer", "blocked", "already asked",
           "over budget", "needs three arguments", "finish needs")  # fmt: skip


def failed(observation: str) -> bool:
    """Whether an executed action failed (first line only: a partner note below it is untrusted text)."""
    first = observation.split("\n", 1)[0]
    return any(f in first for f in _FAILED)


def parse_scores(text: str, n: int) -> list[float]:
    scores = [1.0] * n
    for i, s in _SCORE.findall(text):
        if 1 <= int(i) <= n:
            scores[int(i) - 1] = float(s)
    return scores


class Lats(React):
    name = "lats"

    def __init__(self, llm, *, n_children: int = 3, child_temperature: float = 0.7, **kw):
        super().__init__(llm, **kw)
        self.n_children = n_children
        self.child_temperature = child_temperature

    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict:
        session, budget, ex = self.start(alert, tools, limits, self.name)
        messages = self.first_messages(alert, ex)
        step = 0
        try:
            while ex.finished is None and not session.out_of_time and budget.calls_left > self.n_children:
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
                                max_tokens=16 * len(children), temperature=0.0, seed=self.seed, extra=self.extra)  # fmt: skip
                frontier: list[tuple[float, int, str]] = []  # this expansion's siblings: (-score, tiebreak, child text)
                for tick, (c, s) in enumerate(zip(children, parse_scores(v.text, len(children)), strict=True)):
                    heapq.heappush(frontier, (-s, tick, c))
                traj = messages
                while frontier and ex.finished is None:  # selection, backtracking among siblings
                    _, _, child = heapq.heappop(frontier)
                    step += 1
                    call: ToolCall | None = parse_tool_call(child)
                    obs = ex.run(call, step)
                    messages = [*traj, {"role": "assistant", "content": child}]
                    if ex.finished is None:
                        messages.append({"role": "user", "content": f"Observation {step}: {obs}\nThought {step + 1}:"})
                    if not failed(obs):
                        break
        except (BudgetExceeded, PromptTooLong):
            pass
        self.last_trace = ex.log
        return session.verdict(ex.finished, budget.usage)
