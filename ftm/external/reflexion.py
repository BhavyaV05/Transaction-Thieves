"""Reflexion (Shinn et al., NeurIPS 2023), REIMPLEMENTATION in the shared harness, ported mode.

Owner: T3 (Akshat).  Tier: Core.  Target week: 6.

Official code: github.com/noahshinn/reflexion (MIT); nothing is copied. Kept from the paper: an Actor (here the
same ReAct loop and tools as ftm/external/react.py), an Evaluator, a Self-Reflection step that writes a short verbal
lesson, and an episodic memory of those lessons fed to the next trial on the SAME alert.

Adaptations, fixed before any run (fairness protocol, plan.md 7.2):
  - The Evaluator is the LLM judging its own trial (CONFIDENT / RETRY). The paper's HotpotQA evaluator uses the
    true answer; we never show labels, even on validation, so the agent stays deployable.
  - Trials share ONE eps budget, ONE call cap and the same Investigation: bank answers already received are real
    facts, so the next trial sees them (and repeats are refused). The environment cannot be reset as in the paper.
  - At most `max_trials` (default 3) trials; a new trial needs at least `min_calls_for_retry` calls left.
End-to-end probability = finish[...] of the last trial that finished; None if none did. Verdict.trials records how
many trials ran on the alert (the runner writes it to traces.jsonl and summary.json).
"""

from __future__ import annotations

from ftm.harness.interface import ToolBox
from ftm.llm.budget import BudgetedClient, BudgetExceeded, PromptTooLong
from ftm.llm.investigation import describe
from ftm.planners.llm_prompt import EPS_LEVELS
from ftm.types import AlertContext, Limits, Verdict

from .react import React

EVALUATE = (
    "You are checking an anti-money-laundering investigation you just ran (below). Is the final probability well "
    "supported by the answers obtained, or would another attempt with different questions likely change it? "
    "Reply with exactly one word first: CONFIDENT or RETRY, then one sentence."
)
REFLECT = (
    "You are an advanced reasoning agent that can improve based on self-reflection. You will be given a previous "
    "investigation trial in which you could ask partner banks typed questions. In a few sentences, diagnose a "
    "possible reason the trial may have reached a poor conclusion, and devise a new, concise, high-level plan that "
    "avoids the same failure. Use complete sentences. Do not repeat questions already asked."
)


def render(messages: list[dict]) -> str:
    return "\n".join(m["content"] for m in messages[1:])


class Reflexion(React):
    name = "reflexion"

    def __init__(self, llm, *, max_trials: int = 3, min_calls_for_retry: int = 4, reflection_tokens: int = 160, **kw):
        super().__init__(llm, **kw)
        self.max_trials = max_trials
        self.min_calls_for_retry = min_calls_for_retry
        self.reflection_tokens = reflection_tokens

    def _ask(self, budget: BudgetedClient, system: str, user: str, max_tokens: int) -> str | None:
        try:
            return budget.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                               max_tokens=max_tokens, temperature=self.temperature, seed=self.seed,
                               extra=self.extra).text.strip()  # fmt: skip
        except (BudgetExceeded, PromptTooLong):
            return None

    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict:
        session, budget, ex = self.start(alert, tools, limits, self.name)
        memory: list[str] = []
        final, step, trials = None, 0, 0
        for trial in range(1, self.max_trials + 1):
            ex.finished = None
            trials = trial
            preface = ""
            if memory:
                preface += "Reflections from your earlier attempts on this alert:\n" + "\n".join(f"- {m}" for m in memory) + "\n\n"
            if session.log:
                preface += ("Answers already obtained (real; asking again is refused):\n"
                            + "\n".join(f"- {e.program} to bank {e.bank}: {describe(e)}" for e in session.log) + "\n\n")  # fmt: skip
            messages = self.first_messages(alert, ex, preface)
            step = self.episode(messages, ex, budget, step)
            if ex.finished is not None:
                final = ex.finished
            else:
                break  # calls or time ran out inside the trial
            out_of_room = (budget.calls_left < self.min_calls_for_retry or session.out_of_time
                           or session.eps_left + 1e-9 < min(EPS_LEVELS))  # fmt: skip
            if trial == self.max_trials or out_of_room:
                break
            judgement = self._ask(budget, EVALUATE, render(messages), 48)
            if judgement is None or not judgement.upper().startswith("RETRY"):
                break
            lesson = self._ask(budget, REFLECT, render(messages), self.reflection_tokens)
            if lesson is None:
                break
            memory.append(lesson.replace("\n", " ")[:600])
        self.last_trace = ex.log + [{"reflections": memory}]
        v = session.verdict(final, budget.usage)
        v.trials = trials
        return v
