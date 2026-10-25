"""Shared pieces of the Group D text-loop agents (ReAct, Reflexion, ReWOO, LATS, LLMCompiler) in PORTED mode.

Owner: T3 (Akshat).  Spec: tracks.md 5.7, plan.md 6.17 and 7.2 (fairness protocol).

Every Group D port gets exactly what our planner gets: the same alert summary, the same evidence text, the same
partner banks, the same question language, and `ask_bank` behind the same gate and accountant (through
ftm.llm.investigation.Investigation), with the same caps (BudgetedClient). Only the control loop differs.

Tool syntax (one per Action line; `|` separates arguments because DSL programs contain commas and brackets):
    ask_bank[<bank id> | <eps> | <question>]
    lookup_rules[<query>]
    finish[<probability of laundering, 0-1>]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ftm.harness.interface import ToolBox
from ftm.llm.investigation import DuplicateQuestion, Investigation, describe
from ftm.planners.llm_dsl import DslAdapter, retry_hint
from ftm.planners.llm_prompt import EPS_LEVELS, eps_levels
from ftm.types import AlertContext

TOOL_NAMES = ("ask_bank", "lookup_rules", "finish")
_ACTION = re.compile(r"(?:Action(?:\s*\d+)?\s*:\s*)?\b(ask_bank|lookup_rules|finish)\s*\[(.*)\]", re.DOTALL)
_SUMMARY_HIDDEN = {"evidence", "partner_banks", "shape", "top_features"}  # top_features: Sharma only (docs/t1_planner_view.md)


@dataclass(frozen=True)
class ToolCall:
    tool: str
    arg: str


def parse_tool_call(text: str) -> ToolCall | None:
    """The LAST `tool[...]` call in `text`; the argument runs to the last `]` on that line."""
    best = None
    for line in text.splitlines():
        m = _ACTION.search(line)
        if m:
            best = ToolCall(m.group(1), m.group(2).strip())
    return best


def parse_probability(arg: str) -> float | None:
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*%?", arg)
    if not m:
        return None
    s = m.group(0).strip()
    p = float(s.rstrip("%")) / (100 if s.endswith("%") else 1)
    return p if 0.0 <= p <= 1.0 else None


def tools_doc(dsl: DslAdapter, banks: list, eps_left: float) -> str:
    return (
        "Tools:\n"
        f"(1) ask_bank[<bank id> | <eps> | <question>]: ask one partner bank one question in the question language "
        f"below. Partner banks: {', '.join(map(str, banks))}. eps must be one of {list(EPS_LEVELS)} and at most the "
        f"eps left (now {eps_left:g}). The answer is ONE noisy value.\n"
        "(2) lookup_rules[<query>]: search the rules that say which questions are permitted.\n"
        "(3) finish[<p>]: end with your probability p (0 to 1) that the alert is money laundering.\n\n"
        + dsl.reference_text()
    )


def case_text(alert: AlertContext, evidence: str) -> str:
    summary = {k: v for k, v in alert.summary.items() if k not in _SUMMARY_HIDDEN}
    return f"Alert: {summary}\nEvidence (home bank's view, time-ordered, aliased):\n{evidence or '(none)'}"


@dataclass
class ToolExecutor:
    """Runs parsed tool calls for one alert and returns observation text (errors are observations too)."""

    alert: AlertContext
    tools: ToolBox
    session: Investigation
    dsl: DslAdapter
    banks: list
    finished: float | None = None
    n_invalid: int = 0
    n_questions: int = 0
    log: list[dict] = field(default_factory=list)

    def run(self, call: ToolCall | None, step: int) -> str:
        obs = self._run(call, step)
        self.log.append({"step": step, "tool": call.tool if call else None, "arg": call.arg if call else None,
                         "observation": obs})  # fmt: skip
        return obs

    def _run(self, call: ToolCall | None, step: int) -> str:
        if call is None:
            self.n_invalid += 1
            return "Invalid action. Write exactly one line: Action: ask_bank[bank | eps | question], lookup_rules[query] or finish[p]."
        if call.tool == "finish":
            p = parse_probability(call.arg)
            if p is None:
                self.n_invalid += 1
                return "finish needs a probability between 0 and 1, e.g. finish[0.35]."
            self.finished = p
            return f"Finished with probability {p:g}."
        if call.tool == "lookup_rules":
            rules = self.tools.lookup_rules(call.arg)[:3]
            return "\n".join(f"{r.get('rule_id')}: {str(r.get('text') or r.get('snippet', ''))[:300]}" for r in rules) or "No rules found."
        return self._ask(call.arg, step)

    def _ask(self, arg: str, step: int) -> str:
        prep = self._prepare_ask(arg)
        if isinstance(prep, str):
            return prep
        bank, check, eps = prep
        try:
            entry = self.session.ask(bank, check, eps, step)
        except DuplicateQuestion as e:
            return self._duplicate_obs(e)
        return self._ask_obs(entry)

    @staticmethod
    def _duplicate_obs(e: Exception) -> str:
        return f"{e}. Its answer is above; ask something else or finish."

    def _ask_obs(self, entry) -> str:
        obs = describe(entry)
        if entry.gate_decision != "allowed" and self.session.last_refusal:
            obs += f" Reason: {self.session.last_refusal}."
        return obs + f" eps left: {self.session.eps_left:g}."

    def _prepare_ask(self, arg: str):
        """Validate an ask_bank argument: an observation string if it is bad, else (bank, DslCheck, eps)."""
        parts = [p.strip() for p in arg.split("|", 2)]
        if len(parts) != 3:
            self.n_invalid += 1
            return "ask_bank needs three arguments: ask_bank[bank | eps | question]."
        bank_s, eps_s, program = parts
        bank = next((b for b in self.banks if str(b) == bank_s.removeprefix("bank").strip()), None)
        if bank is None:
            self.n_invalid += 1
            return f"Unknown bank {bank_s!r}. Partner banks: {', '.join(map(str, self.banks))}."
        try:
            eps = float(eps_s)
        except ValueError:
            eps = -1.0
        if eps not in EPS_LEVELS:
            self.n_invalid += 1
            return f"eps must be one of {list(EPS_LEVELS)}; eps left {self.session.eps_left:g} (allowed now: {eps_levels(self.session.eps_left)})."
        check = self.dsl.validate(program)
        self.n_questions += 1
        if not check.ok:
            self.n_invalid += 1
            return f"Invalid question: {check.error}. {retry_hint(program, check.error, self.dsl.reference_text())}"
        return bank, check, eps

    def run_wave(self, calls: list[tuple[ToolCall, int]]) -> list[str]:
        """Run independent (call, step) pairs as one wave: the ask_bank calls are validated first, then sent together
        through Investigation.ask_wave (the wave costs its slowest answer); other tools run as in run(). Observations
        come back in order and are logged like run() does."""
        obs: list[str | None] = [None] * len(calls)
        pending: list[tuple[int, tuple]] = []
        for i, (call, step) in enumerate(calls):
            if call.tool == "ask_bank":
                prep = self._prepare_ask(call.arg)
                if isinstance(prep, str):
                    obs[i] = prep
                else:
                    pending.append((i, (*prep, step)))
            else:
                obs[i] = self._run(call, step)
        if pending:
            for (i, _), res in zip(pending, self.session.ask_wave([r for _, r in pending]), strict=True):
                obs[i] = self._duplicate_obs(res) if isinstance(res, DuplicateQuestion) else self._ask_obs(res)
        for (call, step), o in zip(calls, obs, strict=True):
            self.log.append({"step": step, "tool": call.tool, "arg": call.arg, "observation": o})
        return obs  # type: ignore[return-value]
