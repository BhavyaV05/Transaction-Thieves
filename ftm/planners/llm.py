"""System 2 LLM planner (C10).

Owner: T3 (Akshat).  Component: C10.  Tier: Core (planner); Secondary (fine-tuning).  Spec: plan.md 6.11, tracks.md 5.4.

Loop per alert (one LLM call per step, through BudgetedClient so the shared caps bind):
  1. build the prompt (alert summary, aliased evidence, listed questions, strategies, rules, answers so far,
     eps and calls left, the back-tested posterior) and the per-state JSON schema (llm_prompt.py);
  2. parse the step; on a bad JSON / bad DSL / repeated question, feed the error back and retry (costs a call);
  3. select -> listed question; compose -> new DSL question (exported for back-testing, hand-off #8); stop -> end;
  4. ask through Investigation.ask (gate + accountant behind ToolBox.ask_bank) and update the posterior.
Stops on: stop action, calls used, eps left below the smallest level, simulated time used, or 3 failures in a row.

Verdict probability: the formula's posterior (Fusion) in compose / fixed_list / act_only; the LLM's own `belief`
in no_backtest and llm_decides (the H4 / H7 ablations). LLM text never enters fusion.
Done when >=95% of outputs parse into valid DSL on validation and the loop runs at 8-20 s per alert.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ftm.harness.interface import ToolBox
from ftm.llm.budget import BudgetedClient, BudgetExceeded, PromptTooLong
from ftm.llm.client import ChatClient
from ftm.llm.investigation import DuplicateQuestion, Fusion, Investigation, NaiveLRFusion, outcome
from ftm.planners.llm_batches import QuestionBatchWriter
from ftm.planners.llm_dsl import DslAdapter, V0Dsl
from ftm.planners.llm_prompt import (
    EPS_LEVELS,
    Mode,
    PlannerParseError,
    PlannerState,
    build_messages,
    output_token_budget,
    parse_action,
    planner_schema,
)
from ftm.types import AlertContext, Limits, Reply, Verdict

EvidenceFn = Callable[[AlertContext, ToolBox], str]
RULES_QUERY = "sharing information with another financial institution about suspected money laundering"
_SUMMARY_HIDDEN = {"evidence", "partner_banks", "shape"}  # passed in their own prompt sections


def summary_evidence(alert: AlertContext, tools: ToolBox) -> str:
    """Default evidence: the text the orchestrator put in `alert.summary['evidence']` (e.g. from
    ftm.retrieval.views). Every system gets the same evidence text (fairness protocol)."""
    return str(alert.summary.get("evidence", ""))


@dataclass
class PlannerStats:
    """Aggregated over alerts. `usable_output_rate` is the >=95% acceptance metric (stops count as usable)."""

    alerts: int = 0
    outputs: int = 0
    json_ok: int = 0
    stops: int = 0
    questions: int = 0
    dsl_ok: int = 0
    duplicates: int = 0
    seconds_per_alert: list[float] = field(default_factory=list)

    @property
    def json_parse_rate(self) -> float:
        return self.json_ok / self.outputs if self.outputs else float("nan")

    @property
    def valid_dsl_rate(self) -> float:
        return self.dsl_ok / self.questions if self.questions else float("nan")

    @property
    def usable_output_rate(self) -> float:
        return (self.stops + self.dsl_ok) / self.outputs if self.outputs else float("nan")

    def as_counts(self) -> dict:
        """The integer counters, so a caller can take per-alert differences (E18 per_alert.jsonl)."""
        return {k: getattr(self, k) for k in ("outputs", "json_ok", "stops", "questions", "dsl_ok", "duplicates")}

    def as_dict(self) -> dict:
        s = sorted(self.seconds_per_alert)
        return {"alerts": self.alerts, "outputs": self.outputs, "json_parse_rate": self.json_parse_rate,
                "valid_dsl_rate": self.valid_dsl_rate, "usable_output_rate": self.usable_output_rate,
                "duplicates": self.duplicates,
                "median_seconds_per_alert": s[len(s) // 2] if s else float("nan")}  # fmt: skip


class LLMPlanner:
    """Our System 2 planner. Implements ftm.harness.interface.Investigator."""

    def __init__(
        self,
        llm: ChatClient,
        *,
        mode: Mode | str = Mode.COMPOSE,
        dsl: DslAdapter | None = None,
        fusion: Fusion | None = None,
        questions: list[dict] | None = None,
        evidence_fn: EvidenceFn = summary_evidence,
        use_rules: bool = True,
        use_library: bool = False,
        batch_writer: QuestionBatchWriter | None = None,
        temperature: float = 0.0,
        seed: int = 0,
        extra: dict | None = None,
        max_failures_in_row: int = 3,
        keep_prompts: bool = False,
        max_output_tokens: int | None = None,
    ):
        self.llm = llm
        self.mode = Mode(mode)
        self.dsl = dsl or V0Dsl()
        self.fusion = fusion or NaiveLRFusion()
        self.questions = {q["id"]: q for q in (questions or [])}
        if self.mode is Mode.FIXED_LIST and not self.questions:
            raise ValueError("fixed_list mode needs a question list")
        self.evidence_fn = evidence_fn
        self.use_rules = use_rules
        self.use_library = use_library
        self.batch_writer = batch_writer
        self.temperature, self.seed, self.extra = temperature, seed, extra
        self.max_failures_in_row = max_failures_in_row
        self.max_output_tokens = max_output_tokens  # None = the JSON-sized default; E18 thinking arm: 1,024 per step
        self.keep_prompts = keep_prompts  # teacher runs: traces keep the exact prompts for distillation
        self.stats = PlannerStats()
        self.last_trace: list[dict] = []

    @property
    def system_name(self) -> str:
        return f"ours_{self.mode.value}"

    def _context(self, alert: AlertContext, tools: ToolBox) -> tuple[str, list[dict], list[dict]]:
        evidence = self.evidence_fn(alert, tools)
        rules = []
        if self.use_rules:
            for r in tools.lookup_rules(RULES_QUERY)[:5]:
                rules.append({"rule_id": str(r.get("rule_id")), "text": str(r.get("text") or r.get("snippet", ""))[:300]})
        return evidence, rules, self._strategies(alert, tools, None)

    def _strategies(self, alert: AlertContext, tools: ToolBox, typology: str | None) -> list[dict]:
        """C12 lookup by situation shape; the typology part of the key is the planner's current guess."""
        if not self.use_library:
            return []
        shape = {**alert.summary.get("shape", {}), "typology": typology or "any"}
        return [{"text": str(s.get("text", ""))[:300]} for s in tools.search_library(shape)[:5]]

    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict:
        t0 = time.perf_counter()
        session = Investigation(alert, tools, limits, fusion=self.fusion)
        budget = BudgetedClient(self.llm, limits, tag=alert.alert_id)
        evidence, rules, strategies = self._context(alert, tools)
        banks = list(alert.summary.get("partner_banks", []))
        summary = {k: v for k, v in alert.summary.items() if k not in _SUMMARY_HIDDEN}
        trace: list[dict] = []
        feedback, failures, belief, typology = None, 0, None, None
        step = 0
        while (budget.calls_left > 0 and session.eps_left + 1e-9 >= min(EPS_LEVELS) and not session.out_of_time
               and failures < self.max_failures_in_row):  # fmt: skip
            step += 1
            state = PlannerState(
                alert_summary=summary, evidence_text=evidence, partner_banks=banks, eps_left=session.eps_left,
                calls_left=budget.calls_left, questions=list(self.questions.values()), strategies=strategies,
                rules=rules, answers=session.answers_so_far(), posterior=session.posterior, feedback=feedback,
            )  # fmt: skip
            rec: dict = {"step": step}
            trace.append(rec)
            try:
                msgs = build_messages(self.mode, state, self.dsl, limits.max_tokens)
                resp = budget.chat(msgs, json_schema=planner_schema(self.mode, state),
                                   max_tokens=self.max_output_tokens or output_token_budget(self.mode), temperature=self.temperature,
                                   seed=self.seed, extra=self.extra)  # fmt: skip
            except (BudgetExceeded, PromptTooLong, PlannerParseError) as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                break
            self.stats.outputs += 1
            rec.update(text=resp.text, tokens_in=resp.tokens_in, tokens_out=resp.tokens_out, latency_s=resp.latency_s,
                       posterior_before=session.posterior)
            if self.keep_prompts:
                rec["messages"] = msgs
            try:
                action = parse_action(resp.text, self.mode, state)
            except PlannerParseError as e:
                feedback, failures = f"{e}. Reply with one JSON object matching the keys listed under Output.", failures + 1
                rec["error"] = f"parse: {e}"
                continue
            self.stats.json_ok += 1
            rec.update(action=action.action, typology=action.typology, rationale=action.rationale, belief=action.belief)
            if action.belief is not None:
                belief = action.belief
            if self.use_library and action.typology not in (None, "none", "unsure", typology):
                typology = action.typology
                strategies = self._strategies(alert, tools, typology)
            if action.action == "stop":
                self.stats.stops += 1
                break
            self.stats.questions += 1
            program = self.questions[action.question_id]["program"] if action.action == "select" else action.program
            check = self.dsl.validate(program)
            rec["program"] = program
            if not check.ok:
                feedback, failures = f"invalid question {program!r}: {check.error}", failures + 1
                rec["error"] = f"dsl: {check.error}"
                continue
            self.stats.dsl_ok += 1
            try:
                entry = session.ask(action.bank, check, action.eps, step)
            except DuplicateQuestion as e:
                self.stats.duplicates += 1
                feedback, failures = f"{e}. Ask something new or stop.", failures + 1
                rec["error"] = "duplicate"
                continue
            failures, feedback = 0, None
            rec.update(canonical=check.canonical, bank=action.bank, eps=action.eps, outcome=outcome(entry),
                       posterior_after=entry.posterior_after)  # fmt: skip
            if isinstance(entry.response, Reply):  # the answer, so distillation can re-fuse with the real fusion model
                rec.update(answer_value=entry.response.value, answer_kind=entry.response.kind)
            if action.action == "compose" and self.batch_writer is not None and entry.gate_decision == "allowed":
                self.batch_writer.add(canonical=check.canonical, kind=check.kind, family=check.family, eps=action.eps,
                                      bank=action.bank, alert_id=alert.alert_id, split=alert.split, step=step,
                                      mode=self.mode.value, typology_guess=action.typology)  # fmt: skip
        if self.mode.has_belief:
            probability = belief if belief is not None else alert.p0
        else:
            probability = session.posterior
        self.stats.alerts += 1
        self.stats.seconds_per_alert.append(time.perf_counter() - t0)
        self.last_trace = trace
        return session.verdict(probability, budget.usage)
