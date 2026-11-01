"""System 2 LLM planner (C10).

Owner: T3 (Akshat).  Component: C10.  Tier: Core (planner); Secondary (fine-tuning).  Spec: plan.md 6.11, tracks.md 5.4.

Since the LangGraph rewrite this module is a thin wrapper: the loop is the system graph (ftm/graph/pipeline.py) running our step
logic (ftm/graph/ours.py: `OursPolicy`); the description below is what that loop does. The pre-graph loop is frozen in
tests/legacy_llm_planner.py as the parity oracle.

Loop per alert (one LLM call per step, through BudgetedClient so the shared caps bind):
  1. build the prompt (alert summary, aliased evidence, listed questions, strategies, rules, answers so far,
     eps and calls left, the back-tested posterior) and the per-state JSON schema (llm_prompt.py);
  2. parse the step; on a bad JSON / bad DSL / repeated question, feed the error back and retry (costs a call);
  3. select -> listed question; compose -> new DSL question (exported for back-testing, hand-off #8); stop -> end;
  4. ask through Investigation.ask (gate + accountant behind ToolBox.ask_bank) and update the posterior.
Batched option (E13, H8): `batch_schedule=(2, 1)` makes planner step i propose up to schedule[i] questions in one JSON
object; they go out as ONE wave (Investigation.ask_wave: the clock advances by the slowest answer), then the next
step. Same definition as T2's `waves=` for the non-LLM planners: stop rules are checked before each wave, no two
questions of one family in a wave, the wave's eps must fit the eps left together, and the investigation ends when the
schedule is used up. Off (None) = the sequential loop, unchanged.
Stops on: stop action, calls used, eps left below the smallest level, simulated time used, or 3 failures in a row.

Verdict probability: the formula's posterior (Fusion) in compose / fixed_list / act_only; the LLM's own `belief`
in no_backtest and llm_decides (the H4 / H7 ablations). LLM text never enters fusion.
Done when >=95% of outputs parse into valid DSL on validation and the loop runs at 8-20 s per alert.
"""

from __future__ import annotations

import time

from ftm.graph.ours import MAX_PROMPT_REFITS, PROMPT_REFIT_MARGIN, OursPolicy, PlannerStats
from ftm.graph.pipeline import (
    EvidenceFn,
    GraphInvestigator,
    QuestionFilter,
    family_of,
    route,
    summary_evidence,
)
from ftm.harness.interface import ToolBox
from ftm.llm.client import ChatClient
from ftm.llm.investigation import Fusion, NaiveLRFusion
from ftm.planners.llm_batches import QuestionBatchWriter
from ftm.planners.llm_dsl import DslAdapter, V0Dsl
from ftm.planners.llm_prompt import Mode
from ftm.types import AlertContext, Limits, Verdict

_family = family_of
__all__ = ["MAX_PROMPT_REFITS", "PROMPT_REFIT_MARGIN", "EvidenceFn", "LLMPlanner", "PlannerStats", "QuestionFilter", "route", "summary_evidence"]


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
        question_filter: QuestionFilter | None = None,
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
        batch_schedule: tuple[int, ...] | None = None,
        threshold: float | None = None,
    ):
        self.llm = llm
        self.mode = Mode(mode)
        self.dsl = dsl or V0Dsl()
        self.fusion = fusion or NaiveLRFusion()
        if self.mode is Mode.NO_BACKTEST:  # the ablation sees no back-test information: a plain list, no eps, no LRs
            questions = [{k: q[k] for k in ("id", "program", "family", "note") if k in q} for q in (questions or [])]
        self.questions = {q["id"]: q for q in (questions or [])}
        self.question_filter = question_filter
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
        self.batch_schedule = tuple(int(k) for k in batch_schedule) if batch_schedule else None
        if self.batch_schedule and (min(self.batch_schedule) < 1 or max(self.batch_schedule) > 4):
            raise ValueError(f"batch_schedule sizes must be in 1..4, got {self.batch_schedule}")
        self.threshold = threshold  # shown in the prompt when set (the stop-without-asking hint)
        self.stats = PlannerStats()
        self.last_trace: list[dict] = []
        self.policy = OursPolicy(mode=self.mode, dsl=self.dsl, questions=self.questions, use_library=use_library,
                                 batch_writer=batch_writer, temperature=temperature, seed=seed, extra=extra,
                                 keep_prompts=keep_prompts, max_output_tokens=max_output_tokens,
                                 batch_schedule=self.batch_schedule, stats=self.stats)  # fmt: skip
        self.graph = GraphInvestigator(self.policy, self.fusion, llm, dsl=self.dsl, questions=list(self.questions.values()),
                                       question_filter=question_filter, evidence_fn=evidence_fn, use_rules=use_rules,
                                       use_library=use_library, threshold=threshold, max_failures_in_row=max_failures_in_row,
                                       allow_compose=self.mode.can_compose)  # fmt: skip

    @property
    def system_name(self) -> str:
        return f"ours_{self.mode.value}"

    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict:
        t0 = time.perf_counter()
        v = self.graph.investigate(alert, tools, limits)
        self.stats.alerts += 1
        self.stats.seconds_per_alert.append(time.perf_counter() - t0)
        self.last_trace = self.graph.last_trace
        return v
