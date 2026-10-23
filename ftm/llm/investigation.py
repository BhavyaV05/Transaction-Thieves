"""Per-alert investigation state shared by every LLM investigator (our planner and the Group D ports).

Owner: T3 (Akshat).  Spec: tracks.md 4.8 (harness), 5.4, 5.7; plan.md 7.2 (fairness protocol).

One `Investigation` per alert does what must be identical across systems:
  - asks partner banks only through `ToolBox.ask_bank` (gate + accountant sit behind it, T2/T4);
  - tracks eps left, simulated time and the questions log (QuestionLogEntry, scored questions-only by eval);
  - refuses repeats of the same question to the same bank and questions over the eps left (no call is made);
  - updates the posterior with a `Fusion` (LLM text never enters it, invariant 2);
  - assembles the Verdict with the cost fields from `BudgetedClient`.

`NaiveLRFusion` + `LRTable` are PLACEHOLDERS for T4's fusion v0 (hand-off #18) and T2's back-test table
(hand-off #9): summed log-LRs of accepted questions on top of logit(p0), i.e. the "naive summed" ablation of
plan.md 6.9. Swap in the real ones through the `Fusion` protocol.

ASSUMPTION until harness v0 (T2, week 3) fixes it: a gate refusal surfaces from `ask_bank` as an exception whose
class is PermissionError or named *Gate*/*Blocked*/*Compliance*, and an accountant refusal as one named
*Budget*/*Accountant*. Anything else is a bug and propagates.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol

from ftm.harness.interface import ToolBox
from ftm.llm.budget import Usage
from ftm.llm.untrusted import wrap_untrusted
from ftm.planners.llm_dsl import DslCheck
from ftm.planners.llm_prompt import AnswerSoFar
from ftm.types import AlertContext, Decline, Limits, QuestionLogEntry, Reply, Timeout, Verdict

P_CLIP = 1e-6


class DuplicateQuestion(ValueError):
    pass


def logit(p: float) -> float:
    p = min(max(p, P_CLIP), 1 - P_CLIP)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


# -- back-test LRs and fusion (placeholders, see module docstring) --------------------------------------------
def answer_key(value, bins: list[float] | None = None) -> str:
    """bool -> 'true'/'false'; numbers with bin edges -> 'b<i>' (i = number of edges <= value); else str."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if bins is not None and isinstance(value, (int, float)):
        return f"b{bisect.bisect_right(bins, value)}"
    return str(value)


@dataclass(frozen=True)
class LREntry:
    canonical: str
    eps: float
    accepted: bool  # passed e-BH / e-LOND in back-testing
    lr: dict[str, float]  # answer_key -> clipped LR
    family: str | None = None
    bins: tuple[float, ...] | None = None

    def lr_for(self, value) -> float | None:
        return self.lr.get(answer_key(value, list(self.bins) if self.bins else None))


class LRTable:
    """Back-tested LRs keyed by (canonical program, eps)."""

    def __init__(self, entries: Iterable[LREntry] = ()):
        self._d = {(e.canonical, round(e.eps, 6)): e for e in entries}

    def get(self, canonical: str, eps: float) -> LREntry | None:
        return self._d.get((canonical, round(eps, 6)))

    @classmethod
    def from_rows(cls, rows: Iterable[dict]) -> LRTable:
        return cls(LREntry(r["program_canonical"], float(r["eps"]), bool(r["accepted"]), dict(r["lr"]),
                           r.get("family"), tuple(r["bins"]) if r.get("bins") else None) for r in rows)  # fmt: skip

    def __len__(self) -> int:
        return len(self._d)


class Fusion(Protocol):
    def posterior(self, p0: float, log: list[QuestionLogEntry]) -> float: ...

    def uses(self, entry: QuestionLogEntry) -> bool:
        """Whether this answered question moves the posterior (accepted back-test LR exists)."""
        ...


class NaiveLRFusion:
    def __init__(self, table: LRTable | None = None):
        self.table = table or LRTable()

    def _lr(self, e: QuestionLogEntry) -> float | None:
        if not isinstance(e.response, Reply) or e.gate_decision != "allowed":
            return None
        entry = self.table.get(e.program, e.eps)
        if entry is None or not entry.accepted:
            return None
        return entry.lr_for(e.response.value)

    def uses(self, entry: QuestionLogEntry) -> bool:
        return self._lr(entry) is not None

    def posterior(self, p0: float, log: list[QuestionLogEntry]) -> float:
        z = logit(p0)
        for e in log:
            lr = self._lr(e)
            if lr is not None and lr > 0:
                z += math.log(lr)
        return sigmoid(z)


# -- the session ----------------------------------------------------------------------------------------------
def _gate_outcome(exc: Exception) -> str | None:
    name = type(exc).__name__.lower()
    if "budget" in name or "accountant" in name:
        return "over_budget"
    if isinstance(exc, PermissionError) or any(w in name for w in ("gate", "blocked", "compliance")):
        return "blocked"
    return None


def outcome(entry: QuestionLogEntry) -> str:
    if entry.gate_decision != "allowed":
        return entry.gate_decision
    r = entry.response
    return "reply" if isinstance(r, Reply) else "decline" if isinstance(r, Decline) else "timeout"


def describe(entry: QuestionLogEntry) -> str:
    """Observation text for text-loop agents (ReAct etc.). Notes are marked untrusted."""
    o = outcome(entry)
    if o == "reply":
        r = entry.response
        text = f"bank {entry.bank} answered {r.value!r} ({r.kind}, noisy; eps {r.eps_spent:g} spent, {r.latency_days:.1f} days)"
        if r.note:
            text += f"\n{wrap_untrusted(r.note)}"
        return text
    if o == "decline":
        return f"bank {entry.bank} declined to answer"
    if o == "timeout":
        return f"bank {entry.bank} did not answer in time"
    if o == "blocked":
        return "the compliance gate blocked this question (not permitted); no eps spent"
    return "over budget: not enough eps left for this question; no eps spent"


@dataclass
class Investigation:
    alert: AlertContext
    tools: ToolBox
    limits: Limits
    fusion: Fusion = field(default_factory=NaiveLRFusion)
    log: list[QuestionLogEntry] = field(default_factory=list)
    eps_left: float = field(init=False)
    sim_days: float = 0.0
    posterior: float = field(init=False)
    _asked: set = field(default_factory=set, repr=False)
    _batch: list | None = field(default=None, repr=False)

    def __post_init__(self):
        self.eps_left = self.limits.eps_per_alert
        self.posterior = self.alert.p0

    @property
    def out_of_time(self) -> bool:
        return self.sim_days >= self.limits.max_sim_time_days

    def ask(self, bank: int | str, check: DslCheck, eps: float, step: int) -> QuestionLogEntry:
        """Ask one validated question. Raises DuplicateQuestion; never raises for gate/budget refusals."""
        if not check.ok:
            raise ValueError(f"ask() needs a validated question: {check.error}")
        key = (check.canonical, str(bank))
        if key in self._asked:
            raise DuplicateQuestion(f"already asked bank {bank}: {check.canonical}")
        entry = QuestionLogEntry(step=step, bank=str(bank), program=check.canonical, eps=float(eps))
        if eps > self.eps_left + 1e-9:
            entry.gate_decision = "over_budget"
        else:
            try:
                entry.response = self.tools.ask_bank(bank, check.canonical, eps)
            except Exception as exc:  # noqa: BLE001 - classified, unknown errors re-raised
                decision = _gate_outcome(exc)
                if decision is None:
                    raise
                entry.gate_decision = decision
        if entry.gate_decision == "allowed":
            self._asked.add(key)
            r = entry.response
            self.eps_left -= float(getattr(r, "eps_spent", 0.0) or 0.0)
            days = r.latency_days if isinstance(r, Reply) else r.after_days if isinstance(r, Timeout) else 0.0
            if self._batch is None:
                self.sim_days += days
            else:
                self._batch.append(days)
        self.log.append(entry)
        self.posterior = self.fusion.posterior(self.alert.p0, self.log)
        entry.posterior_after = self.posterior
        return entry

    @contextmanager
    def parallel(self) -> Iterator[None]:
        """Questions asked inside this block are sent at once: simulated time advances by the SLOWEST answer,
        not the sum (RQ5 / E13; LLMCompiler). Calls still run one by one, so ToolBox needs no thread safety."""
        if self._batch is not None:
            raise RuntimeError("parallel blocks do not nest")
        self._batch = []
        try:
            yield
        finally:
            self.sim_days += max(self._batch, default=0.0)
            self._batch = None

    def answers_so_far(self) -> list[AnswerSoFar]:
        out = []
        for e in self.log:
            o = outcome(e)
            r = e.response if isinstance(e.response, Reply) else None
            out.append(AnswerSoFar(step=e.step, bank=e.bank, program=e.program, eps=e.eps, outcome=o,
                                   value=r.value if r else None, note=r.note if r else None,
                                   used_in_posterior=self.fusion.uses(e) if r else None))  # fmt: skip
        return out

    def verdict(self, probability: float | None, usage: Usage) -> Verdict:
        return Verdict(alert_id=self.alert.alert_id, probability=probability, questions_log=list(self.log),
                       tokens_in=usage.tokens_in, tokens_out=usage.tokens_out, llm_calls=usage.llm_calls,
                       gpu_seconds=usage.gpu_seconds)  # fmt: skip
