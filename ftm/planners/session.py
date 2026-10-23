"""One alert's investigation state for the non-LLM planners: budgets, the posterior, the stop rule, the log.

Owner: T2 (Bhavya).  The same loop, budgets and stop rule T4 specified (ftm/decision/threshold.py `should_stop`) that
every system uses; the LLM planners have their own twin of this (T3's `Investigation`, same rules).

Evidence enters only as numbers: the back-tested log-LR of an answer cell, summed per family and fused by T4's
`ftm.fusion.fuse.posterior`. Reply notes are never read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ftm.compliance.gate import BudgetRefused, GateBlocked
from ftm.decision.threshold import StopDecision, should_stop
from ftm.fusion.fuse import Fusion, posterior
from ftm.harness.interface import ToolBox
from ftm.planners.pool import Candidate
from ftm.types import AlertContext, Limits, QuestionLogEntry, Reply, Timeout, Verdict

MIN_EPS = 0.25  # smallest eps level (configs/compliance/gate.yaml)


def counterparty(alert: AlertContext) -> tuple[str, int]:
    """(direction, partner bank) of the alert from its aliased summary: 'out' if the home bank sent the money."""
    s = alert.summary
    if "direction" not in s or "counterparty_bank" not in s:
        raise KeyError("non-LLM planners need alert.summary['direction'] ('out'|'in') and ['counterparty_bank']")
    return s["direction"], int(s["counterparty_bank"])


@dataclass
class PlanSession:
    alert: AlertContext
    tools: ToolBox
    limits: Limits
    fusion: Fusion | None = None
    threshold: float | None = None  # the decision threshold, for the cannot_cross stop (None: budget stops only)
    max_delta: float = 1.0
    max_weight: float = 1.0
    log: list = field(default_factory=list)
    asked: set = field(default_factory=set)
    families_asked: set = field(default_factory=set)
    log_lr_by_family: dict = field(default_factory=dict)
    eps_left: float = field(init=False)
    sim_days: float = 0.0
    p: float = field(init=False)

    def __post_init__(self):
        self.eps_left = self.limits.eps_per_alert
        self.p = float(self.alert.p0)
        self.direction, self.bank = counterparty(self.alert)

    def stop(self) -> StopDecision:
        if self.threshold is None:
            out = self.eps_left + 1e-12 < MIN_EPS or self.sim_days >= self.limits.max_sim_time_days
            return StopDecision(out, "budget" if out else "continue")
        return should_stop(self.p, self.threshold, eps_left=self.eps_left, min_eps=MIN_EPS,
                           calls_left=self.limits.max_llm_calls - len(self.log),
                           sim_days_left=self.limits.max_sim_time_days - self.sim_days,
                           max_delta=self.max_delta, max_weight=self.max_weight)  # fmt: skip

    def stop_reason(self) -> str:
        """Why the investigation ended (run-log `stop_reason`): the stop rule's reason, or `finish` when it was still
        allowed to continue but had nothing left to ask."""
        d = self.stop()
        return d.reason if d.stop else "finish"

    def affordable(self, candidates: list[Candidate], one_per_family: bool = True) -> list[Candidate]:
        return [c for c in candidates
                if c.cost <= self.eps_left + 1e-9 and (c.program, self.bank) not in self.asked
                and not (one_per_family and c.family in self.families_asked)]  # fmt: skip

    def ask(self, c: Candidate) -> QuestionLogEntry:
        entry = QuestionLogEntry(step=len(self.log) + 1, bank=str(self.bank), program=c.program, eps=c.eps)
        self.asked.add((c.program, self.bank))
        try:
            entry.response = self.tools.ask_bank(self.bank, c.program, c.eps)
        except GateBlocked:
            entry.gate_decision = "blocked"
        except BudgetRefused:
            entry.gate_decision = "over_budget"
        wait = self._absorb(c, entry)
        self.sim_days += wait
        return self._finish(entry)

    def ask_batch(self, cands: list[Candidate]) -> list[QuestionLogEntry]:
        """Ask several questions as one wave (the toolbox's `ask_banks`; the wave costs its slowest answer). Falls back to
        one after another if the toolbox has no waves. The posterior is updated once, after all the answers are in."""
        if not hasattr(self.tools, "ask_banks"):
            return [self.ask(c) for c in cands]
        results = self.tools.ask_banks([(self.bank, c.program, c.eps) for c in cands])
        entries, waits = [], [0.0]
        for c, res in zip(cands, results, strict=True):
            entry = QuestionLogEntry(step=len(self.log) + len(entries) + 1, bank=str(self.bank), program=c.program, eps=c.eps)
            self.asked.add((c.program, self.bank))
            if isinstance(res, GateBlocked):
                entry.gate_decision = "blocked"
            elif isinstance(res, BudgetRefused):
                entry.gate_decision = "over_budget"
            else:
                entry.response = res
            waits.append(self._absorb(c, entry))
            entries.append(entry)
        self.sim_days += max(waits)
        for e in entries:
            self._finish(e)
        return entries

    def _absorb(self, c: Candidate, entry: QuestionLogEntry) -> float:
        """Book one response into the budgets and the evidence; returns the days it took."""
        r = entry.response
        if isinstance(r, Reply):
            if math.isfinite(r.eps_spent):
                self.eps_left -= r.eps_spent
            self.families_asked.add(c.family)
            llr = c.log_lr(r.value)
            if llr is not None:
                self.log_lr_by_family[c.family] = self.log_lr_by_family.get(c.family, 0.0) + llr
            return r.latency_days
        return r.after_days if isinstance(r, Timeout) else 0.0

    def _finish(self, entry: QuestionLogEntry) -> QuestionLogEntry:
        self.p = posterior(float(self.alert.p0), self.log_lr_by_family, self.fusion)
        entry.posterior_after = self.p
        self.log.append(entry)
        return entry

    def verdict(self) -> Verdict:
        return Verdict(alert_id=self.alert.alert_id, probability=self.p, questions_log=list(self.log))
