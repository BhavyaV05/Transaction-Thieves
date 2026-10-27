"""Harness v0 (C16): the real ToolBox. Every system, ours or external, reaches the partner banks only through here.

Owner: T2 (Bhavya).  Spec: plan.md 6.16, tracks.md 4.8.  Contract: ftm/harness/interface.py (frozen at week 4).

`ask_bank(bank, program, eps)` is the only road to a partner bank and it is always the same road:
  1. the compliance gate (T4): default deny, valid question, no identifying literal, legal basis, allowed eps level;
  2. the privacy accountant (T2): the per-alert budget (the smaller of `Limits.eps_per_alert` and the accountant's)
     and the per-partner-account cap, with the accounts the question can see (`charge_keys`);
  3. the clock: a reply that would arrive after `Limits.max_sim_time_days` is cut off as a Timeout;
  4. the bank answers (exact + noise + latency); eps is charged only when a Reply comes back, because a decline or a
     timeout releases nothing.
A refusal raises `GateBlocked` (a PermissionError) or `BudgetRefused`, the classes T3's Investigation layer expects.
Nothing here hands out a bank, the gate or the accountant, and `run_one` replaces the investigator's own account of its
questions with the toolbox's log, so a system cannot misreport what it asked.

One toolbox object is reused across alerts (T3's runner builds it once): `begin(alert, limits)` starts a session for
one alert, `end()` closes it. `wrap(investigator)` does both around `investigate`.

Modes: "ported" (ask_bank available, the main mode), "native" (no ask_bank: systems run as published, one-bank view).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml

from ftm.banks.binding import AlertBinding
from ftm.banks.partner import PartnerBank, Question, _seed
from ftm.compliance.gate import BudgetRefused, Gate, GateBlocked
from ftm.dsl.parser import DslError, arity, canonical, gate_validator, parse
from ftm.guards import TimeGuardError
from ftm.privacy.accountant import BudgetExceeded, PrivacyAccountant, charge_keys
from ftm.types import (
    AlertContext,
    BankResponse,
    Decline,
    Limits,
    QuestionLogEntry,
    Reply,
    Timeout,
    Verdict,
)

TOL = 1e-9
PRIVACY_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "privacy" / "default.yaml"


class ToolUnavailable(RuntimeError):
    """The tool does not exist in this mode (e.g. ask_bank in native mode)."""


class SessionError(RuntimeError):
    """A tool was used outside begin()/end()."""


# --- a bank that answers at random, for offline development (T3/T4) -----------------------------------------------
class MockBank:
    """Same `respond` interface as `PartnerBank`, no ledger: random typed answers, seeded per question. Never a result."""

    needs_binding = False  # random answers: no ledger, no real ids

    def __init__(self, bank: int, seed: int = 0, p_decline: float = 0.05):
        self.bank, self.seed, self.p_decline = int(bank), seed, p_decline

    def respond(self, question: Question, as_of: datetime, purpose: str = "live") -> tuple[BankResponse, float]:
        try:
            node = parse(question.program)
        except DslError as e:
            return Decline(self.bank, f"invalid question: {e}"), 0.02
        alert_id = question.alert.alert_id if question.alert else "-"
        rng = np.random.default_rng(_seed("mock", self.seed, self.bank, alert_id, canonical(node), question.eps))
        latency = float(rng.uniform(0.05, 1.5))
        if rng.random() < self.p_decline:
            return Decline(self.bank, "declined"), latency
        kind = node.prim.kind
        value = (bool(rng.random() < 0.5) if kind == "bool" else int(rng.integers(0, arity(node)))
                 if kind == "kary" else int(rng.poisson(2.0)))  # fmt: skip
        return Reply(self.bank, kind, value, question.eps, latency, None), latency


# --- the session ------------------------------------------------------------------------------------------------
@dataclass
class Harness:
    """The world a run lives in: banks, gate, accountant. Build toolboxes from it, one session per alert."""

    banks: Mapping[int, PartnerBank | MockBank]
    gate: Gate
    accountant: PrivacyAccountant
    bindings: Mapping[str, AlertBinding] = field(default_factory=dict)  # alert_id -> real ids (simulator-internal)
    graph_fn: Callable[[str, datetime], list[dict]] | None = None  # T1's aliased evidence views
    library_fn: Callable[[dict], list[dict]] | None = None  # T3's strategy library
    mode: str = "ported"
    allow_exact: bool = False  # the no-noise arm (eps = inf); never set outside the eps sweep
    exact_all: bool = False  # LLM no-noise arm (E2, eps = inf): every ask is answered exactly and free, whatever level the system names
    t0_of: Callable[[AlertContext], datetime] | None = None

    def toolbox(self) -> HarnessToolBox:
        return HarnessToolBox(self)


class HarnessToolBox:
    def __init__(self, harness: Harness):
        self._h = harness
        self._alert: AlertContext | None = None
        self._limits: Limits | None = None
        self._binding: AlertBinding | None = None
        self._log: list[QuestionLogEntry] = []
        self._eps_spent = 0.0
        self._elapsed = 0.0
        self._t0: datetime | None = None
        self.last_elapsed_days = 0.0  # simulated days the last closed session took

    # --- sessions ------------------------------------------------------------------------------------------------
    def begin(self, alert: AlertContext, limits: Limits) -> None:
        self._alert, self._limits = alert, limits
        self._binding = self._h.bindings.get(alert.alert_id)
        self._t0 = self._binding.t0 if self._binding else datetime.fromisoformat(alert.as_of)
        self._log, self._eps_spent, self._elapsed = [], 0.0, 0.0

    def end(self) -> list[QuestionLogEntry]:
        log = list(self._log)
        self.last_elapsed_days = self._elapsed
        self._alert = self._limits = self._binding = self._t0 = None
        return log

    def wrap(self, investigator):
        return _Wrapped(investigator, self)

    def _need(self) -> tuple[AlertContext, Limits]:
        if self._alert is None or self._limits is None:
            raise SessionError("call begin(alert, limits) first (or run the investigator through toolbox.wrap(...))")
        return self._alert, self._limits

    @property
    def questions_log(self) -> list[QuestionLogEntry]:
        return list(self._log)

    @property
    def eps_spent(self) -> float:
        return self._eps_spent

    @property
    def elapsed_days(self) -> float:
        return self._elapsed

    # --- the tools -----------------------------------------------------------------------------------------------
    def local_graph(self, query: str, as_of: str) -> list[dict]:
        alert, _ = self._need()
        if datetime.fromisoformat(as_of) > datetime.fromisoformat(alert.as_of):
            raise TimeGuardError(f"as_of {as_of} is after the alert time {alert.as_of}")
        self._allow("local_graph", alert)
        return list(self._h.graph_fn(query, datetime.fromisoformat(as_of))) if self._h.graph_fn else []

    def lookup_rules(self, query: str) -> list[dict]:
        alert, _ = self._need()
        self._allow("lookup_rules", alert)
        return self._h.gate.lookup_rules(query, self._h.gate.jurisdiction(alert.home_bank))

    def search_library(self, shape: dict) -> list[dict]:
        alert, _ = self._need()
        self._allow("search_library", alert)
        return list(self._h.library_fn(shape)) if self._h.library_fn else []

    def _allow(self, action: str, alert: AlertContext) -> None:
        d = self._h.gate.check(action, alert_id=alert.alert_id, home_bank=alert.home_bank)
        if not d.allowed:
            raise GateBlocked(d.reason)

    def ask_bank(self, bank, program: str, eps: float, rule_id: str | None = None) -> BankResponse:
        response, wait = self._ask_one(bank, program, eps, rule_id, self._elapsed)
        self._elapsed += wait
        return response

    def ask_banks(self, requests: list[tuple]) -> list:
        """A wave: several questions asked at the same simulated moment. Each goes through the gate and the accountant
        on its own; the wave costs its slowest answer (not the sum), the rule T3's LLMCompiler also uses. Returns one
        entry per request, in order: the bank's response, or the `GateBlocked` / `BudgetRefused` it ran into (not raised,
        so one refusal does not lose the others). Not part of the frozen v1 `ToolBox`: an extension (`BatchToolBox`)."""
        start, results, waits = self._elapsed, [], [0.0]
        for req in requests:
            bank, program, eps = req[:3]
            try:
                response, wait = self._ask_one(bank, program, eps, req[3] if len(req) > 3 else None, start)
            except (GateBlocked, BudgetRefused) as exc:
                results.append(exc)
                continue
            results.append(response)
            waits.append(wait)
        self._elapsed = start + max(waits)
        return results

    def _ask_one(self, bank, program: str, eps: float, rule_id: str | None, start: float):
        """One question asked at simulated time `start` (days since the alert). Returns (response, days to wait);
        the caller moves the clock."""
        alert, limits = self._need()
        h = self._h
        if h.mode == "native":
            raise ToolUnavailable("ask_bank does not exist in native mode")
        entry = QuestionLogEntry(step=len(self._log) + 1, bank=str(bank), program=_canon_or_raw(program), eps=float(eps))
        self._log.append(entry)

        def refuse(decision: str, reason: str, exc: type[Exception]):
            entry.gate_decision = decision
            raise exc(reason)

        partners = [b for b in h.banks if str(b) != str(alert.home_bank)]
        # The no-noise arm of the eps sweep (eps = inf) is an explicit opt-in of the harness: the gate still checks
        # everything but the eps level (it is checked at the largest level), and there is nothing to price.
        exact = h.allow_exact and (h.exact_all or (isinstance(eps, float) and math.isinf(eps) and eps > 0))  # exact_all: any eps, int too
        d = h.gate.check("ask_bank", alert_id=alert.alert_id, home_bank=alert.home_bank, bank=bank, program=program,
                         eps=h.gate.eps_levels[-1] if exact else eps, rule_id=rule_id, partners=partners)  # fmt: skip
        if d.decision == "blocked":
            refuse("blocked", d.reason, GateBlocked)
        if d.decision == "over_budget":
            refuse("over_budget", d.reason, BudgetRefused)
        try:
            node = parse(program)  # the gate may have no validator plugged in: never run what does not parse
        except DslError as e:
            refuse("blocked", f"invalid question: {e}", GateBlocked)
        if not exact and (not (eps > 0) or math.isinf(eps)):
            refuse("blocked", f"eps must be positive and finite, got {eps!r}", GateBlocked)
        bank_obj = h.banks.get(int(bank))
        if bank_obj is None:
            refuse("blocked", f"bank {bank} is not a partner bank", GateBlocked)
        if self._binding is None and getattr(bank_obj, "needs_binding", True):
            entry.gate_decision = "blocked"
            raise SessionError(f"no real-id binding for alert {alert.alert_id}: cannot ask a bank with a ledger")
        if not exact and self._eps_spent + eps > limits.eps_per_alert + TOL:
            refuse("over_budget", "per-alert eps budget of this run is used up", BudgetRefused)
        if start >= limits.max_sim_time_days - TOL:
            refuse("over_budget", "simulated time for this alert is used up", BudgetRefused)
        keys = charge_keys(node, self._binding, int(bank)) if self._binding is not None else []
        if not exact and not h.accountant.can_spend(alert.alert_id, int(bank), eps, keys):
            refuse("over_budget", "privacy accountant refused", BudgetRefused)

        as_of = self._t0 + timedelta(days=start)
        binding = self._binding
        response, wait = bank_obj.respond(Question(program, math.inf if exact else eps, binding), as_of)
        if start + wait > limits.max_sim_time_days + TOL:  # would arrive after the deadline
            wait = limits.max_sim_time_days - start
            response = Timeout(int(bank), after_days=wait)
        if isinstance(response, Reply) and not exact:
            try:
                h.accountant.spend(alert.alert_id, int(bank), eps, keys)
            except BudgetExceeded as e:  # cannot happen after can_spend; if it does, the answer is withheld
                refuse("over_budget", str(e), BudgetRefused)
            self._eps_spent += eps
        entry.response = response
        return response, wait


def _canon_or_raw(program: str) -> str:
    try:
        return canonical(parse(program))
    except DslError:
        return str(program)


class _Wrapped:
    """An Investigator whose every `investigate` call runs inside a toolbox session, and whose verdict carries the
    toolbox's log of the questions, not the investigator's own."""

    def __init__(self, inner, tools: HarnessToolBox):
        self._inner, self._tools = inner, tools

    def __getattr__(self, name):  # let scripts read system.last_trace etc.
        return getattr(self._inner, name)

    def investigate(self, alert: AlertContext, tools, limits: Limits) -> Verdict:
        return run_one(self._inner, alert, self._tools, limits)


def run_one(investigator, alert: AlertContext, tools: HarnessToolBox, limits: Limits) -> Verdict:
    tools.begin(alert, limits)
    try:
        v = investigator.investigate(alert, tools, limits)
    finally:
        log = tools.end()
    return replace(v, questions_log=log)


# --- construction ---------------------------------------------------------------------------------------------------
def make_toolbox(seed: int = 0, mode: str = "mock", n_banks: int = 30, data: str | None = None, split: str = "val",
                 eps_levels: tuple[float, ...] | None = None, gate_overrides: dict | None = None,
                 max_alerts: int | None = None, eps_per_alert: float | None = None,
                 no_noise: bool = False) -> HarnessToolBox:  # fmt: skip
    """The toolbox T3's runner asks for (`ftm.harness.make_toolbox(seed=...)`).

    mode "mock": random typed answers from banks 1..n_banks, through the real gate and accountant (offline dev).
    mode "world": the real partner banks and alerts of `data` ("hi_small"): replies come from the ledgers with real
    noise. `split` is train or val (test only under the G5 lock, FTM_FINAL_RUN=1).

    E2 (privacy-detection curve) for the LLM systems: `eps_per_alert` replaces the accountant's per-alert budget (0.5, 1, 4);
    `no_noise` is the eps = inf arm: every ask is answered exactly and costs nothing (`Harness.exact_all`), and the
    per-alert budget is unlimited (pass `Limits(eps_per_alert=inf)` to the system too).
    """
    with open(PRIVACY_CONFIG, encoding="utf-8") as f:
        acc = PrivacyAccountant.from_config(yaml.safe_load(f))
    if eps_per_alert is not None:
        acc.eps_per_alert = float(eps_per_alert)
    if no_noise:
        acc.eps_per_alert = math.inf
    gate = Gate.from_config(validator=gate_validator, accountant=acc, **(gate_overrides or {}))
    if eps_levels is not None:
        gate.eps_levels = tuple(eps_levels)
    if mode == "world":
        from ftm.backtest.world import load_world

        w = load_world(data or "hi_small", split, seed, max_alerts=max_alerts)
        banks: dict = dict(w.banks)
        bindings = {a.alert_id: a.binding for a in w.alerts}
        return Harness(banks, gate, acc, bindings, allow_exact=no_noise, exact_all=no_noise).toolbox()
    if mode != "mock":
        raise ValueError("mode must be 'mock' or 'world'")
    banks = {b: MockBank(b, seed) for b in range(1, n_banks + 1)}
    return Harness(banks, gate, acc, {}, allow_exact=no_noise, exact_all=no_noise).toolbox()


__all__ = [
    "Harness", "HarnessToolBox", "MockBank", "SessionError", "ToolUnavailable", "make_toolbox", "run_one",
]  # fmt: skip
