"""THE system graph: one LangGraph StateGraph per investigator, shared by our planner and every ADAPTED arm.

Owner: T3 (Akshat).  Spec: LangGraph rewrite (wave 2).

    prepare -> check -> plan -> act -> fuse -> check -> ... -> decide

  prepare  evidence view, `rules_query(alert)` rules, library strategies, the routed menu (with the bin -> LR map), p0;
  check    stop rules: a stop reason already set, LLM calls / eps / simulated time used up, `max_failures_in_row` steps in a row
           without an allowed answer, and (opt-in) `cannot_cross`: the remaining eps cannot move the posterior across the threshold;
  plan     `policy.propose(view, llm, scratch)` -> Asks / Stop;
  act      route each Ask to the bank that holds its subject (ftm.llm.routing), enforce the wave rules, ask through the harness
           (Investigation.ask / ask_wave: gate + accountant + bank behind the ToolBox); the policy never names a bank;
  fuse     the configured Fusion (NaiveLRFusion, or ConditionalLR when a params file is given) turns the answers into the posterior
           (Investigation does the arithmetic as it books each answer; the node publishes it);
  decide   verdict = the fused posterior. Only a policy `finish` hook can override it (our llm_decides / no_backtest ablations).

LLM calls go through `BudgetedClient` (token / call caps, JSON schema, guided decoding); LangGraph is the orchestration layer only.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from ftm.compliance.query import rules_query
from ftm.decision.threshold import max_log_odds_shift
from ftm.dsl.parser import family as dsl_family
from ftm.dsl.parser import parse as dsl_parse
from ftm.graph.policy import Ask, MenuItem, PlannerPolicy, PlannerView, Stop
from ftm.graph.state import AlertState
from ftm.harness.interface import ToolBox
from ftm.library.store import DEFAULT_K as LIBRARY_K
from ftm.llm.budget import BudgetedClient
from ftm.llm.client import ChatClient
from ftm.llm.investigation import DuplicateQuestion, Fusion, Investigation, NaiveLRFusion, outcome
from ftm.llm.routing import NO_BANK, asked_bank, expected_bank
from ftm.planners.llm_dsl import DslAdapter, DslCheck, V0Dsl
from ftm.planners.llm_prompt import EPS_LEVELS
from ftm.types import AlertContext, Limits, QuestionLogEntry, Verdict

EvidenceFn = Callable[[AlertContext, ToolBox], str]
QuestionFilter = Callable[[dict, AlertContext], bool]  # does this listed question apply to this alert (slot binding)?
SUMMARY_HIDDEN = {"evidence", "partner_banks", "shape", "top_features"}  # own prompt sections; top_features: Sharma only (docs/t1_planner_view.md)
RECURSION_LIMIT = 10_000  # graph steps per alert (4 nodes per loop; the loop is bounded by the call cap and the failure rule)


def family_of(canonical: str) -> str | None:
    try:
        return dsl_family(dsl_parse(canonical))
    except Exception:  # noqa: BLE001 - an unparsable program has no family
        return None


def route(canonical: str, alert: AlertContext, requested) -> tuple[object, str | None]:
    """(bank to ask, refusal hint). The bank is the one that holds the question's subject, whatever the model wrote
    (the back-test only measured that bank); the hint is set when no partner holds it. Unknown routing (a summary
    without bank fields) keeps the requested bank."""
    eb = expected_bank(canonical, alert.summary, alert.home_bank)
    if eb is None:
        return requested, None
    if eb == NO_BANK:
        return requested, ("its subject is an account at the home bank, so no partner bank holds anything about it. "
                           "Use ALERT.sender or ALERT.receiver only for the party that is NOT at the home bank, or a set slot "
                           "(ALERT.payers / ALERT.payees)")
    return asked_bank(canonical, alert.summary, alert.home_bank), None


def summary_evidence(alert: AlertContext, tools: ToolBox) -> str:
    """Default evidence: the text the orchestrator put in `alert.summary['evidence']` (e.g. from
    ftm.retrieval.views). Every system gets the same evidence text (fairness protocol)."""
    return str(alert.summary.get("evidence", ""))


def library_strategies(alert: AlertContext, tools: ToolBox, typology: str | None) -> list[dict]:
    """C12 lookup by situation shape; the typology part of the key is the planner's current guess."""
    shape = {**alert.summary.get("shape", {}), "typology": typology or "any", "receiver": alert.summary.get("receiver")}  # receiver: E11 account keying only
    return [{"text": str(s.get("text", ""))[:300]} for s in tools.search_library(shape)[:LIBRARY_K]]


def _gate_of(tools):
    return getattr(getattr(tools, "_h", None), "gate", None)  # the harness's gate (jurisdictions); others: default jurisdiction


@dataclass
class AskResult:
    """What became of one Ask. Exactly one of `entry` (the question went out; it may still be a refusal: see
    entry.gate_decision), `duplicate` (the session had it already) or `rejected` (the pipeline did not send it)."""

    ask: Ask
    bank: Any = None
    eps: float = 0.0
    check: DslCheck | None = None
    entry: QuestionLogEntry | None = None
    duplicate: DuplicateQuestion | None = None
    rejected: str | None = None  # invalid_dsl | no_partner_holds_subject | not_in_menu | not_enough_eps | duplicate_refused | duplicate_in_wave | same_family_in_wave | wave_over_eps
    detail: str = ""

    @property
    def allowed(self) -> bool:
        return self.entry is not None and self.entry.gate_decision == "allowed"


class GraphInvestigator:
    """An Investigator (ftm.harness.interface) that runs `policy` inside the system graph. `llm` is the raw ChatClient: one
    BudgetedClient per alert is made from it."""

    def __init__(self, policy: PlannerPolicy, fusion: Fusion | None = None, llm: ChatClient | None = None, *,
                 dsl: DslAdapter | None = None, questions: list[dict] | None = None, question_filter: QuestionFilter | None = None,
                 evidence_fn: EvidenceFn = summary_evidence, use_rules: bool = True, use_library: bool = False,
                 threshold: float | None = None, max_failures_in_row: int = 3, allow_compose: bool = False,
                 stop_cannot_cross: bool = False):  # fmt: skip
        self.policy, self.llm = policy, llm
        self.fusion = fusion or NaiveLRFusion()
        self.dsl = dsl or V0Dsl()
        self.questions = list(questions or [])
        self.question_filter = question_filter
        self.evidence_fn = evidence_fn
        self.use_rules, self.use_library = use_rules, use_library
        self.threshold = threshold
        self.max_failures_in_row = max_failures_in_row
        self.allow_compose = allow_compose
        self.stop_cannot_cross = stop_cannot_cross
        self.last_trace: list[dict] = []
        self.last_state: AlertState | None = None
        self.graph = self._build()

    @property
    def system_name(self) -> str:
        return self.policy.name

    # -- the graph ------------------------------------------------------------------------------------------------
    def _build(self):
        g = StateGraph(AlertState)
        for name, fn in (("prepare", self._prepare), ("check", self._check), ("plan", self._plan), ("act", self._act),
                         ("fuse", self._fuse), ("decide", self._decide)):  # fmt: skip
            g.add_node(name, fn)
        g.add_edge(START, "prepare")
        g.add_edge("prepare", "check")
        g.add_conditional_edges("check", lambda s: "decide" if s.get("stop_reason") else "plan", {"decide": "decide", "plan": "plan"})
        g.add_conditional_edges("plan", self._after_plan, {"decide": "decide", "act": "act", "check": "check"})
        g.add_edge("act", "fuse")
        g.add_edge("fuse", "check")
        g.add_edge("decide", END)
        return g.compile()

    @staticmethod
    def _after_plan(s: AlertState) -> str:
        return "decide" if s.get("stop_reason") else "act" if s.get("pending") else "check"  # nothing proposed = a failed step

    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict:
        out = self.graph.invoke({"alert": alert, "tools": tools, "limits": limits, "scratch": {}},
                                {"recursion_limit": RECURSION_LIMIT})
        self.last_state = out
        self.last_trace = out["scratch"].get("trace", [])
        return out["verdict"]

    # -- nodes ----------------------------------------------------------------------------------------------------
    def _prepare(self, s: AlertState) -> dict:
        alert, tools, limits = s["alert"], s["tools"], s["limits"]
        session = Investigation(alert, tools, limits, fusion=self.fusion)
        budget = BudgetedClient(self.llm, limits, tag=alert.alert_id, trim=False)  # trims itself below
        evidence = self.evidence_fn(alert, tools)
        rules = []
        if self.use_rules:
            for r in tools.lookup_rules(rules_query(alert, _gate_of(tools)))[:5]:
                rules.append({"rule_id": str(r.get("rule_id")), "text": str(r.get("text") or r.get("snippet", ""))[:300]})
        strategies = library_strategies(alert, tools, None) if self.use_library else []
        menu_raw = []
        for q in self.questions:
            if self.question_filter is not None and not self.question_filter(q, alert):
                continue
            bank, none_held = route(q["program"], alert, None)
            if none_held is None:  # offered only where a partner holds the subject; each item names its bank
                menu_raw.append({**q, "bank": bank} if bank is not None else q)
        return {"session": session, "budget": budget, "evidence": evidence, "rules": rules, "strategies": strategies,
                "menu_raw": menu_raw, "menu": self._menu(menu_raw), "refused": {}, "step": 0, "failures": 0,
                "stop_reason": None, "pending": [], "results": [], "log": session.log, "posterior": session.posterior}  # fmt: skip

    def _menu(self, menu_raw: list[dict]) -> list[MenuItem]:
        table = getattr(self.fusion, "table", None)
        items = []
        for q in menu_raw:
            for eps in q.get("eps") or ():
                entry = table.get(q["program"], eps) if table is not None else None
                bins = sorted(entry.lr.items()) if entry is not None else []
                items.append(MenuItem(str(q["id"]), q["program"], float(eps), str(q.get("bank") or ""), str(q.get("family") or ""), bins))
        return items

    def _view(self, s: AlertState) -> PlannerView:
        alert, session, budget = s["alert"], s["session"], s["budget"]
        lr_of = getattr(self.fusion, "_lr", None)
        answers = []
        for e in session.log:
            r = e.response
            lr = lr_of(e) if lr_of is not None else None
            answers.append({"program": e.program, "bank": e.bank, "eps": e.eps, "outcome": outcome(e), "step": e.step,
                            "value": getattr(r, "value", None), "lr": lr})  # fmt: skip
        summary = {"home_bank": alert.home_bank, **{k: v for k, v in alert.summary.items() if k not in SUMMARY_HIDDEN}}
        return PlannerView(alert_summary=summary, p0=alert.p0, posterior=session.posterior, threshold=self.threshold,
                           evidence=s["evidence"], rules=s["rules"], strategies=s["strategies"], menu=s["menu"], answers=answers,
                           eps_left=session.eps_left, calls_left=budget.calls_left, step=s["step"],
                           allow_compose=self.allow_compose)  # fmt: skip

    def _cannot_cross(self, s: AlertState) -> bool:
        """The remaining eps cannot move the fused log-odds across the threshold. The bound per eps is derived from the menu's
        LRs (largest |log LR| per eps spent) and, for a conditional fusion, its largest family weight."""
        if self.threshold is None:
            return False
        session = s["session"]
        per_eps = max((max((abs(math.log(v)) for _, v in m.lr_bins if v > 0), default=0.0) / m.eps for m in s["menu"] if m.eps > 0),
                      default=0.0)  # fmt: skip
        model = getattr(self.fusion, "model", None)
        w = [abs(float(x)) for x in getattr(model, "w", ())]
        weight = max([1.0, *w, float(getattr(model, "default_w", 1.0))]) * float(getattr(model, "shrink", 1.0)) if model is not None else 1.0
        gap = abs(_logit(session.posterior) - _logit(self.threshold))
        return gap > max_log_odds_shift(session.eps_left, max_delta=per_eps, max_weight=weight)

    def _check(self, s: AlertState) -> dict:
        session, budget = s["session"], s["budget"]
        if s.get("stop_reason"):
            return {}
        reason = None
        if budget.calls_left <= 0:
            reason = "calls_used"
        elif session.eps_left + 1e-9 < min(EPS_LEVELS):
            reason = "eps_used"
        elif session.out_of_time:
            reason = "time_used"
        elif s["failures"] >= self.max_failures_in_row:
            reason = "failures"
        elif self.stop_cannot_cross and self._cannot_cross(s):
            reason = "cannot_cross"
        return {"stop_reason": reason}

    def _plan(self, s: AlertState) -> dict:
        step = s["step"] + 1
        view = self._view({**s, "step": step})
        scratch = s["scratch"]
        scratch["_state"] = s
        actions = list(self.policy.propose(view, s["budget"], scratch))
        no_fail = scratch.pop("_no_fail", False)  # the policy retried for a reason that is not a failure (a prompt refit)
        asks = [a for a in actions if isinstance(a, Ask)]
        stops = [a for a in actions if isinstance(a, Stop)]
        out: dict = {"step": step, "view": view, "pending": asks}
        if not asks:
            if stops:
                out["stop_reason"] = stops[0].reason or "policy_stop"
            elif not no_fail:
                out["failures"] = s["failures"] + 1
        elif stops:
            scratch["_stop_after"] = stops[0].reason or "policy_stop"
        return out

    def _act(self, s: AlertState) -> dict:
        alert, session, scratch = s["alert"], s["session"], s["scratch"]
        asks: list[Ask] = s["pending"]
        wave = scratch.pop("_wave", False) or len(asks) > 1
        hints = scratch.get("_bank_hint", {})
        refused = s["refused"]
        results = [AskResult(a) for a in asks]
        reserved, families, keys, ok = 0.0, set(), set(), []
        for r in results:
            a = r.ask
            r.check = self.dsl.validate(a.canonical)
            if not r.check.ok:
                r.rejected, r.detail = "invalid_dsl", str(r.check.error)
                continue
            canon = r.check.canonical
            requested = hints[canon] if canon in hints else (alert.summary.get("partner_banks") or [None])[0]
            r.bank, why_not = route(canon, alert, requested)
            if why_not:
                r.rejected, r.detail = "no_partner_holds_subject", why_not
                continue
            r.eps = float(a.eps)
            left = session.eps_left - reserved
            in_menu, levels = self._levels(s["menu_raw"], canon)
            if not self.allow_compose and not in_menu:
                r.rejected, r.detail = "not_in_menu", f"{canon} is not on the menu"
                continue
            if self.allow_compose:
                levels = None  # compose-capable policies pick their own level (ours snaps menu picks itself)
            if levels is not None:  # back-tested at these levels only: any other level would not move the probability
                ok_eps = [e for e in levels if e <= left + 1e-9]
                if not ok_eps:
                    r.rejected, r.detail = "not_enough_eps", f"{canon} needs eps {min(levels):g} but only {left:g} is left"
                    continue
                if r.eps not in ok_eps:
                    r.eps = min(ok_eps, key=lambda e: (abs(e - r.eps), e))
            if (canon, str(r.bank)) in refused:
                r.rejected, r.detail = "duplicate_refused", str(refused[(canon, str(r.bank))])
                continue
            if wave:
                fam = family_of(canon)
                if (canon, str(r.bank)) in keys:
                    r.rejected = "duplicate_in_wave"
                    continue
                if fam is not None and fam in families:
                    r.rejected, r.detail = "same_family_in_wave", fam
                    continue
                if r.eps > left + 1e-9:
                    r.rejected = "wave_over_eps"
                    continue
                keys.add((canon, str(r.bank)))
                if fam is not None:
                    families.add(fam)
                reserved += r.eps
            ok.append(r)
        step = s["step"]
        if wave and ok:
            for r, entry in zip(ok, session.ask_wave([(r.bank, r.check, r.eps, step) for r in ok]), strict=True):
                if isinstance(entry, DuplicateQuestion):
                    r.duplicate = entry
                else:
                    r.entry = entry
        elif ok:
            r = ok[0]
            try:
                r.entry = session.ask(r.bank, r.check, r.eps, step)
            except DuplicateQuestion as e:
                r.duplicate = e
        any_allowed, why_last = False, None
        for r in results:
            if r.entry is None:
                continue
            if r.entry.gate_decision == "allowed":
                any_allowed = True
                continue
            why_last = session.last_refusal or r.entry.gate_decision
            refused[(r.check.canonical, str(r.bank))] = why_last
        hook = getattr(self.policy, "observe", None)
        scratch["_state"] = s
        if hook is not None:
            hook(results, scratch)
        out: dict = {"results": results, "failures": 0 if any_allowed else s["failures"] + 1}
        stop_after = scratch.pop("_stop_after", None)
        if not any_allowed and why_last and "used up" in why_last:  # the clock or the per-alert eps budget is spent: no question can succeed any more
            out["stop_reason"] = "budget_used_up"
        elif stop_after:
            out["stop_reason"] = stop_after
        return out

    @staticmethod
    def _levels(menu_raw: list[dict], canonical: str) -> tuple[bool, list[float] | None]:
        """(is it on the menu, its back-tested eps levels if the menu names them)."""
        for q in menu_raw:
            if q["program"] == canonical:
                return True, (list(q["eps"]) if q.get("eps") else None)
        return False, None

    def _fuse(self, s: AlertState) -> dict:
        session = s["session"]
        return {"posterior": session.posterior, "log": session.log}

    def _decide(self, s: AlertState) -> dict:
        session, scratch = s["session"], s["scratch"]
        scratch["_state"] = s
        probability = None
        hook = getattr(self.policy, "finish", None)
        if hook is not None:
            probability = hook(s, scratch)
        if probability is None:
            probability = session.posterior
        return {"verdict": session.verdict(probability, s["budget"].usage)}


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))
