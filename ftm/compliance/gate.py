"""C6 compliance gate v0: an allowlist in front of every tool call, identical for every system.

Owner: T4 (Shreyas).  Spec: plan.md 6.7, tracks.md 6.3 (hand-off #15 to T2/T3, week 5).

Default deny. A request is allowed only if its action is listed in configs/compliance/gate.yaml and every check
below passes. `ask_bank` checks, in order:
  1. the partner bank is not the home bank and is a known partner (when a partner list is given);
  2. eps is one of the allowed levels (> 0);
  3. the program parses (DSL validator from T2, plugged in) and its answer kind is permitted by a legal basis
     that applies to the partner bank's jurisdiction (US 314(b) or SG COSMIC);
  4. the program carries no identifying literal: no account alias (A + 7 hex), no raw account number, no date
     (questions refer to the alert only through ALERT.<slot>, plan.md 6.5);
  5. a cited rule_id exists and applies to that jurisdiction (required only when require_citation is on);
  6. the privacy accountant (T2, plugged in) can pay eps; otherwise the decision is "over_budget".
Raw-record requests are blocked unless `allow_raw_share` (the H2 baseline arm only).

Decisions use the QuestionLogEntry vocabulary: "allowed" | "blocked" | "over_budget". `GatedToolBox` raises
`GateBlocked` (a PermissionError) or `BudgetRefused`, the exceptions T3's Investigation layer expects (PR #7).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import yaml

from ftm.types import BankResponse, DslProgram

ALIAS_RE = re.compile(r"\bA[0-9a-f]{7}\b")
RAW_ACCOUNT_RE = re.compile(r"\b(?=[0-9A-F]*\d)(?=[0-9A-F]*[A-F])[0-9A-F]{8,}\b")  # AMLworld account numbers
DATE_RE = re.compile(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b")
RAW_SHARE_ACTIONS = {"request_records", "share_records"}
ROOT = Path(__file__).resolve().parents[2]
GATE_CONFIG = ROOT / "configs" / "compliance" / "gate.yaml"

# validator(program) -> (ok, answer_kind or None, error or None); T2's DSL plugs in here
Validator = Callable[[DslProgram], tuple[bool, str | None, str | None]]


class GateBlocked(PermissionError):
    """The gate refused the request (not allowed by any rule)."""


class BudgetRefused(RuntimeError):
    """The privacy accountant cannot pay for the question."""


class Accountant(Protocol):
    """T2's accountant. `can_spend` must not spend; `spend` is called only after the gate allows."""

    def can_spend(self, alert_id: str, bank: str, eps: float) -> bool: ...

    def spend(self, alert_id: str, bank: str, eps: float) -> None: ...


@dataclass(frozen=True)
class Rule:
    rule_id: str
    jurisdiction: str
    text: str
    source: str
    permits: tuple[str, ...]
    kinds: tuple[str, ...]

    def applies_to(self, jurisdiction: str) -> bool:
        return self.jurisdiction in (jurisdiction, "ANY")


@dataclass(frozen=True)
class Decision:
    decision: str  # allowed | blocked | over_budget
    reason: str = ""
    rule_id: str | None = None

    @property
    def allowed(self) -> bool:
        return self.decision == "allowed"


def load_rules(path: str | Path = ROOT / "configs" / "compliance" / "rules.yaml") -> dict[str, Rule]:
    path = Path(path)
    raw = yaml.safe_load((path if path.is_absolute() else ROOT / path).read_text())["rules"]
    return {r["rule_id"]: Rule(r["rule_id"], r["jurisdiction"], r["text"].strip(), r["source"],
                               tuple(r.get("permits", [])), tuple(r.get("kinds", []))) for r in raw}  # fmt: skip


def identifying_literals(program: str) -> list[str]:
    """Literals that would point a question at a specific account or day instead of an alert-relative slot."""
    return ALIAS_RE.findall(program) + RAW_ACCOUNT_RE.findall(program) + DATE_RE.findall(program)


@dataclass
class Gate:
    rules: dict[str, Rule]
    allowed_actions: frozenset[str]
    eps_levels: tuple[float, ...]
    validator: Validator | None = None
    accountant: Accountant | None = None
    require_citation: bool = False
    default_jurisdiction: str = "US"
    jurisdiction_by_bank: dict = field(default_factory=dict)
    allow_raw_share: bool = False

    @classmethod
    def from_config(cls, path: str | Path = GATE_CONFIG, **overrides) -> Gate:
        cfg = yaml.safe_load(Path(path).read_text())
        kw = dict(rules=load_rules(cfg["rules_file"]), allowed_actions=frozenset(cfg["allowed_actions"]),
                  eps_levels=tuple(float(e) for e in cfg["eps_levels"]), require_citation=cfg["require_citation"],
                  default_jurisdiction=cfg["default_jurisdiction"],
                  jurisdiction_by_bank={str(k): v for k, v in (cfg.get("jurisdiction_by_bank") or {}).items()},
                  allow_raw_share=cfg["allow_raw_share"])  # fmt: skip
        kw.update(overrides)
        return cls(**kw)

    def jurisdiction(self, bank) -> str:
        return self.jurisdiction_by_bank.get(str(bank), self.default_jurisdiction)

    def _rule_for(self, action: str, jur: str, kind: str | None, rule_id: str | None) -> Rule | None:
        if rule_id is not None:
            r = self.rules.get(rule_id)
            ok = r is not None and r.applies_to(jur) and action in r.permits and (kind is None or kind in r.kinds)
            return r if ok else None
        for r in self.rules.values():
            if r.applies_to(jur) and action in r.permits and (kind is None or kind in r.kinds):
                return r
        return None

    def check(self, action: str, *, alert_id: str = "", home_bank=None, bank=None, program: str | None = None,
              eps: float = 0.0, rule_id: str | None = None, partners=None) -> Decision:  # fmt: skip
        if action in RAW_SHARE_ACTIONS:
            if not self.allow_raw_share:
                return Decision("blocked", "raw records never leave a partner bank (raw-share arm only)")
            return Decision("allowed", "raw-share baseline arm")
        if action not in self.allowed_actions:
            return Decision("blocked", f"action {action!r} is not on the allowlist")
        if action != "ask_bank":
            if action in ("draft_sar", "escalate_to_human"):
                if bank is not None and home_bank is not None and str(bank) != str(home_bank):
                    return Decision("blocked", f"{action} stays inside the home bank")
                r = self._rule_for(action, "ANY", None, rule_id)
                return Decision("allowed", rule_id=r.rule_id) if r else Decision("blocked", f"no rule permits {action}")
            return Decision("allowed")
        # ask_bank
        if bank is None or program is None:
            return Decision("blocked", "ask_bank needs bank and program")
        if home_bank is not None and str(bank) == str(home_bank):
            return Decision("blocked", "a bank cannot ask itself")
        if partners is not None and str(bank) not in {str(p) for p in partners}:
            return Decision("blocked", f"bank {bank} is not a partner bank")
        if not any(abs(eps - e) < 1e-9 for e in self.eps_levels):
            return Decision("blocked", f"eps {eps} is not one of {list(self.eps_levels)}")
        kind = None
        if self.validator is not None:
            ok, kind, err = self.validator(program)
            if not ok:
                return Decision("blocked", f"invalid question: {err}")
        lits = identifying_literals(program)
        if lits:
            return Decision("blocked", f"question names specific accounts or dates: {lits[:3]}")
        jur = self.jurisdiction(bank)
        if self.require_citation and rule_id is None:
            return Decision("blocked", "a rule_id citation is required")
        r = self._rule_for("ask_bank", jur, kind, rule_id)
        if r is None:
            why = f"rule {rule_id!r} does not permit this question in {jur}" if rule_id else f"no {jur} rule permits it"
            return Decision("blocked", why)
        if self.accountant is not None and not self.accountant.can_spend(alert_id, str(bank), eps):
            return Decision("over_budget", "privacy accountant refused", r.rule_id)
        return Decision("allowed", rule_id=r.rule_id)

    def lookup_rules(self, query: str, jurisdiction: str | None = None, k: int = 5) -> list[dict]:
        """v0 keyword lookup (the BM25 + embedding RAG is secondary). Returns rule_id, jurisdiction, text, source."""
        words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
        scored = []
        for r in self.rules.values():
            if jurisdiction and not r.applies_to(jurisdiction):
                continue
            hay = f"{r.rule_id} {r.text} {r.source}".lower()
            scored.append((sum(w in hay for w in words), r.rule_id, r))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [{"rule_id": r.rule_id, "jurisdiction": r.jurisdiction, "text": r.text, "source": r.source}
                for s, _, r in scored[:k] if s > 0 or not words]  # fmt: skip


class GatedToolBox:
    """Wraps any ToolBox so every ask_bank goes through the gate and then the accountant, for every system."""

    def __init__(self, inner, gate: Gate, alert_id: str, home_bank, partners=None):
        self.inner, self.gate, self.alert_id, self.home_bank, self.partners = inner, gate, alert_id, home_bank, partners
        self.decisions: list[Decision] = []

    def ask_bank(self, bank: str, program: DslProgram, eps: float, rule_id: str | None = None) -> BankResponse:
        d = self.gate.check("ask_bank", alert_id=self.alert_id, home_bank=self.home_bank, bank=bank,
                            program=program, eps=eps, rule_id=rule_id, partners=self.partners)  # fmt: skip
        self.decisions.append(d)
        if d.decision == "blocked":
            raise GateBlocked(d.reason)
        if d.decision == "over_budget":
            raise BudgetRefused(d.reason)
        if self.gate.accountant is not None:
            self.gate.accountant.spend(self.alert_id, str(bank), eps)
        return self.inner.ask_bank(bank, program, eps)

    def lookup_rules(self, query: str) -> list[dict]:
        return self.gate.lookup_rules(query)

    def local_graph(self, query: str, as_of: str) -> list[dict]:
        return self.inner.local_graph(query, as_of)

    def search_library(self, shape: dict) -> list[dict]:
        return self.inner.search_library(shape)
