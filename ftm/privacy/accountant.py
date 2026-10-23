"""The privacy accountant (C5): per-alert budget, per-partner-account cap, separate eps_cal. Basic composition.

Owner: T2 (Bhavya).  Spec: plan.md 6.6, tracks.md 4.5.  Plugs into the compliance gate (ftm/compliance/gate.py,
`Accountant` protocol): the gate calls `can_spend` and, after it allows, the harness calls `spend`.

Three limits, all by summed eps (basic composition):
  * per alert: `eps_per_alert` (default 2) across every question asked for one alert, whichever bank is asked;
  * per partner account: `per_account_cap` across all alerts and all asking banks for the whole test period. A
    question is charged to every partner account whose transactions it can see (`charge_keys`): the subject for an
    account question, every member of the set for a set question. `accounts` empty means no account is charged;
  * calibration: back-testing spends from a separate ledger (`*_cal`, limit `eps_cal`), reported on its own.

`None` for a cap means unlimited (it is still tracked and reported). The no-noise arm (eps = inf) uses
`NullAccountant`, which never refuses and records nothing.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from ftm.dsl.parser import Node
from ftm.dsl.spec import ACCOUNT_SLOTS, SET_SLOTS

TOL = 1e-9
AccountKey = tuple  # (partner bank, node id)


class BudgetExceeded(RuntimeError):
    """A spend that `can_spend` would have refused (the gate normally stops it first)."""


def charge_keys(node: Node, binding, bank) -> list[AccountKey]:
    """Partner accounts whose transactions the question can see, as (bank, node id) keys.

    `binding` is an `ftm.banks.binding.AlertBinding` (anything with `slot(name)`)."""
    while isinstance(node.subject, Node):  # at_least wraps a count question
        node = node.subject
    if node.subject in ACCOUNT_SLOTS:
        return [(bank, binding.slot(node.subject))]
    if node.subject in SET_SLOTS:
        return [(bank, a) for a in sorted(binding.slot(node.subject))]
    return []


@dataclass
class PrivacyAccountant:
    eps_per_alert: float = 2.0
    per_account_cap: float | None = None
    eps_cal: float | None = None
    alert_spent: dict = field(default_factory=lambda: defaultdict(float))
    account_spent: dict = field(default_factory=lambda: defaultdict(float))
    cal_spent: float = 0.0
    cal_by_bank: dict = field(default_factory=lambda: defaultdict(float))
    cal_account_spent: dict = field(default_factory=lambda: defaultdict(float))
    n_spends: int = 0
    refusals: Counter = field(default_factory=Counter)

    # --- live questions --------------------------------------------------------------------------------------
    def _reason(self, alert_id: str, eps: float, accounts: Iterable[AccountKey]) -> str | None:
        if not (eps > 0) or math.isinf(eps):
            raise ValueError(f"the accountant only prices finite positive eps, got {eps!r}")
        if self.alert_spent.get(alert_id, 0.0) + eps > self.eps_per_alert + TOL:
            return "alert"
        if self.per_account_cap is not None:
            if any(self.account_spent.get(k, 0.0) + eps > self.per_account_cap + TOL for k in accounts):
                return "account"
        return None

    def can_spend(self, alert_id: str, bank, eps: float, accounts: Iterable[AccountKey] = ()) -> bool:
        """True if the question fits every limit. Does not spend. Counts a refusal when it says no."""
        reason = self._reason(alert_id, eps, list(accounts))
        if reason:
            self.refusals[reason] += 1
        return reason is None

    def spend(self, alert_id: str, bank, eps: float, accounts: Iterable[AccountKey] = ()) -> None:
        accounts = list(accounts)
        reason = self._reason(alert_id, eps, accounts)
        if reason:
            self.refusals[reason] += 1
            raise BudgetExceeded(f"over the {reason} budget (alert {alert_id}, bank {bank}, eps {eps})")
        self.alert_spent[alert_id] += eps
        for k in set(accounts):
            self.account_spent[k] += eps
        self.n_spends += 1

    def remaining(self, alert_id: str) -> float:
        return max(0.0, self.eps_per_alert - self.alert_spent.get(alert_id, 0.0))

    # --- calibration (back-testing) --------------------------------------------------------------------------
    def can_spend_cal(self, eps: float) -> bool:
        return self.eps_cal is None or self.cal_spent + eps <= self.eps_cal + TOL

    def spend_cal(self, bank, eps: float, accounts: Iterable[AccountKey] = ()) -> None:
        """Back-testing cost. Separate from live spending, reported as eps_cal."""
        if not (eps > 0) or math.isinf(eps):
            raise ValueError(f"the accountant only prices finite positive eps, got {eps!r}")
        if not self.can_spend_cal(eps):
            self.refusals["cal"] += 1
            raise BudgetExceeded(f"over eps_cal (bank {bank}, eps {eps})")
        self.cal_spent += eps
        self.cal_by_bank[bank] += eps
        for k in set(accounts):
            self.cal_account_spent[k] += eps

    # --- reporting and state ---------------------------------------------------------------------------------
    def report(self) -> dict:
        per_acc = list(self.account_spent.values())
        return {
            "eps_per_alert": self.eps_per_alert,
            "per_account_cap": self.per_account_cap,
            "eps_cal_limit": self.eps_cal,
            "n_spends": self.n_spends,
            "n_alerts": len(self.alert_spent),
            "total_eps_live": sum(self.alert_spent.values()),
            "max_alert_eps": max(self.alert_spent.values(), default=0.0),
            "n_accounts_charged": len(per_acc),
            "max_account_eps": max(per_acc, default=0.0),
            "eps_cal_spent": self.cal_spent,
            "refusals": dict(self.refusals),
        }

    def to_dict(self) -> dict:
        key = lambda k: list(k)  # noqa: E731
        return {
            "eps_per_alert": self.eps_per_alert, "per_account_cap": self.per_account_cap, "eps_cal": self.eps_cal,
            "alert_spent": dict(self.alert_spent),
            "account_spent": [[key(k), v] for k, v in self.account_spent.items()],
            "cal_spent": self.cal_spent, "cal_by_bank": {str(k): v for k, v in self.cal_by_bank.items()},
            "n_spends": self.n_spends, "refusals": dict(self.refusals),
        }  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict) -> PrivacyAccountant:
        a = cls(d["eps_per_alert"], d["per_account_cap"], d["eps_cal"])
        a.alert_spent.update(d["alert_spent"])
        a.account_spent.update({tuple(k): v for k, v in d["account_spent"]})
        a.cal_spent = d["cal_spent"]
        a.cal_by_bank.update(d["cal_by_bank"])
        a.n_spends = d["n_spends"]
        a.refusals.update(d["refusals"])
        return a

    @classmethod
    def from_config(cls, cfg: dict) -> PrivacyAccountant:
        return cls(float(cfg["eps_per_alert"]), cfg.get("per_account_cap"), cfg.get("eps_cal"))


class NullAccountant:
    """For the no-noise arm (eps = inf): nothing to price, nothing refused, nothing recorded."""

    def can_spend(self, alert_id, bank, eps, accounts=()) -> bool:
        return True

    def spend(self, alert_id, bank, eps, accounts=()) -> None:
        return None
