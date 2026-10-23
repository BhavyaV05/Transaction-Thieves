"""The question language (C4): the registry of primitives and their types.

Owner: T2 (Bhavya).  Spec: plan.md 6.5, tracks.md 4.3.  Grammar: ftm/dsl/grammar.lark.  Docs: ftm/dsl/README.md.

A question is one call `name(SUBJECT, key=value, ...)`. Calls nest in two places only: a time window
(`before(72)`) and the threshold wrapper `at_least(<count question>, k=5)`. Everything else is a flat primitive.
Nothing in the language can name an account, an alias, a date or an amount of a particular transaction: the only
account references are the ALERT slots, which the asked bank binds on its own ledger.

Each primitive carries what the rest of the system needs about it: the answer kind (C5 picks the mechanism), the
sensitivity (C5 scales the noise), the family keys (C7/C8 group questions into families) and the types of its
parameters (the type checker and the LLM's reference text are both generated from this table).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ftm.types import AnswerKind

REQUIRED: Any = object()  # sentinel: the parameter has no default

# Slots are relative to the alert. Account slots name one account, set slots name a set of accounts.
ACCOUNT_SLOTS = {
    "sender": "the account that sent the alerted transaction",
    "receiver": "the account that received the alerted transaction",
}
SET_SLOTS = {
    "payers": "accounts that paid ALERT.receiver in the 72 h before the alert (home bank's view)",
    "payees": "accounts that ALERT.sender paid in the 72 h before the alert (home bank's view)",
}

WINDOW_MODES = ("before", "after", "around")
WINDOW_MODE_DOC = {
    "before": "the h hours up to and including the alert time",
    "after": "the h hours after the alert time (only as far as the partner's answer time)",
    "around": "h/2 hours on each side of the alert time",
}
MAX_WINDOW_H = 336  # 14 days
PAYMENT_FORMATS = ("ach", "bitcoin", "cash", "cheque", "credit_card", "reinvestment", "wire")
DIRECTIONS = ("in", "out", "any")
MAX_BANKS = 5


@dataclass(frozen=True)
class Window:
    mode: str
    hours: int

    def __str__(self) -> str:
        return f"{self.mode}({self.hours})"


@dataclass(frozen=True)
class P:
    """One parameter. kind: int | float | enum | window | banks | set_slot."""

    kind: str
    lo: float | None = None
    hi: float | None = None
    choices: tuple[str, ...] = ()
    default: Any = REQUIRED


@dataclass(frozen=True)
class Prim:
    name: str
    kind: AnswerKind
    subject: str  # "account" | "set" | "count" (the first argument is itself a count/set_count question)
    params: dict[str, P]
    sensitivity: int  # max change of the exact answer when one transaction is added or removed
    doc: str
    example: str
    family_keys: tuple[str, ...] = ()  # parameters that split the primitive into families (C7/C8)
    restricted: bool = False  # registered but not askable (needs a privacy decision)
    needs_entities: bool = False  # the executor needs the entity map
    arity: int | None = None  # k-ary: number of answer values; None = derived from `banks`
    extra: dict = field(default_factory=dict)


_W_ANY = P("window", choices=WINDOW_MODES)
_W_AFTER = P("window", choices=("after",))
_W_AROUND = P("window", choices=("around",))
_DIR = P("enum", choices=("in", "out", "any"))
_DIR_IO = P("enum", choices=("in", "out"))
_NO_DEFAULT_AMOUNT = 1e9

PRIMS: dict[str, Prim] = {}


def _add(p: Prim) -> None:
    assert p.name not in PRIMS
    PRIMS[p.name] = p


# --- counts over SUBJECT's own transfers --------------------------------------------------------------------
_add(Prim("n_counterparties", "count", "account", {"dir": _DIR, "window": _W_ANY}, 1,
          "Number of distinct accounts SUBJECT received from (in), sent to (out) or either (any) in the window.",
          "n_counterparties(ALERT.receiver, dir=in, window=before(72))", family_keys=("dir",)))
_add(Prim("n_banks", "count", "account", {"dir": _DIR, "window": _W_ANY}, 1,
          "Number of distinct banks SUBJECT's counterparties belong to in the window.",
          "n_banks(ALERT.receiver, dir=out, window=after(24))", family_keys=("dir",)))
_add(Prim("n_tx", "count", "account",
          {"dir": _DIR, "window": _W_ANY, "min_amount": P("float", 0, _NO_DEFAULT_AMOUNT, default=0.0),
           "max_amount": P("float", 0, _NO_DEFAULT_AMOUNT, default=_NO_DEFAULT_AMOUNT)}, 1,
          "Number of SUBJECT's transfers in the window with min_amount <= amount < max_amount "
          "(the amount on SUBJECT's side of the transfer).",
          "n_tx(ALERT.receiver, dir=in, window=before(72), max_amount=3000)", family_keys=("dir",)))
_add(Prim("n_round", "count", "account", {"window": _W_ANY, "unit": P("float", 1, 1e6)}, 1,
          "Number of SUBJECT's transfers in the window whose amount is a whole multiple of unit.",
          "n_round(ALERT.receiver, window=around(72), unit=1000)"))
_add(Prim("n_format", "count", "account", {"window": _W_ANY, "format": P("enum", choices=PAYMENT_FORMATS)}, 1,
          "Number of SUBJECT's transfers in the window that used the given payment format.",
          "n_format(ALERT.receiver, window=around(72), format=cash)"))
_add(Prim("n_repeated", "count", "account", {"window": _W_ANY}, 2,
          "Number of SUBJECT's transfers in the window that repeat another of its transfers within 1% in amount.",
          "n_repeated(ALERT.receiver, window=around(72))"))
_add(Prim("n_same_entity", "count", "account", {"window": _W_ANY}, 1,
          "Number of SUBJECT's transfers in the window with another account of SUBJECT's own entity.",
          "n_same_entity(ALERT.receiver, window=around(72))", needs_entities=True))
_add(Prim("n_new_counterparties", "count", "account", {"window": _W_AFTER}, 1,
          "Number of distinct accounts SUBJECT paid in the window that it had never paid up to the alert.",
          "n_new_counterparties(ALERT.receiver, window=after(24))"))
_add(Prim("chain_length", "count", "account", {"window": _W_AFTER, "max_hops": P("int", 1, 6, default=3)}, 6,
          "Longest chain of consecutive transfers (each not earlier than the previous) that starts at SUBJECT and "
          "stays among accounts of the asked bank, within the window; capped at max_hops.",
          "chain_length(ALERT.receiver, window=after(72), max_hops=3)",
          extra={"sens_param": "max_hops"}))
# --- behaviour yes/no -----------------------------------------------------------------------------------------
_add(Prim("passed_on", "bool", "account", {"window": _W_AFTER, "min_frac": P("float", 0.05, 1.0)}, 1,
          "Did SUBJECT send out, within the window after the alert, at least min_frac of what it received in the "
          "72 h up to the alert?",
          "passed_on(ALERT.receiver, window=after(24), min_frac=0.8)"))
_add(Prim("returns_home", "bool", "account", {"window": _W_AFTER}, 1,
          "Did SUBJECT send money to an account at the alert's home bank in the window?",
          "returns_home(ALERT.receiver, window=after(72))"))
_add(Prim("matching_amount_out", "bool", "account", {"tol": P("float", 0.0, 0.5)}, 1,
          "Did SUBJECT send out, within 24 h after the alert, a single amount within tol (a fraction) of the "
          "alerted amount?",
          "matching_amount_out(ALERT.receiver, tol=0.05)"))
_add(Prim("velocity_balanced", "bool", "account", {"window": _W_AROUND}, 1,
          "Over the window around the alert, were SUBJECT's total outflows between 90% and 110% of its total "
          "inflows, with inflows at least the alerted amount?",
          "velocity_balanced(ALERT.receiver, window=around(144))"))
_add(Prim("cross_currency_out", "bool", "account", {"window": _W_AFTER}, 1,
          "Did SUBJECT send money on in a currency different from the alerted payment's, in the window?",
          "cross_currency_out(ALERT.receiver, window=after(72))"))
_add(Prim("dormant", "bool", "account", {"days": P("int", 1, 30)}, 1,
          "Did SUBJECT have no transfers at all in the given number of days before the 72 h window up to the alert?",
          "dormant(ALERT.receiver, days=7)"))
_add(Prim("new_account", "bool", "account", {"days": P("int", 1, 30)}, 1,
          "Was SUBJECT's first-ever transfer at the asked bank within the given number of days before the alert?",
          "new_account(ALERT.receiver, days=5)"))
_add(Prim("volume_spike", "bool", "account", {"mult": P("float", 1.5, 100)}, 1,
          "Was SUBJECT's total inflow in the 24 h up to the alert at least mult times its average daily inflow "
          "over the 7 days before that?",
          "volume_spike(ALERT.receiver, mult=5)"))
_add(Prim("payees_pay_each_other", "bool", "account", {"window": _W_AFTER}, 1,
          "Did at least two of the accounts SUBJECT paid in the window also pay each other in the window?",
          "payees_pay_each_other(ALERT.receiver, window=after(72))"))
# --- routing (k-ary) --------------------------------------------------------------------------------------------
_add(Prim("top_outflow_bank", "kary", "account", {"banks": P("banks", 2, MAX_BANKS), "window": _W_ANY}, 1,
          "Which of the listed banks received the largest amount of SUBJECT's outflow in the window? The answer is "
          "the position in the sorted bank list, or the list length for none.",
          "top_outflow_bank(ALERT.receiver, banks=[70, 1588, 220], window=after(24))"))
_add(Prim("time_to_forward", "kary", "account", {}, 1,
          "How soon after the alert did SUBJECT send on at least half of the alerted amount: 0 = under 1 h, "
          "1 = 1-6 h, 2 = 6-24 h, 3 = later or never.",
          "time_to_forward(ALERT.receiver)", arity=4))
_add(Prim("inflow_bucket", "kary", "account", {}, 1,
          "SUBJECT's total inflow in the 72 h up to the alert relative to the alerted amount: 0 = under 1x, "
          "1 = 1-3x, 2 = 3-10x, 3 = over 10x.",
          "inflow_bucket(ALERT.receiver)", arity=4))
# --- set questions (DP-PSI style) -------------------------------------------------------------------------------
_add(Prim("overlap", "set_count", "account",
          {"dir": _DIR_IO, "window": _W_ANY, "among": P("set_slot")}, 1,
          "How many of SUBJECT's distinct counterparties in the window (received from = in, sent to = out) are in "
          "the given alert set?",
          "overlap(ALERT.receiver, dir=out, window=after(72), among=ALERT.payers)", family_keys=("dir",)))
_add(Prim("set_contacts", "set_count", "set", {"dir": _DIR_IO, "window": _W_ANY}, 1,
          "How many accounts in the SUBJECT set received from (in) or sent to (out) an account at the asked bank "
          "in the window?",
          "set_contacts(ALERT.payers, dir=out, window=before(72))", family_keys=("dir",)))
# --- wrapper ----------------------------------------------------------------------------------------------------
_add(Prim("at_least", "bool", "count", {"k": P("int", 1, 1000)}, 1,
          "Is the answer of the count question given as first argument at least k? (A count or set-count question "
          "turned into yes/no.)",
          "at_least(n_counterparties(ALERT.receiver, dir=in, window=before(72)), k=5)"))
# --- registered but not askable ------------------------------------------------------------------------------------
_add(Prim("entity_accounts", "count", "account", {}, 0,
          "Number of accounts SUBJECT's entity holds at the asked bank. A property of the customer record, not of "
          "transactions, so transaction-level DP does not cover it: NOT askable until T2 and T4 decide.",
          "entity_accounts(ALERT.receiver)", restricted=True, needs_entities=True))
