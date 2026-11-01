"""Which bank a question must be put to (one place for the LLM side).

Owner: T3 (Akshat).  Spec: audit finding 1 (bank routing).

Mirrors `ftm.backtest.engine.asked_bank` (the back-test asks only the bank that holds the question's subject; a
test pins the two together): an account slot (sender / receiver) goes to that party's bank, and has no valid bank
when the party is at the home bank; a set slot (payers / payees) goes to the partner on the other side of the alerted
transfer. Answers from any other bank carry no back-tested LR, so menus, prompts, the planner and fusion all use this.
"""

from __future__ import annotations

from ftm.dsl.parser import Node
from ftm.dsl.parser import parse as dsl_parse
from ftm.dsl.spec import ACCOUNT_SLOTS

NO_BANK = "none"  # expected_bank of a question no partner holds (its subject is a home-bank account)


def subject_slot(program: str) -> str | None:
    """The innermost slot a program is about (`at_least(...)` wraps another call), None if it does not parse."""
    try:
        node = dsl_parse(program, allow_restricted=True)
    except Exception:  # noqa: BLE001 - an unparsable program has no subject
        return None
    while isinstance(node.subject, Node):
        node = node.subject
    return node.subject


def partner_of(summary: dict, home_bank) -> int | str | None:
    """The bank on the other side of the alerted transfer, None if both parties are at the home bank / unknown."""
    cp = summary.get("counterparty_bank")
    if cp is not None:
        return cp
    s, r = summary.get("sender_bank"), summary.get("receiver_bank")
    if s is None or r is None:
        return None
    if str(s) != str(home_bank):
        return s
    return r if str(r) != str(home_bank) else None


def slot_bank(slot: str, summary: dict, home_bank) -> int | str | None:
    """Bank for a slot, None when no partner holds it. Raises KeyError when the summary has no bank fields at all."""
    if slot in ACCOUNT_SLOTS:
        if f"{slot}_bank" not in summary:
            raise KeyError(f"{slot}_bank")
        b = summary[f"{slot}_bank"]
        return None if b is None or str(b) == str(home_bank) else b
    return partner_of(summary, home_bank)


def asked_bank(program: str, summary: dict, home_bank) -> int | str | None:
    """The bank a question must be asked of, or None when no partner holds its subject (or it does not parse)."""
    slot = subject_slot(program)
    if slot is None:
        return None
    try:
        return slot_bank(slot, summary, home_bank)
    except KeyError:
        return None


def expected_bank(program: str, summary: dict, home_bank) -> str | None:
    """What a log entry stores: the bank as str, NO_BANK when nobody holds the subject, None when the summary cannot
    say (no bank fields: routing unknown, fusion then does not filter)."""
    if "sender_bank" not in summary and "counterparty_bank" not in summary:
        return None
    b = asked_bank(program, summary, home_bank)
    return NO_BANK if b is None else str(b)
