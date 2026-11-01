"""Plain-English gloss of a DSL question, and why a question goes to the bank it goes to.

Owner: T3 (Akshat).
"""

from __future__ import annotations

from ftm.dsl.parser import DslError, Node, parse
from ftm.dsl.spec import ACCOUNT_SLOTS
from ftm.llm.routing import NO_BANK, expected_bank, subject_slot

SLOT_TEXT = {"sender": "the sender account", "receiver": "the receiver account",
             "payers": "the accounts that paid into the receiver", "payees": "the accounts the receiver paid"}
DIR_TEXT = {"in": "incoming", "out": "outgoing", "any": "incoming or outgoing"}
WINDOW_TEXT = {"before": "the {h} h up to the alert", "after": "the {h} h after the alert", "around": "{h} h around the alert ({half} h each side)"}


def _window(w) -> str:
    mode, h = getattr(w, "mode", None), getattr(w, "hours", None)
    if mode not in WINDOW_TEXT:
        return str(w)
    return WINDOW_TEXT[mode].format(h=h, half=(h or 0) / 2)


def gloss(program: str) -> str:
    """One or two sentences a non-expert can read. Falls back to the program itself if it does not parse."""
    try:
        node = parse(program, allow_restricted=True)
    except DslError:
        return "(not a valid question)"
    return _gloss(node)


def _gloss(node: Node) -> str:
    args = dict(node.args)
    if node.name == "at_least":
        k = args.get("k")
        inner = node.subject if isinstance(node.subject, Node) else None
        return f"Is the answer to this at least {k:g}? " + (_gloss(inner) if inner else "")
    subject = SLOT_TEXT.get(str(node.subject), str(node.subject))
    text = node.prim.doc.replace("SUBJECT's", f"{subject}'s").replace("SUBJECT", subject)
    extra = []
    if "window" in args:
        extra.append("window: " + _window(args["window"]))
    if "dir" in args:
        extra.append("direction: " + DIR_TEXT.get(str(args["dir"]), str(args["dir"])))
    default = {k: v.default for k, v in node.prim.params.items() if hasattr(v, "default")}
    for k, v in args.items():
        if k not in ("window", "dir") and not (k in default and default[k] == v and v is not None):
            extra.append(f"{k}: {v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"{k}: {v}")
    return text + (f" [{'; '.join(extra)}]" if extra else "")


def bank_why(program: str, summary: dict, home_bank, bank, requested=None) -> str:
    """Why `bank` is the one asked (ftm.llm.routing): the bank that holds the question's subject."""
    slot = subject_slot(program)
    if slot is None:
        return "the question's subject could not be read"
    eb = expected_bank(program, summary, home_bank)
    if eb is None:
        return f"the alert has no bank fields, so the requested bank {bank} is kept"
    if eb == NO_BANK:
        return f"ALERT.{slot} is an account at the home bank {home_bank}: no partner holds it (this question cannot be routed)"
    if slot in ACCOUNT_SLOTS:
        why = f"ALERT.{slot} sits at bank {eb}, and only the bank that holds the subject was back-tested"
    else:
        why = f"{SLOT_TEXT.get(slot, slot)} are at the partner on the other side of the transfer (bank {eb})"
    if requested is not None and str(requested) != str(bank):
        why += f"; the planner named bank {requested}, the pipeline re-routed it"
    return why
