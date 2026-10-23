"""The planner's view of the question language: an adapter interface plus a PROVISIONAL stand-in grammar.

Owner: T3 (Akshat).  Spec: tracks.md 4.3 (C4, owner T2) and 5.4; hand-off #3 (DSL grammar v0, T2 -> T3).

PROVISIONAL. DSL grammar v0 (hand-off #3) has not arrived yet, so this file holds a small stand-in with the
three question types of plan.md 6.5 and slots relative to the alert. Its only job is to let the planner,
its JSON schema and the parse-rate metric be built and tested now. When ftm/dsl ships, write a `DslAdapter`
over it (validate -> T2's parser + type checker + canonical form; reference_text -> T2's grammar doc),
pass it to the planner, and delete `ProvisionalDsl`. Nothing outside ftm/planners/llm*.py and ftm/external/
may import ProvisionalDsl. Partner banks never execute this stand-in; ftm/banks executes T2's DSL only.

Stand-in syntax:  name(SLOT, key=value, ...)      e.g.  passthrough(ALERT.receiver, min_frac=0.8, within_h=24)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from lark import Lark, Token, Transformer
from lark.exceptions import LarkError

from ftm.types import AnswerKind


@dataclass(frozen=True)
class DslCheck:
    ok: bool
    canonical: str | None = None
    kind: AnswerKind | None = None
    family: str | None = None  # template the question belongs to (C7 families, C8 per-family weights)
    error: str | None = None


class DslAdapter(Protocol):
    def validate(self, program: str) -> DslCheck: ...

    def reference_text(self) -> str:
        """The language description shown to the LLM (grammar, slots, functions, one example each)."""
        ...


# ---------------------------------------------------------------------------------------------------------
_GRAMMAR = r"""
    start: NAME "(" [arg ("," arg)*] ")"
    ?arg: kwarg | value
    kwarg: NAME "=" value
    ?value: slot | SIGNED_NUMBER -> number | NAME -> ident | list
    slot: "ALERT" "." NAME
    list: "[" [value ("," value)*] "]"
    NAME: /[A-Za-z_][A-Za-z0-9_]*/
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS
"""

ACCOUNT_SLOTS = {
    "sender": "the account that sent the alerted transaction",
    "receiver": "the account that received the alerted transaction",
}
SET_SLOTS = {
    "payers": "accounts that paid ALERT.receiver in the 72 h before the alert (home bank's view)",
    "payees": "accounts that ALERT.sender paid in the 72 h before the alert (home bank's view)",
}


@dataclass(frozen=True)
class Param:
    type: Literal["int", "float", "enum", "banks", "account"]
    lo: float | None = None
    hi: float | None = None
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class Function:
    kind: AnswerKind
    subject: Literal["account", "set"]
    params: dict[str, Param]
    doc: str
    example: str
    sensitivity: int = 1  # max change of the true answer when one transaction is added/removed (C5)
    defaults: dict = field(default_factory=dict)


_WIN = Param("int", 1, 72)
_DIR = Param("enum", choices=("in", "out"))
FUNCTIONS: dict[str, Function] = {
    "passthrough": Function(
        "bool", "account", {"min_frac": Param("float", 0.1, 1.0), "within_h": _WIN},
        "Of the money SUBJECT received in the 72 h before the alert, did it send on at least min_frac within within_h hours of receiving it?",
        "passthrough(ALERT.receiver, min_frac=0.8, within_h=24)",
    ),
    "fan_in": Function(
        "bool", "account", {"min_senders": Param("int", 2, 50), "window_h": _WIN},
        "Did SUBJECT receive money from at least min_senders distinct accounts in the window_h hours before the alert?",
        "fan_in(ALERT.receiver, min_senders=5, window_h=48)",
    ),
    "fan_out": Function(
        "bool", "account", {"min_receivers": Param("int", 2, 50), "window_h": _WIN},
        "Did SUBJECT send money to at least min_receivers distinct accounts in the window_h hours before the alert?",
        "fan_out(ALERT.sender, min_receivers=5, window_h=48)",
    ),
    "n_counterparties": Function(
        "count", "account", {"direction": _DIR, "window_h": _WIN},
        "How many distinct accounts did SUBJECT receive from (in) or send to (out) in the window_h hours before the alert?",
        "n_counterparties(ALERT.receiver, direction=out, window_h=24)",
    ),
    "n_tx": Function(
        "count", "account", {"direction": _DIR, "window_h": _WIN, "min_amount": Param("float", 0, 1e9)},
        "How many transactions of at least min_amount did SUBJECT receive (in) or send (out) in the window_h hours before the alert?",
        "n_tx(ALERT.sender, direction=in, window_h=72, min_amount=1000)",
    ),
    "returns_to": Function(
        "bool", "account", {"origin": Param("account"), "max_hops": Param("int", 1, 4), "window_h": _WIN},
        "Did money leaving SUBJECT reach origin again within max_hops transfers, all in the window_h hours before the alert (a cycle)?",
        "returns_to(ALERT.receiver, origin=ALERT.sender, max_hops=3, window_h=72)",
    ),
    "top_outflow_bank": Function(
        "kary", "account", {"banks": Param("banks", 2, 5), "window_h": _WIN},
        "Which of these banks (2-5 bank ids) received the largest share of SUBJECT's outflow in the window_h hours before the alert?",
        "top_outflow_bank(ALERT.receiver, banks=[70, 1588, 220], window_h=48)",
    ),
    "shared_counterparties": Function(
        "set_count", "set", {"direction": _DIR, "window_h": _WIN},
        "How many accounts in SUBJECT (a set slot) also received from (in) or sent to (out) an account at the asked bank in the window_h hours before the alert?",
        "shared_counterparties(ALERT.payers, direction=out, window_h=72)",
    ),
}  # fmt: skip


class _ToPy(Transformer):
    def number(self, t):
        v = float(t[0])
        return int(v) if v.is_integer() and "." not in t[0] and "e" not in t[0].lower() else v

    def ident(self, t):
        return str(t[0])

    def slot(self, t):
        return ("slot", str(t[0]))

    def list(self, items):
        return [i for i in items if i is not None]

    def kwarg(self, t):
        return ("kw", str(t[0]), t[1])

    def start(self, t):
        name, *args = t
        return str(name), [a for a in args if a is not None]


def _fmt(v) -> str:
    if isinstance(v, tuple) and v[0] == "slot":
        return f"ALERT.{v[1]}"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


class ProvisionalDsl:
    """Stand-in DslAdapter (see module docstring). Canonical form: sorted kwargs, %g numbers, sorted bank lists."""

    def __init__(self):
        self._parser = Lark(_GRAMMAR, parser="lalr")

    def parse(self, program: str) -> tuple[str, list]:
        return _ToPy().transform(self._parser.parse(program.strip()))

    def validate(self, program: str) -> DslCheck:
        if not isinstance(program, str) or not program.strip():
            return DslCheck(False, error="empty program")
        try:
            name, args = self.parse(program)
        except (LarkError, ValueError) as e:
            return DslCheck(False, error=f"syntax: {str(e).splitlines()[0][:160]}")
        fn = FUNCTIONS.get(name)
        if fn is None:
            return DslCheck(False, error=f"unknown function {name!r}; use one of {sorted(FUNCTIONS)}")
        pos = [a for a in args if not (isinstance(a, tuple) and a[0] == "kw")]
        kws = {a[1]: a[2] for a in args if isinstance(a, tuple) and a[0] == "kw"}
        if len(pos) != 1 or not (isinstance(pos[0], tuple) and pos[0][0] == "slot"):
            return DslCheck(False, error=f"{name} takes exactly one ALERT.<slot> subject first")
        subject = pos[0][1]
        allowed = ACCOUNT_SLOTS if fn.subject == "account" else SET_SLOTS
        if subject not in allowed:
            return DslCheck(False, error=f"{name} needs a {fn.subject} slot: one of {['ALERT.' + s for s in allowed]}")
        missing = set(fn.params) - set(kws)
        extra = set(kws) - set(fn.params)
        if missing or extra:
            return DslCheck(False, error=f"{name} parameters: missing {sorted(missing)}, unknown {sorted(extra)}")
        norm = {}
        for k, p in fn.params.items():
            err, val = _check_param(k, p, kws[k])
            if err:
                return DslCheck(False, error=f"{name}: {err}")
            norm[k] = val
        canon = f"{name}(ALERT.{subject}" + "".join(f", {k}={_fmt(norm[k])}" for k in sorted(norm)) + ")"
        return DslCheck(True, canonical=canon, kind=fn.kind, family=name)

    def reference_text(self) -> str:
        lines = [
            "QUESTION LANGUAGE (one function call per question; the partner bank computes the exact answer on its "
            "own ledger and returns ONE noisy value):",
            "  name(SUBJECT, key=value, ...)",
            "Account slots: " + "; ".join(f"ALERT.{k} = {v}" for k, v in ACCOUNT_SLOTS.items()),
            "Set slots: " + "; ".join(f"ALERT.{k} = {v}" for k, v in SET_SLOTS.items()),
            "Never write account aliases or dates in a question; only ALERT.<slot>.",
            "Functions (answer type in brackets):",
        ]
        for name, fn in FUNCTIONS.items():
            ps = ", ".join(_param_doc(k, p) for k, p in fn.params.items())
            lines.append(f"- {name} [{fn.kind}] ({ps}): {fn.doc}\n    e.g. {fn.example}")
        return "\n".join(lines)


def _param_doc(k: str, p: Param) -> str:
    if p.type == "enum":
        return f"{k} in {{{'|'.join(p.choices)}}}"
    if p.type == "banks":
        return f"{k}=[{int(p.lo)}-{int(p.hi)} bank ids]"
    if p.type == "account":
        return f"{k}=ALERT.<account slot>"
    return f"{k} {p.type} {p.lo:g}..{p.hi:g}"


def _check_param(k: str, p: Param, v) -> tuple[str | None, object]:
    if p.type == "enum":
        return (None, v) if isinstance(v, str) and v in p.choices else (f"{k} must be one of {p.choices}", None)
    if p.type == "account":
        ok = isinstance(v, tuple) and v[0] == "slot" and v[1] in ACCOUNT_SLOTS
        return (None, v) if ok else (f"{k} must be an account slot", None)
    if p.type == "banks":
        if not isinstance(v, list) or not all(isinstance(b, int) and not isinstance(b, bool) for b in v):
            return f"{k} must be a list of integer bank ids", None
        uniq = sorted(set(v))
        if not (p.lo <= len(uniq) <= p.hi):
            return f"{k} needs {int(p.lo)}-{int(p.hi)} distinct banks", None
        return None, uniq
    if isinstance(v, (tuple, list, str, Token)) or isinstance(v, bool):
        return f"{k} must be a number", None
    if p.type == "int" and not isinstance(v, int):
        if float(v).is_integer():
            v = int(v)
        else:
            return f"{k} must be an integer", None
    v = float(v) if p.type == "float" else v
    if not (p.lo <= v <= p.hi):
        return f"{k}={v:g} out of range {p.lo:g}..{p.hi:g}", None
    return None, v
