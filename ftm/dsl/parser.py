"""Parser, type checker, canonical form, size, sensitivity and family of DSL questions (C4).

Owner: T2 (Bhavya).  `parse(text)` returns a typed, immutable `Node` or raises `DslError` with a message the planner
can act on. `check(text)` is the non-raising form used by the planner's validator and the parse-rate metric.

Canonical form: primitives in spec order, parameters in spec order, parameters equal to their default omitted,
numbers as %g, bank lists sorted and de-duplicated, one space after each comma. `parse(canonical(p)) == p`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from lark import Lark, Transformer
from lark.exceptions import LarkError, VisitError

from ftm.dsl import GRAMMAR_PATH
from ftm.dsl.spec import (
    ACCOUNT_SLOTS,
    MAX_WINDOW_H,
    PRIMS,
    REQUIRED,
    SET_SLOTS,
    WINDOW_MODES,
    P,
    Prim,
    Window,
)
from ftm.types import AnswerKind


class DslError(ValueError):
    """The program is not a valid question. The message says what to change."""


# ---------------------------------------------------------------------------------------------------------------
# Raw tree
@dataclass(frozen=True)
class Slot:
    name: str


@dataclass
class RawCall:
    name: str
    pos: list
    kw: dict


class _ToRaw(Transformer):
    def number(self, t):
        s = str(t[0])
        return int(s) if s.lstrip("-").isdigit() else float(s)

    def ident(self, t):
        return str(t[0])

    def slot(self, t):
        return Slot(str(t[0]))

    def list(self, items):
        return [i for i in items if i is not None]

    def kwarg(self, t):
        return ("kw", str(t[0]), t[1])

    def expr(self, t):
        name, *args = t
        args = [a for a in args if a is not None]
        kw = {}
        for a in args:
            if isinstance(a, tuple) and len(a) == 3 and a[0] == "kw":
                if a[1] in kw:
                    raise DslError(f"{name}: parameter {a[1]!r} given twice")
                kw[a[1]] = a[2]
        pos = [a for a in args if not (isinstance(a, tuple) and len(a) == 3 and a[0] == "kw")]
        return RawCall(str(name), pos, kw)

    def start(self, t):
        return t[0]


@lru_cache(maxsize=1)
def _parser() -> Lark:
    return Lark(GRAMMAR_PATH.read_text(encoding="utf-8"), parser="lalr")


# ---------------------------------------------------------------------------------------------------------------
# Typed tree
@dataclass(frozen=True)
class Node:
    """A type-checked question. `subject` is a slot name ("receiver", "payers", ...) or, for `at_least`, the
    wrapped Node. `args` holds every parameter (defaults filled) in spec order."""

    name: str
    subject: Any
    args: tuple[tuple[str, Any], ...]

    @property
    def prim(self) -> Prim:
        return PRIMS[self.name]

    def arg(self, key: str) -> Any:
        return dict(self.args)[key]


def parse(text: str, allow_restricted: bool = False) -> Node:
    if not isinstance(text, str) or not text.strip():
        raise DslError("empty program")
    if len(text) > 2000:
        raise DslError("program too long")
    try:
        raw = _ToRaw().transform(_parser().parse(text.strip()))
    except VisitError as e:  # a DslError raised while building the tree
        if isinstance(e.orig_exc, DslError):
            raise e.orig_exc from None
        raise
    except LarkError as e:
        raise DslError(f"syntax error: {str(e).splitlines()[0][:160]}") from None
    return _check_call(raw, allow_restricted)


def _check_call(raw: Any, allow_restricted: bool) -> Node:
    if not isinstance(raw, RawCall):
        raise DslError("a question is a function call, e.g. n_counterparties(ALERT.receiver, dir=in, window=before(72))")
    prim = PRIMS.get(raw.name)
    if prim is None:
        raise DslError(f"unknown function {raw.name!r}; use one of {sorted(n for n, p in PRIMS.items() if not p.restricted)}")
    if prim.restricted and not allow_restricted:
        raise DslError(f"{raw.name} is not askable (it needs a privacy decision)")
    if len(raw.pos) != 1:
        raise DslError(f"{raw.name} takes exactly one subject as its first argument")
    subject = _check_subject(prim, raw.pos[0], allow_restricted)
    unknown = set(raw.kw) - set(prim.params)
    missing = {k for k, p in prim.params.items() if p.default is REQUIRED} - set(raw.kw)
    if unknown or missing:
        raise DslError(f"{raw.name} parameters: missing {sorted(missing)}, unknown {sorted(unknown)}; "
                       f"it takes {list(prim.params)}")
    args = []
    for key, p in prim.params.items():
        value = raw.kw[key] if key in raw.kw else p.default
        args.append((key, _check_param(prim.name, key, p, value)))
    return Node(prim.name, subject, tuple(args))


def _check_subject(prim: Prim, s: Any, allow_restricted: bool) -> Any:
    if prim.subject == "count":
        inner = _check_call(s, allow_restricted)
        if inner.prim.kind not in ("count", "set_count"):
            raise DslError(f"{prim.name} wraps a count or set_count question, not a {inner.prim.kind} one")
        return inner
    slots, what = (ACCOUNT_SLOTS, "account") if prim.subject == "account" else (SET_SLOTS, "set")
    if not isinstance(s, Slot) or s.name not in slots:
        raise DslError(f"{prim.name} needs a {what} slot first: one of {['ALERT.' + k for k in slots]}")
    return s.name


def _num(name: str, key: str, v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise DslError(f"{name}: {key} must be a number")
    return v


def _check_param(name: str, key: str, p: P, v: Any) -> Any:
    if p.kind == "int":
        v = _num(name, key, v)
        if isinstance(v, float):
            if not v.is_integer():
                raise DslError(f"{name}: {key} must be an integer")
            v = int(v)
        _range(name, key, p, v)
        return v
    if p.kind == "float":
        v = float(_num(name, key, v))
        _range(name, key, p, v)
        return v
    if p.kind == "enum":
        if not isinstance(v, str) or v not in p.choices:
            raise DslError(f"{name}: {key} must be one of {list(p.choices)}")
        return v
    if p.kind == "set_slot":
        if not isinstance(v, Slot) or v.name not in SET_SLOTS:
            raise DslError(f"{name}: {key} must be one of {['ALERT.' + k for k in SET_SLOTS]}")
        return v.name
    if p.kind == "banks":
        if not isinstance(v, list) or not all(isinstance(b, int) and not isinstance(b, bool) for b in v):
            raise DslError(f"{name}: {key} must be a list of integer bank ids, e.g. [70, 220]")
        uniq = tuple(sorted(set(v)))
        if not (p.lo <= len(uniq) <= p.hi):
            raise DslError(f"{name}: {key} needs {int(p.lo)}-{int(p.hi)} distinct banks")
        return uniq
    if p.kind == "window":
        if not isinstance(v, RawCall) or v.name not in WINDOW_MODES or v.kw or len(v.pos) != 1:
            raise DslError(f"{name}: {key} must be one of {[m + '(h)' for m in p.choices]}, e.g. {p.choices[0]}(72)")
        if v.name not in p.choices:
            raise DslError(f"{name}: {key} must be {' or '.join(m + '(h)' for m in p.choices)}")
        h = _num(name, key, v.pos[0])
        if isinstance(h, float) and not h.is_integer():
            raise DslError(f"{name}: window hours must be a whole number")
        h = int(h)
        if not (1 <= h <= MAX_WINDOW_H):
            raise DslError(f"{name}: window hours must be 1..{MAX_WINDOW_H}")
        return Window(v.name, h)
    raise AssertionError(p.kind)


def _range(name: str, key: str, p: P, v: float) -> None:
    if not (p.lo <= v <= p.hi):
        raise DslError(f"{name}: {key}={v:g} out of range {p.lo:g}..{p.hi:g}")


# ---------------------------------------------------------------------------------------------------------------
# Derived properties
def _fmt_value(v: Any) -> str:
    if isinstance(v, Window):
        return str(v)
    if isinstance(v, tuple):
        return "[" + ", ".join(str(b) for b in v) + "]"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def canonical(node: Node) -> str:
    """Stable text form used for dedup and as the back-test cache key."""
    prim = node.prim
    subj = canonical(node.subject) if isinstance(node.subject, Node) else f"ALERT.{node.subject}"
    parts = [subj]
    for key, v in node.args:
        if v == prim.params[key].default:
            continue
        parts.append(f"{key}={'ALERT.' + v if prim.params[key].kind == 'set_slot' else _fmt_value(v)}")
    return f"{node.name}({', '.join(parts)})"


def kind(node: Node) -> AnswerKind:
    return node.prim.kind


def arity(node: Node) -> int | None:
    """Number of possible answer values for a k-ary question (None for the other kinds)."""
    if node.prim.kind != "kary":
        return None
    if node.prim.arity is not None:
        return node.prim.arity
    return len(node.arg("banks")) + 1


def sensitivity(node: Node) -> int:
    """Max change of the exact answer when one transaction is added or removed (the DP protection unit)."""
    sp = node.prim.extra.get("sens_param")
    return int(node.arg(sp)) if sp else node.prim.sensitivity


def size(node: Node) -> int:
    """Program size for the enumerator: one per call plus one per parameter that differs from its default."""
    n = 1 + sum(1 for k, v in node.args if v != node.prim.params[k].default)
    return n + (size(node.subject) if isinstance(node.subject, Node) else 0)


def family(node: Node) -> str:
    """Question family (C7 back-test grouping, C8 per-family weights): primitive plus its family keys."""
    if isinstance(node.subject, Node):
        return f"{node.name}:{family(node.subject)}"
    keys = [str(node.arg(k)) for k in node.prim.family_keys]
    return node.name + (":" + ",".join(keys) if keys else "")


@dataclass(frozen=True)
class Check:
    ok: bool
    canonical: str | None = None
    kind: AnswerKind | None = None
    family: str | None = None
    sensitivity: int | None = None
    size: int | None = None
    error: str | None = None


def check(text: str, allow_restricted: bool = False) -> Check:
    """Non-raising `parse`: what the planner's validator and the parse-rate metric call."""
    try:
        n = parse(text, allow_restricted)
    except DslError as e:
        return Check(False, error=str(e))
    return Check(True, canonical(n), kind(n), family(n), sensitivity(n), size(n))


def reference_text() -> str:
    """The language description shown to the LLM, generated from the registry so it cannot drift."""
    from ftm.dsl.spec import WINDOW_MODE_DOC

    lines = [
        "QUESTION LANGUAGE (one call per question; the partner bank computes the exact answer on its own ledger and "
        "returns ONE noisy value):",
        "  name(SUBJECT, key=value, ...)",
        "Account slots: " + "; ".join(f"ALERT.{k} = {v}" for k, v in ACCOUNT_SLOTS.items()),
        "Set slots: " + "; ".join(f"ALERT.{k} = {v}" for k, v in SET_SLOTS.items()),
        "Windows: " + "; ".join(f"{m}(h) = {d}" for m, d in WINDOW_MODE_DOC.items()) + f"; h is 1..{MAX_WINDOW_H}.",
        "Never write account aliases, dates or amounts of specific transfers; only ALERT.<slot>.",
        "Functions (answer kind in brackets):",
    ]
    for name, p in PRIMS.items():
        if p.restricted:
            continue
        ps = ", ".join(_param_doc(k, q) for k, q in p.params.items())
        lines.append(f"- {name} [{p.kind}] ({ps}): {p.doc}\n    e.g. {p.example}")
    return "\n".join(lines)


def _param_doc(key: str, p: P) -> str:
    d = "" if p.default is REQUIRED else f", default {_fmt_value(p.default)}"
    if p.kind == "enum":
        return f"{key} in {{{'|'.join(p.choices)}}}{d}"
    if p.kind == "window":
        return f"{key}={'|'.join(m + '(h)' for m in p.choices)}"
    if p.kind == "banks":
        return f"{key}=[{int(p.lo)}-{int(p.hi)} bank ids]"
    if p.kind == "set_slot":
        return f"{key}=ALERT.<set slot>"
    return f"{key} {p.kind} {p.lo:g}..{p.hi:g}{d}"
