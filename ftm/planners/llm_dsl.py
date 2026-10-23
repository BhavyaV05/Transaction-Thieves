"""The planner's view of the question language: the `DslAdapter` interface and its implementation over T2's DSL v0.

Owner: T3 (Akshat).  Language owner: T2 (ftm/dsl; spec ftm/dsl/spec.py, docs ftm/dsl/README.md).
Spec: tracks.md 4.3 (C4) and 5.4; hand-off #3 (DSL grammar v0, T2 -> T3).

`V0Dsl` is a thin wrapper: `validate` is `ftm.dsl.parser.check`, `reference_text` is T2's registry-generated
language description. The canonical form it returns is T2's and keys the back-test cache (with the eps level).
`DslCheck` is T2's `Check` (ok, canonical, kind, family, sensitivity, size, error). Askable primitives only:
restricted ones (`entity_accounts`) are rejected until T2 and T4 allow them.
"""

from __future__ import annotations

import difflib
import re
from typing import Protocol

from ftm.dsl.parser import Check as DslCheck
from ftm.dsl.parser import check, reference_text

__all__ = ["DslAdapter", "DslCheck", "V0Dsl"]


class DslAdapter(Protocol):
    def validate(self, program: str) -> DslCheck: ...

    def reference_text(self) -> str:
        """The language description shown to the LLM (grammar, slots, functions, one example each)."""
        ...


class V0Dsl:
    """DslAdapter over ftm.dsl (grammar v0). Stateless; the default everywhere a planner needs a DSL."""

    def validate(self, program: str) -> DslCheck:
        return check(program)

    def reference_text(self) -> str:
        return reference_text()


# -- retry feedback helpers (shared by the planner and the Group D ports) --------------------------------------
_FN_LINE = re.compile(r"^- (\w+) \[")
_EG_LINE = re.compile(r"^\s+e\.g\. (.+)$")
_IDENT = re.compile(r"\w+")


def examples_by_function(reference: str) -> dict[str, str]:
    """function name -> its one example, read from the language description (`reference_text`), so nothing about
    the language is duplicated here."""
    out: dict[str, str] = {}
    current = None
    for line in reference.splitlines():
        m = _FN_LINE.match(line)
        if m:
            current = m.group(1)
            continue
        m = _EG_LINE.match(line)
        if m and current and current not in out:
            out[current] = m.group(1).strip()
    return out


def retry_hint(program: str, error: str | None, reference: str, limit: int = 3) -> str:
    """Feedback for a rejected question: the error plus a valid example of the function the model was trying to use
    (or the closest function names with their examples when it invented one)."""
    ex = examples_by_function(reference)
    m = _IDENT.match((program or "").strip())
    name = m.group(0) if m else ""
    if name in ex:
        return f"Write a full call with a slot and its parameters, like {ex[name]}"
    close = difflib.get_close_matches(name, list(ex), n=limit, cutoff=0.4) if name else []
    if not close:
        close = list(ex)[:limit]
    return "There is no such function. Valid functions have the form name(ALERT.<slot>, key=value), for example " + "; ".join(
        ex[c] for c in close
    )
