"""The planner's view of the question language: the `DslAdapter` interface and its implementation over T2's DSL v0.

Owner: T3 (Akshat).  Language owner: T2 (ftm/dsl; spec ftm/dsl/spec.py, docs ftm/dsl/README.md).
Spec: tracks.md 4.3 (C4) and 5.4; hand-off #3 (DSL grammar v0, T2 -> T3).

`V0Dsl` is a thin wrapper: `validate` is `ftm.dsl.parser.check`, `reference_text` is T2's registry-generated
language description. The canonical form it returns is T2's and keys the back-test cache (with the eps level).
`DslCheck` is T2's `Check` (ok, canonical, kind, family, sensitivity, size, error). Askable primitives only:
restricted ones (`entity_accounts`) are rejected until T2 and T4 allow them.
"""

from __future__ import annotations

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
