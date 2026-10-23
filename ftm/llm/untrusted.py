"""Quoting untrusted text (partner-bank notes) before it goes inside an <untrusted_note> wrapper.

Owner: T3 (Akshat).  Spec: plan.md 10 (injection), E15.

A note is truncated, then `& < >` are escaped and newlines collapsed, so a note can neither close the wrapper
(`</untrusted_note>`) nor start a line that looks like the system or a tool speaking. Every LLM system uses this
one helper (planner prompt and the Group D observations), so the wrapper holds the same way for all of them.
"""

from __future__ import annotations

import re

NOTE_MAX_CHARS = 300


def quote_untrusted(text: str | None, max_chars: int = NOTE_MAX_CHARS) -> str:
    """Escaped, single-line, truncated text (truncated first, so the cut cannot split an entity)."""
    s = (text or "")[:max_chars]
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"\s*[\r\n]+\s*", " ", s)


def wrap_untrusted(text: str | None, max_chars: int = NOTE_MAX_CHARS) -> str:
    return f"<untrusted_note>{quote_untrusted(text, max_chars)}</untrusted_note>"
