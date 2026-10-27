"""Shared evidence trimming for prompts that run over the cap.

Owner: T3 (Akshat).  Spec: tracks.md 1.2 rule 4 (fixed prompt cap), plan.md 7.2.

The prompt-cap estimate (`ftm.llm.budget.CHARS_PER_TOKEN`) is fitted on Qwen tokenizers; Gemma 4 counts about 5% more, so a
prompt the estimate lets through can be over `Limits.max_tokens` by the server's count. Every text system builds its prompt
with the same evidence block ("Evidence (home bank's view, time-ordered, aliased)", `case_text` in `external/text_agent.py`
and `_section` in `planners/llm_prompt.py`). `BudgetedClient` trims that block the way the planner does: drop the oldest rows,
keep the newest, mark the cut, and resend once the server's own count is known (`BudgetedClient.chat`).
"""

from __future__ import annotations

import re

EVIDENCE_HEAD = re.compile(r"(?m)^(?:## )?Evidence \(home bank's view[^\n]*\n")
OMITTED = "(older evidence rows omitted to fit the prompt cap)"


def _block(text: str) -> tuple[int, int] | None:
    """(start, end) of the evidence rows in `text`: after the header line, to the next blank line or the end."""
    m = EVIDENCE_HEAD.search(text)
    if not m:
        return None
    end = text.find("\n\n", m.end())
    return m.end(), len(text) if end < 0 else end


def trim_evidence(messages: list[dict], drop_fraction: float) -> list[dict] | None:
    """Copy of `messages` with the oldest ~`drop_fraction` of the evidence rows removed (at least one row), or None when
    there is no evidence block with more than one row left to drop. The newest rows stay; the cut is marked."""
    for i, m in enumerate(messages):
        content = m.get("content")
        if not isinstance(content, str):
            continue
        span = _block(content)
        if span is None:
            continue
        rows = content[span[0]:span[1]].split("\n")
        if rows and rows[0] == OMITTED:
            rows = rows[1:]
        if len(rows) <= 1:
            continue
        drop = min(len(rows) - 1, max(1, int(len(rows) * drop_fraction + 0.999)))
        kept = [OMITTED, *rows[drop:]]
        return [*messages[:i], {**m, "content": content[:span[0]] + "\n".join(kept) + content[span[1]:]}, *messages[i + 1:]]
    return None
