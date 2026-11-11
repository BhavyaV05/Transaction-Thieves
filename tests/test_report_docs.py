"""Consistency check for docs/report/00_outline.md (T4).

The Markdown chapter drafts were replaced by the final LaTeX report (docs/report/final/). 00_outline.md stays because
scripts/final_analysis.py reads the placeholder ids of its "Numbers and figures the report needs" table
(ftm.eval.final_tables.outline_ids). This test checks that the table is there and that no id appears twice.
"""

from __future__ import annotations

import re
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1] / "docs" / "report"
PLACEHOLDER_RE = re.compile(r"\{\{TBD:([a-z0-9_]+)\}\}")


def _read(name: str) -> str:
    return (REPORT / name).read_text(encoding="utf-8")


def _numbers_table_ids() -> set[str]:
    """Ids defined in the 'Numbers and figures the report needs' table of 00_outline.md."""
    text = _read("00_outline.md")
    start = text.index("## Numbers and figures the report needs")
    section = text[start:]
    nxt = re.search(r"\n## (?!Numbers and figures)", section)
    if nxt:
        section = section[: nxt.start()]
    ids = set()
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        first = line.split("|")[1] if line.count("|") >= 2 else ""
        m = PLACEHOLDER_RE.search(first)
        if m:
            ids.add(m.group(1))
    return ids


def test_numbers_table_has_rows_and_unique_ids():
    ids = _numbers_table_ids()
    assert len(ids) >= 20
    rows = [line for line in _read("00_outline.md").splitlines() if line.startswith("| `{{TBD:")]
    assert len(rows) == len(ids), "an id appears twice in the numbers table"
