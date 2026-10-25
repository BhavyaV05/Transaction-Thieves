"""Consistency checks for the report skeleton in docs/report/ (T4).

(a) every [n] citation in 01_introduction.md and 02_related_work.md has an entry in references.md;
(b) every {{TBD:id}} placeholder in docs/report/*.md is a row of the numbers table in 00_outline.md;
(c) every CHECK marker in docs/report/*.md is listed in the "Open checks" table of README.md, one row per marker.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1] / "docs" / "report"
CITING = ["01_introduction.md", "02_related_work.md"]

# [12], [7-9], [3, 6], [27], [41]  (not [CHECK ...] and not [text](link))
CITE_RE = re.compile(r"\[(\d+(?:\s*(?:,|-|–)\s*\d+)*)\](?!\()")
PLACEHOLDER_RE = re.compile(r"\{\{TBD:([a-z0-9_]+)\}\}")
CHECK = "[CHECK"


def _read(name: str) -> str:
    return (REPORT / name).read_text(encoding="utf-8")


def _expand(spec: str) -> set[int]:
    out: set[int] = set()
    for part in re.split(r"\s*,\s*", spec.strip()):
        m = re.fullmatch(r"(\d+)\s*(?:-|–)\s*(\d+)", part)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            assert lo <= hi, f"bad citation range {part!r}"
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return out


def _reference_numbers() -> list[int]:
    nums = []
    for line in _read("references.md").splitlines():
        m = re.match(r"^\[(\d+)\]\s", line)
        if m:
            nums.append(int(m.group(1)))
    return nums


def _report_files() -> list[Path]:
    return sorted(REPORT.glob("*.md"))


def test_report_files_exist():
    for name in [
        "00_outline.md",
        "01_introduction.md",
        "02_related_work.md",
        "references.md",
        "README.md",
    ]:
        assert (REPORT / name).is_file(), name


def test_references_are_numbered_without_gaps_or_repeats():
    nums = _reference_numbers()
    assert nums, "no numbered entries found in references.md"
    assert len(nums) == len(set(nums)), "a reference number is used twice"
    assert nums == sorted(nums)
    assert nums == list(range(1, max(nums) + 1)), "reference numbers have a gap"
    # [1]-[41] are the proposal's numbering and must not move
    assert max(nums) >= 41


def test_every_citation_has_a_reference():
    known = set(_reference_numbers())
    problems = []
    for name in CITING:
        text = _read(name)
        used = set()
        for m in CITE_RE.finditer(text):
            used |= _expand(m.group(1))
        assert used, f"{name} cites nothing"
        missing = sorted(used - known)
        if missing:
            problems.append(f"{name}: {missing}")
    assert not problems, "citations without a references.md entry: " + "; ".join(problems)


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
    rows = [
        line
        for line in _read("00_outline.md").splitlines()
        if line.startswith("| `{{TBD:")
    ]
    assert len(rows) == len(ids), "an id appears twice in the numbers table"


def test_every_placeholder_is_in_the_numbers_table():
    ids = _numbers_table_ids()
    unknown = {}
    for path in _report_files():
        for m in PLACEHOLDER_RE.finditer(path.read_text(encoding="utf-8")):
            if m.group(1) not in ids:
                unknown.setdefault(path.name, set()).add(m.group(1))
    assert not unknown, f"placeholders missing from the numbers table of 00_outline.md: {unknown}"


def _open_checks_rows() -> Counter:
    text = _read("README.md")
    assert "## Open checks" in text, "README.md needs an 'Open checks' section"
    section = text[text.index("## Open checks") :]
    nxt = re.search(r"\n## (?!Open checks)", section)
    if nxt:
        section = section[: nxt.start()]
    rows: Counter = Counter()
    for line in section.splitlines():
        m = re.match(r"^\|\s*([\w.]+\.md)\s*\|", line)
        if m:
            rows[m.group(1)] += 1
    return rows


def test_every_check_marker_is_listed_in_readme():
    listed = _open_checks_rows()
    found: Counter = Counter()
    for path in _report_files():
        if path.name == "README.md":
            continue
        n = path.read_text(encoding="utf-8").count(CHECK)
        if n:
            found[path.name] = n
    unlisted = {f: n for f, n in found.items() if listed.get(f, 0) != n}
    stale = {f: n for f, n in listed.items() if f not in found}
    assert not unlisted, (
        "CHECK markers per file differ from the rows in README.md 'Open checks' "
        f"(file: markers in file, rows in README): "
        f"{ {f: (found[f], listed.get(f, 0)) for f in unlisted} }"
    )
    assert not stale, f"README.md lists checks for files that have none: {stale}"
