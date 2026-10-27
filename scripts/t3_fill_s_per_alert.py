"""Write the validation-run medians (Gemma 4 12B, reference machine) into `llm.s_per_alert` of configs/final/jobs.yaml (T3).

    python scripts/t3_fill_s_per_alert.py [--table outputs/t3_val/table.csv] [--jobs configs/final/jobs.yaml] [--dry-run]

Reads the table written by scripts/val_runs_table.py (median seconds per alert on the 160 val `eval` alerts). `ours` takes the median of
the compose run; a Group C system with two modes takes the slower mode (scheduling should not be optimistic). Only the numbers on the
`s_per_alert` lines change; comments and the rest of the file stay. Estimates for scheduling only (docs/t4_final_run.md).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY = {"debate": "debate_autogen"}


def medians(table: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in csv.DictReader(table.open()):
        spec = r["spec"]
        if spec.startswith("e16_"):
            continue
        if spec == "ours_compose":
            key = "ours"
        elif spec.startswith("ours_"):
            continue
        else:
            key = KEY.get(re.sub(r"_(native|ported)$", "", spec), re.sub(r"_(native|ported)$", "", spec))
        out[key] = max(out.get(key, 0.0), float(r["median_s"]))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, default=ROOT / "outputs" / "t3_val" / "table.csv")
    ap.add_argument("--jobs", type=Path, default=ROOT / "configs" / "final" / "jobs.yaml")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    med = medians(a.table)
    lines = a.jobs.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("s_per_alert:"))
    changed = []
    for i in range(start + 1, len(lines)):
        m = re.match(r"^(\s+)([a-z_0-9]+):\s*([0-9.]+)(\s*#.*)?$", lines[i])
        if not m:
            if lines[i].strip() and not lines[i].startswith(" " * 4):
                break
            continue
        k = m.group(2)
        if k in med:
            lines[i] = f"{m.group(1)}{k}: {med[k]:.1f}{m.group(4) or ''}"
            changed.append(f"{k}: {m.group(3)} -> {med[k]:.1f}")
    if not a.dry_run:
        a.jobs.write_text("\n".join(lines) + "\n")
    print("\n".join(changed) or "nothing changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
