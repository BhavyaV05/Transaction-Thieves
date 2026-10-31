"""T3's inputs for T4's final analysis (`configs/analysis/inputs.yaml`): the E18 sweep CSV and the E10 fast-model JSON (T3, docs/t3_e10.md).

    python scripts/t3_final_inputs.py [--out outputs/t3_final_inputs]

Writes
  <out>/e18_sweep.csv   `e18.summary_csv`: one row per candidate and stage (stage 1: 160 val eval alerts; stage 2: the top three on all 460
                        askable val alerts), columns of outputs/e18/<stage>/summary.csv (candidate, family, params_b, active_b, quant, stack,
                        size_gb, n_alerts, brier, brier_ci_lo, brier_ci_hi, no_verdict_rate, valid_output_rate, ..., median_s, p90_s, ...)
                        with `stage` first. `final_tables.e18_table` reads it as-is into the table `e18_sweep`.
  <out>/fast.json       `fast_model.path`: {"decision_types": [{name, alpha, error_marginal, kept, speedup, brier_delta}], ...} from
                        scripts/e10_report.py (outputs/e10/fast.json), copied after a format check (`final.h5` reads exactly these keys;
                        `speedup` = large latency / cascade mean latency, `brier_delta` null: the fast model is not wired into the planner).
and prints the two lines to put in inputs.yaml. Validation outputs only; nothing here reads the test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def e18_csv(e18_dir: Path, out: Path) -> int:
    rows, cols = [], []
    for stage in ("stage1", "stage2"):
        f = e18_dir / stage / "summary.csv"
        if not f.exists():
            continue
        with f.open(newline="") as fh:
            rd = csv.DictReader(fh)
            cols = cols or list(rd.fieldnames or [])
            rows += [{"stage": stage, **r} for r in rd]
    if not rows:
        raise SystemExit(f"no summary.csv under {e18_dir}/stage1|stage2")
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["stage", *cols], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def check_fast(path: Path) -> dict:
    d = json.loads(path.read_text())
    types = d.get("decision_types")
    if not types:
        raise SystemExit(f"{path}: no decision_types")
    for t in types:
        missing = [k for k in ("name", "alpha", "error_marginal", "kept", "speedup") if k not in t]
        if missing:
            raise SystemExit(f"{path}: decision type {t.get('name')} lacks {missing}")
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e18", type=Path, default=ROOT / "outputs" / "e18")
    ap.add_argument("--fast", type=Path, default=ROOT / "outputs" / "e10" / "fast.json")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "t3_final_inputs")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    n = e18_csv(a.e18, a.out / "e18_sweep.csv")
    print(f"e18_sweep.csv: {n} rows -> {a.out / 'e18_sweep.csv'}")
    if a.fast.exists():
        d = check_fast(a.fast)
        shutil.copy(a.fast, a.out / "fast.json")
        print(f"fast.json: {len(d['decision_types'])} decision types -> {a.out / 'fast.json'}")
    else:
        print(f"fast.json: {a.fast} not found yet (scripts/e10_report.py --json)", file=sys.stderr)
    print(f"inputs.yaml:\n  e18: {{summary_csv: {a.out / 'e18_sweep.csv'}}}\n  fast_model: {{path: {a.out / 'fast.json'}}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
