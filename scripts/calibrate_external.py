"""Fit, check and freeze a probability calibration map for one system; apply a frozen map to run logs (T4, finding 12).

    # fit on the val `fit` part of one or more validation runs, check on `cal`, write the frozen map (JSON with its own sha256)
    python scripts/calibrate_external.py fit --system react --runs outputs/runs/<fit_run> outputs/runs/<cal_run> \\
        --out outputs/calibration/react.json
    # apply a frozen map to an eval / test run log (no label is read; the raw value stays in `probability_raw`)
    python scripts/calibrate_external.py apply --map outputs/calibration/react.json --runlog IN.parquet --out OUT.parquet

`fit` needs validation run logs with labels and `val_part` (a run of `run_llm_agent.py --alerts val --val-part fit|cal`).
The method is `auto` (levels for FinFRE-style score levels, `VerdictMap.fit` for Sharma, Platt on logit(p) otherwise) or
--method platt|levels|sharma_verdict. A map that does not improve the Brier on `cal` is frozen as the identity and says why.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftm.eval import calibrate as cal  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--system", required=True, help="run-log `system` (e.g. react, sharma_ported, finfre_rag_native, ours_compose)")
    f.add_argument("--runs", type=Path, nargs="+", required=True, help="run dirs (or runlog.parquet files) of validation runs")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--method", choices=["auto", "platt", "levels", "sharma_verdict"], default="auto")
    f.add_argument("--fit-parts", nargs="+", default=["fit"])
    f.add_argument("--check-parts", nargs="+", default=["cal"])
    f.add_argument("--strength", type=float, default=4.0, help="levels: pseudo-count toward the level itself")
    f.add_argument("--min-rows", type=int, default=30)
    g = sub.add_parser("apply")
    g.add_argument("--map", type=Path, required=True)
    g.add_argument("--expect-sha", default=None, help="the map's sha256 as recorded (freeze manifest / docs)")
    g.add_argument("--runlog", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    if a.cmd == "fit":
        df = cal.labelled_verdicts(a.runs)
        m = cal.freeze_map(cal.fit_map(df, a.system, a.method, a.fit_parts, a.check_parts, a.strength, a.min_rows), a.out)
        m["sha256"] = cal.load_map(a.out)["sha256"]
        print(cal.describe(m))
        print(f"wrote {a.out}  sha256 {m['sha256']}")
        return 0
    m = cal.load_map(a.map, a.expect_sha)
    out = cal.apply_map_to_frame(pl.read_parquet(a.runlog), m)  # time-guard: not transactions
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(a.out)  # time-guard: not transactions
    print(f"{a.runlog} -> {a.out}: {m['system']} mapped with {m['method']} (map {m['sha256'][:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
