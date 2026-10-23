"""Build (and optionally freeze) the strategy library from planner / teacher runs on train + val (C12, week 5).

Owner: T3 (Akshat).  Spec: plan.md 6.13, tracks.md 5.5.

    python scripts/build_library.py --runs outputs/runs/<id1> outputs/runs/<id2> --data hi_small \
        --backtest outputs/backtest/lr_table.jsonl --out outputs/library/shape_v1.json --freeze

Inputs: each run dir's runlog.parquet and alerts.jsonl (written by run_llm_agent.py), the alert labels from
data/alerts/<ds>/{train,val}.parquet (server side), and the back-test table as JSONL rows of LRTable.from_rows
(placeholder format until T2's hand-off #9). Test runs are refused. Prints the library hash to put in prereg/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import polars as pl  # noqa: E402

from ftm.library.build import build_library, records_from_runs  # noqa: E402
from ftm.llm.investigation import LRTable  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--backtest", type=Path, required=True)
    ap.add_argument("--keying", choices=["shape", "account"], default="shape")
    ap.add_argument("--n-min", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--freeze", action="store_true")
    a = ap.parse_args(argv)

    labels = {}
    for split in ("train", "val"):
        p = ROOT / "data" / "alerts" / a.data / f"{split}.parquet"
        if p.exists():
            df = pl.read_parquet(p, columns=["alert_id", "label"])  # time-guard: not transactions
            labels.update(dict(zip(df["alert_id"].to_list(), df["label"].to_list(), strict=True)))
    alerts, steps = {}, []
    for run in a.runs:
        for line in (run / "alerts.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["split"] == "test":
                raise SystemExit(f"{run} has test alerts; the library is built on train + val only")
            alerts[r["alert_id"]] = {**r, "label": labels.get(r["alert_id"])}
        log = pl.read_parquet(run / "runlog.parquet")  # time-guard: not transactions
        steps += log.filter(pl.col("event") == "step").to_dicts()
    table = LRTable.from_rows(json.loads(x) for x in a.backtest.read_text(encoding="utf-8").splitlines() if x.strip())

    def accepted(canonical, eps):
        e = table.get(canonical, eps)
        return e is not None and e.accepted

    recs = records_from_runs(steps, alerts, accepted)
    lib = build_library(recs, keying=a.keying, n_min=a.n_min)
    h = lib.freeze() if a.freeze else lib.content_hash()
    lib.save(a.out)
    kinds = [s.kind for s in lib.strategies]
    print(json.dumps({"out": str(a.out), "hash": h, "frozen": lib.frozen, "records": len(recs),
                      "successes": kinds.count("success"), "failures": kinds.count("failure")}, indent=2))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
