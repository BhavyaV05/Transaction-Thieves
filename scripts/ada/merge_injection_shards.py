"""Merge E15 shards (run_injection_tests.py runs over disjoint alert lists of the same alert set) into the single run the report expects.  Owner: T3 (Akshat).

    python scripts/ada/merge_injection_shards.py --out outputs/injection_v3 --run-id e15_planner --system planner --shards e15_planner_s1 e15_planner_s2

Reads <out>/<shard>/<system>/{pairs.jsonl, runlog.parquet, summary.json}, checks that the shards used the same arms, kinds, channel, injection set, model, data and val part and have no alert in
common, then writes <out>/<run-id>/<system>/{pairs.jsonl, runlog.parquet, summary.json} and <out>/<run-id>/summary.json in the format of an unsharded run: the rows are `summarize()` of ALL pairs
(a pure function of the pairs, so sharding changes nothing but which runs the harness noise stream sees), and the guard applies again (exit 3 when an arm is NOT TESTED)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ftm.safety.camel_compare import compare_arms  # noqa: E402
from ftm.safety.injection_run import MIN_EXPOSED_SHARE, summarize  # noqa: E402

SAME = ("arms", "kinds", "channel", "injection_set_sha256", "model", "llm", "toolbox", "alerts", "val_part", "data")


def merge(out: Path, run_id: str, system: str, shards: list[str], min_exposed_share: float = MIN_EXPOSED_SHARE) -> dict:
    sums = [json.loads((out / s / system / "summary.json").read_text()) for s in shards]
    for k in SAME:
        vals = {json.dumps(s.get(k), sort_keys=True) for s in sums}
        if len(vals) != 1:
            raise SystemExit(f"shards disagree on {k}: {sorted(vals)}")
    pairs = []
    for s in shards:
        pairs += [json.loads(ln) for ln in (out / s / system / "pairs.jsonl").read_text().splitlines() if ln.strip()]
    seen: dict[tuple, str] = {}
    for s in shards:
        for ln in (out / s / system / "pairs.jsonl").read_text().splitlines():
            p = json.loads(ln)
            key = (p["alert_id"], p["arm"], p["injection_id"])
            if key in seen:
                raise SystemExit(f"alert/arm/item {key} is in both {seen[key]} and {s}: the shards must be disjoint")
            seen[key] = s
    rows = summarize(pairs, min_exposed_share)
    dest = out / run_id / system
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))
    pl.concat([pl.read_parquet(out / s / system / "runlog.parquet") for s in shards], how="vertical_relaxed").write_parquet(dest / "runlog.parquet")
    n_alerts = sum(s["n_alerts"] for s in sums)
    base = sums[0]
    summary = {**base, "run_id": run_id, "n_alerts": n_alerts, "n_pairs": len(pairs), "seconds": sum(s["seconds"] for s in sums), "rows": rows, "camel_compare": compare_arms(rows),
               "merged_from_shards": shards}  # fmt: skip
    (dest / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    top = {"run_id": run_id, "systems": [system], "skipped": [], "arms": base["arms"], "kinds": base["kinds"], "llm": base["llm"], "alerts": base["alerts"], "n_alerts": n_alerts,
           "rows": rows, "camel_compare": compare_arms(rows), "merged_from_shards": shards}  # fmt: skip
    (out / run_id / "summary.json").write_text(json.dumps(top, indent=2, default=str))
    return top


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--min-exposed-share", type=float, default=MIN_EXPOSED_SHARE)
    a = ap.parse_args(argv)
    top = merge(a.out, a.run_id, a.system, a.shards, a.min_exposed_share)
    print(json.dumps({"run_id": a.run_id, "n_alerts": top["n_alerts"], "n_summary_rows": len(top["rows"])}))
    untested = sorted({r["arm"] for r in top["rows"] if r.get("status") == "NOT TESTED"})
    if untested:
        print("NOT TESTED (not a safe result): " + ", ".join(untested), file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
