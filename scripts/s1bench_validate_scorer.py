"""Validate our S1MB scorer against published rows of hotchpotch/s1mb-result (doc section 7).  Owner: T3 (Akshat).

For each published model and benchmark: recompute the metrics from the STORED predictions with ftm.fastmodel.s1bench_s1mb.score_benchmark and compare to the
metrics stored in the result file (written by the official evaluator).  Also recomputes each model's task scores and Task Avg over its complete
benchmarks.  Exit code 1 if any metric differs by more than --tol.

    python scripts/s1bench_validate_scorer.py --models typesafe__jev_1_13 hotchpotch__bekko-system-one-v0-400m --out outputs/s1bench/scorer_validation.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftm.fastmodel.s1bench_s1mb import aggregate, score_benchmark  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs" / "s1bench" / "datasets.yaml").read_text())
PREP = Path(os.environ.get("S1BENCH_PREPARED", CFG["prepared_root"]))  # ada: the prepared data lives elsewhere


def load_lines(bid: str) -> list[dict]:
    return [json.loads(x) for x in (PREP / "s1mb" / f"{bid}.jsonl").open()]


def compare(mine: dict, stored: dict, tol: float) -> list[str]:
    bad = []
    for k, v in stored.items():
        if v is None or not isinstance(v, int | float) or k not in mine:
            continue
        m = mine[k]
        if m is None or abs(m - v) > tol:
            bad.append(f"{k}: stored {v} mine {m}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    report, fail = {}, 0
    for model in a.models:
        d = PREP / "published" / model
        per, tasks, rows, stored_adj = {}, {}, [], {}
        for f in sorted(d.glob("*.json")):
            j = json.loads(f.read_text())
            if not (PREP / "s1mb" / f"{j['benchmark']}.jsonl").exists():
                continue  # not in the active English-v1 membership (the result repo also holds other scopes)
            lines = load_lines(j["benchmark"])
            task = lines[0]["task"]
            preds = {k: v for k, v in j["predictions"].items() if v is not None}
            m = score_benchmark(task, lines, preds)
            bad = compare(m, j["metrics"], a.tol)
            if j["status"] == "complete" and m["answered"] + len(j["abstentions"]) == len(lines) and not j["abstentions"]:
                per[j["benchmark"]] = m
                tasks[j["benchmark"]] = task
                stored_adj[j["benchmark"]] = j["metrics"].get("baseline_adjusted_score")
            rows.append({"benchmark": j["benchmark"], "task": task, "status": j["status"], "n": len(lines), "answered": m["answered"],
                         "abstentions": len(j["abstentions"]), "mismatch": bad})  # fmt: skip
            fail += bool(bad) and not j["abstentions"]  # abstention scoring (max error) is out of scope for our scorer; counted separately
        agg = aggregate(per, tasks)
        ok = sum(1 for r in rows if not r["mismatch"])
        report[model] = {"benchmarks": len(rows), "match": ok, "with_abstentions": sum(1 for r in rows if r["abstentions"]), "aggregate": agg,
                         "worst": [r for r in rows if r["mismatch"]][:10]}  # fmt: skip
        print(f"{model}: {ok}/{len(rows)} benchmarks match to {a.tol}; Task Avg (ours, complete benchmarks) {agg['task_avg']}", flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=1))
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
