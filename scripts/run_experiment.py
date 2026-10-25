"""Run one non-LLM system (random, fixed, eig, enumerator) over validation alerts through the harness.

Owner: T2 (Bhavya).  Spec: tracks.md 4.7 and 4.8.  Run from the repo root.

    # offline smoke test on the synthetic world (never a result): back-test on the first 60% of alerts, run on the rest
    python scripts/run_experiment.py --system eig --synthetic 360
    # real: the LR table that `scripts/backtest.py --assemble` wrote, the val alerts of T1's alert set
    python scripts/run_experiment.py --system eig --data hi_small --split val \\
        --table data/backtest/hi_small/<namespace>/lr_table.jsonl --threshold 0.3

Writes outputs/runs/<run_id>/runlog.parquet (ftm.eval.runlog schema) and summary.json (Brier before and after, questions
and eps per alert). The test split is never read.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftm.backtest.world import load_world  # noqa: E402
from ftm.harness.runlog_rows import write_runlog  # noqa: E402
from ftm.orchestrator.runner import (  # noqa: E402
    ALL_SYSTEMS,
    build_planner,
    make_harness,
    run_alerts,
    synthetic_experiment,
)
from ftm.planners.pool import QuestionPool  # noqa: E402
from ftm.types import Limits  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return ""


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=ALL_SYSTEMS, required=True)
    ap.add_argument("--policy", default=None, help="fitted policy for --system learned (scripts/train_policy.py); the run "
                    "then scores the eval half of the validation alerts only, the half the policy never saw")
    ap.add_argument("--synthetic", type=int, default=None, help="n alerts of the synthetic world (smoke test)")
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--table", default=None, help="lr_table.jsonl from scripts/backtest.py --assemble")
    ap.add_argument("--eps", type=float, default=1.0, help="the eps level the fixed-order planner asks at")
    ap.add_argument("--threshold", type=float, default=None, help="decision threshold for the cannot-cross stop")
    ap.add_argument("--max-alerts", type=int, default=None)
    ap.add_argument("--max-size", type=int, default=4, help="enumerator program size (informational)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--machine", default="t2-3050")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "runs"))
    a = ap.parse_args(argv)

    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    out = Path(a.out) / run_id
    if a.synthetic:
        if a.system == "learned":
            raise SystemExit("the learned policy on the synthetic world: scripts/run_ablations.py --synthetic N --only learned_policy")
        res = synthetic_experiment(a.system, a.synthetic, a.seed, limits=Limits())
        variant = "synthetic"
    else:
        if not a.table:
            raise SystemExit("--table is required for real data (run scripts/backtest.py --assemble first)")
        if a.system == "learned" and not a.policy:
            raise SystemExit("--policy is required for --system learned (run scripts/train_policy.py first)")
        world = load_world(a.data, a.split, a.seed, max_alerts=a.max_alerts)
        alerts, policy = world.alerts, None
        if a.system == "learned":
            from ftm.planners.learned import LearnedPolicy, split_val

            policy = LearnedPolicy.load(a.policy)
            alerts = split_val(world.alerts)[1]  # the policy was fitted on the first half of the validation alerts
        pool = QuestionPool.from_jsonl(a.table)
        harness = make_harness(world.banks, {x.alert_id: x.binding for x in alerts})
        planner = build_planner(a.system, pool, a.seed, a.eps, a.threshold, max_size=a.max_size, policy=policy)
        base = {"run_id": run_id, "commit": git_commit(), "machine": a.machine, "system": a.system,
                "variant": "nonllm", "seed": a.seed}  # fmt: skip
        res = run_alerts(planner, alerts, harness, Limits(), base, a.split)
        res.summary["pool"] = len(pool)
        variant = "nonllm"
    for r in res.rows:
        r["run_id"], r["commit"], r["machine"], r["variant"] = run_id, git_commit(), a.machine, variant
    write_runlog(res.rows, out / "runlog.parquet")
    (out / "summary.json").write_text(json.dumps(res.summary, indent=1))
    print(json.dumps(res.summary, indent=1))
    return res.summary


if __name__ == "__main__":
    main()
