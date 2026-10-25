"""Async runs (E13, H8, stretch): sequential vs batched questions for the non-LLM planners, and the queue under load.

Owner: T2 (Bhavya).  Spec: tracks.md 4.9 and 4.11 (E13), plan.md 8.  Run from the repo root.

    # offline smoke test on the synthetic world (never a result)
    python scripts/async_study.py --synthetic 600 --systems random eig --budgets 2 4
    # real: the LR table that `scripts/backtest.py --assemble` wrote; scored on the eval half of the validation alerts
    python scripts/async_study.py --data hi_medium --split val --table <lr_table.jsonl> \\
        --systems random fixed eig learned --policy outputs/policy/hi_medium.json --budgets 2 4 --out outputs/async

For each system and per-alert budget: one question at a time vs waves of (2, 1), the same alerts, pool and budget. Rows carry
Brier, questions, eps, simulated days, the SimPy makespan, and `share_2plus` (the share of alerts that asked two or more
questions at all: a wave can only help where two questions fit the eps left). `h8` rows give the share of the sequential gain
the batched arm keeps and its share of the sequential time (the claim is 80% in 60%). `load` rows put the same alerts on one
SimPy clock at several arrival gaps. T3's `docs/t3_e13.md` has the LLM planner's arm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftm.orchestrator.async_study import batching_grid, load_study  # noqa: E402
from ftm.orchestrator.runner import ALL_SYSTEMS  # noqa: E402
from ftm.planners.learned import split_val  # noqa: E402
from ftm.planners.pool import QuestionPool  # noqa: E402

EPS_LEVELS = (0.25, 0.5, 1.0, 2.0)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="+", choices=ALL_SYSTEMS, default=["random", "fixed", "eig"])
    ap.add_argument("--budgets", type=float, nargs="+", default=[2.0, 4.0], help="eps per alert, both arms")
    ap.add_argument("--waves", type=int, nargs="+", default=[2, 1])
    ap.add_argument("--gaps", type=float, nargs="+", default=[0.5, 0.1, 0.02, 0.005], help="arrival gaps (days) for the load study")
    ap.add_argument("--synthetic", type=int, default=None, help="n alerts of the synthetic world (smoke test)")
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--split", default="val", choices=["val"])
    ap.add_argument("--table", default=None)
    ap.add_argument("--policy", default=None, help="fitted policy, needed for the system 'learned'")
    ap.add_argument("--max-alerts", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    policy = None
    if "learned" in a.systems:
        if not a.policy:
            raise SystemExit("--policy is required for the system 'learned' (scripts/train_policy.py)")
        from ftm.planners.learned import LearnedPolicy

        policy = LearnedPolicy.load(a.policy)
    if a.synthetic:
        from ftm.orchestrator.ablations import synthetic_env

        env = synthetic_env(a.synthetic, a.seed)
        banks, val, pool = env.banks, env.val, env.pools["realistic"]
    else:
        if not a.table:
            raise SystemExit("--table is required for real data (run scripts/backtest.py --assemble first)")
        from ftm.backtest.world import load_world

        world = load_world(a.data, a.split, a.seed, max_alerts=a.max_alerts)
        banks, val, pool = world.banks, world.alerts, QuestionPool.from_jsonl(a.table)
    ev = split_val(val)[1]  # the half no policy or fusion was fitted on, the same alerts as scripts/run_ablations.py
    waves = tuple(a.waves)
    grid = batching_grid(a.systems, banks, ev, pool, a.budgets, waves, EPS_LEVELS, a.seed, policy)
    load = load_study(a.systems[0] if a.systems[0] != "learned" else "eig", banks, ev, pool, a.gaps, waves, EPS_LEVELS, a.seed)
    out = {"alerts": len(ev), "waves": list(waves), "rows": [r for r in grid if r["arm"] != "h8"],
           "h8": [r for r in grid if r["arm"] == "h8"], "load": load}
    for r in out["rows"]:
        print({k: (round(r[k], 3) if isinstance(r[k], float) else r[k]) for k in
               ("system", "budget", "arm", "brier_gain", "mean_questions", "share_2plus", "mean_sim_days", "makespan_days")})
    for r in out["h8"]:
        print("H8", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "arm"})
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "async.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
