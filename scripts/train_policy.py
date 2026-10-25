"""Train the learned question-selection policy (C9, week 9) on the `fit` half of the validation alerts.

Owner: T2 (Bhavya).  Spec: tracks.md 4.7 ("learned policy: gradient-boosted ranker trained on validation to predict Brier
gain from (alert features, question)").  Run from the repo root; needs LightGBM (`pip install -e ".[gbt]"`).

    # offline smoke test on the synthetic world (never a result)
    python scripts/train_policy.py --synthetic 600 --out outputs/policy/synthetic.json
    # real: the LR table that `scripts/backtest.py --assemble` wrote, the val alerts of T1's alert set
    python scripts/train_policy.py --data hi_medium --split val --table data/backtest/hi_medium/<ns>/lr_table.jsonl \\
        --out outputs/policy/hi_medium.json

The validation alerts are cut in time (`ftm.planners.learned.split_val`); the policy is fitted on the first half and checked on
the second: the rank correlation of its predicted gain with the gain the answers really gave, next to the same number for EIG's
closed-form expected gain (the `eig` feature). `scripts/run_experiment.py --system learned --policy <out>` and
`scripts/run_ablations.py` score it on that second half. The questions asked to build the rows are charged to eps_cal (printed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from ftm.orchestrator.policy_data import collect_policy_data  # noqa: E402
from ftm.planners.learned import FEATURES, LearnedPolicy, split_val  # noqa: E402
from ftm.planners.pool import QuestionPool  # noqa: E402


def heldout_check(policy: LearnedPolicy, data) -> dict:
    """Spearman correlation with the realised gain on rows the policy never saw: its prediction, and EIG's expected gain."""
    if len(data.y) < 10 or np.std(data.y) == 0:
        return {"rows": int(len(data.y)), "spearman_policy": None, "spearman_eig": None}
    pred = policy.predict(data.X)
    eig = data.X[:, FEATURES.index("eig")]
    rho = lambda a: None if np.std(a) == 0 else float(spearmanr(a, data.y)[0])  # noqa: E731
    return {"rows": int(len(data.y)), "spearman_policy": rho(pred), "spearman_eig": rho(eig),
            "mse_policy": float(np.mean((pred - data.y) ** 2)), "mse_zero": float(np.mean(data.y**2))}  # fmt: skip


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", type=int, default=None, help="n alerts of the synthetic world (smoke test)")
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--split", default="val", choices=["val"], help="the policy is trained on validation alerts only")
    ap.add_argument("--table", default=None, help="lr_table.jsonl from scripts/backtest.py --assemble")
    ap.add_argument("--fit-frac", type=float, default=0.5)
    ap.add_argument("--depth", type=int, default=3, help="questions per rollout")
    ap.add_argument("--max-options", type=int, default=24, help="questions asked per state (random sample if more)")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--max-alerts", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

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
    fit, held = split_val(val, a.fit_frac)
    data = collect_policy_data(banks, fit, pool, depth=a.depth, max_options=a.max_options, seed=a.seed)
    policy = LearnedPolicy.fit(data, rounds=a.rounds, seed=a.seed)
    check = heldout_check(policy, collect_policy_data(banks, held, pool, depth=a.depth, max_options=a.max_options,
                                                      seed=a.seed + 1, families=data.families))  # fmt: skip
    policy.meta.update({"fit_alerts": len(fit), "heldout": check, "pool": len(pool), "world": "synthetic" if a.synthetic else a.data})
    path = policy.save(a.out)
    summary = {"policy": str(path), **policy.meta, "importance": policy.importance()}
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
