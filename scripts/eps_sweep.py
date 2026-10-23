"""The eps sweep (E2): one system at per-alert budgets 0.5, 1, 2, 4 and no noise, with the theoretical ceiling.

Owner: T2 (Bhavya).  Spec: plan.md 6.6, tracks.md 4.11.  Run from the repo root.

    python scripts/eps_sweep.py --system eig --synthetic 360                       # smoke test, never a result
    python scripts/eps_sweep.py --system eig --data hi_medium --split val \\
        --table-noisy data/backtest/hi_medium/<ns>/lr_table.jsonl --table-exact data/backtest/hi_medium/<ns_exact>/lr_table.jsonl

`--table-noisy` is the assembled LR table of a back-test at the eps levels 0.25 to 2 (`scripts/backtest.py --eps 0.25 0.5 1 2`),
`--table-exact` that of `scripts/backtest.py --exact`. Writes one JSON row per budget to `--out` and prints the table.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.orchestrator.eps_sweep import BUDGETS, EPS_LEVELS, eps_sweep, synthetic_eps_sweep  # noqa: E402
from ftm.orchestrator.runner import SYSTEMS  # noqa: E402
from ftm.planners.pool import QuestionPool  # noqa: E402


def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=SYSTEMS, required=True)
    ap.add_argument("--synthetic", type=int, default=None)
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--table-noisy", default=None)
    ap.add_argument("--table-exact", default=None)
    ap.add_argument("--budgets", type=float, nargs="+", default=[*BUDGETS[:-1]])
    ap.add_argument("--no-noise", action=argparse.BooleanOptionalAction, default=True, help="add the eps = inf point")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-alerts", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    budgets = [*a.budgets, *([math.inf] if a.no_noise else [])]
    if a.synthetic:
        rows = synthetic_eps_sweep(a.system, a.synthetic, a.seed, budgets=budgets)
    else:
        from ftm.backtest.world import load_world

        if not a.table_noisy:
            raise SystemExit("--table-noisy is required for real data (scripts/backtest.py --assemble)")
        world = load_world(a.data, a.split, a.seed, max_alerts=a.max_alerts)
        exact = QuestionPool.from_jsonl(a.table_exact) if a.table_exact else None
        rows = eps_sweep(a.system, world.banks, world.alerts, QuestionPool.from_jsonl(a.table_noisy), exact, budgets,
                         EPS_LEVELS, a.seed, a.threshold)  # fmt: skip
    keep = ("budget", "brier_p0", "brier_final", "brier_gain", "mean_questions", "mean_eps", "ceiling_lr")
    for r in rows:
        print({k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in keep})
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=1))
    return rows


if __name__ == "__main__":
    main()
