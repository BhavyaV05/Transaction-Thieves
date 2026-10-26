"""All non-LLM planners and their ablation switches on validation: the G4 table for T2 (week 9).

Owner: T2 (Bhavya).  Spec: tracks.md 4.10 week 9, plan.md 7.4.  Run from the repo root.

    # offline smoke test on the synthetic world (never a result)
    python scripts/run_ablations.py --synthetic 600
    # real: the LR tables that `scripts/backtest.py --assemble` wrote (the enumerated and exact-label ones are optional)
    python scripts/run_ablations.py --data hi_medium --split val --table data/backtest/hi_medium/<ns>/lr_table.jsonl \\
        --table-enumerator data/backtest/hi_medium/<ns_enum>/lr_table.jsonl \\
        --table-exact-labels data/backtest/hi_medium/<ns_exact>/lr_table.jsonl --out outputs/g4

The arms and their settings are in `configs/experiments/ablations_nonllm.yaml`. Writes `ablations.json` (one row per arm) and
`ablations.md` (the table) to `--out`. An arm whose table is missing is listed as skipped, with the reason.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftm.orchestrator.ablations import (  # noqa: E402
    CONFIG,
    format_table,
    load_spec,
    run_ablations,
    synthetic_ablations,
)
from ftm.planners.learned import LearnedPolicy  # noqa: E402
from ftm.planners.pool import QuestionPool  # noqa: E402


def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--synthetic", type=int, default=None, help="n alerts of the synthetic world (smoke test)")
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--split", default="val", choices=["val"])
    ap.add_argument("--table", default=None, help="realistic lr_table.jsonl")
    ap.add_argument("--table-enumerator", default=None, help="lr_table.jsonl of the enumerated programs")
    ap.add_argument("--table-exact-labels", default=None, help="lr_table.jsonl of the --exact-labels back-test")
    ap.add_argument("--policy", default=None, help="a fitted policy (scripts/train_policy.py); default: fit it on the fit half")
    ap.add_argument("--only", nargs="+", default=None, help="run only these arms (the reference arm always runs)")
    ap.add_argument("--max-alerts", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "g4"))
    a = ap.parse_args(argv)

    spec = load_spec(a.config)
    if a.synthetic:
        rows = synthetic_ablations(a.synthetic, a.seed, spec, a.n_boot, a.only)
    else:
        if not a.table:
            raise SystemExit("--table is required for real data (run scripts/backtest.py --assemble first)")
        from ftm.backtest.world import load_world

        world = load_world(a.data, a.split, a.seed, max_alerts=a.max_alerts)
        pools = {"realistic": QuestionPool.from_jsonl(a.table)}
        if a.table_enumerator:
            pools["enumerator"] = QuestionPool.from_jsonl(a.table_enumerator)
        if a.table_exact_labels:
            pools["exact_labels"] = QuestionPool.from_jsonl(a.table_exact_labels)
        policy = LearnedPolicy.load(a.policy) if a.policy else None
        rows = run_ablations(world.banks, world.alerts, pools, spec, policy, a.seed, a.n_boot, a.only)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table = format_table(rows, spec.reference)
    (out / "ablations.json").write_text(json.dumps(rows, indent=1))
    (out / "ablations.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    return rows


if __name__ == "__main__":
    main()
