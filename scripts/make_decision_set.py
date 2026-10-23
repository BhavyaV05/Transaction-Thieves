"""Build the small-decision dataset (hand-off #13, T2 to T3): labelled one-token decisions for the fast model.

Owner: T2 (Bhavya).  Spec: tracks.md 4.9.  Run from the repo root.

    # offline smoke test on the synthetic world (never a result)
    python scripts/make_decision_set.py --synthetic 300 --out /tmp/decisions
    # real: T1's alert sets and views, the LR table from scripts/backtest.py --assemble (for the 'stop' decisions)
    python scripts/make_decision_set.py --data hi_medium --table data/backtest/hi_medium/<ns>/lr_table.jsonl \
        --threshold 0.3 --max-total 50000

Writes data/decisions/<ds>/decisions_{train,val}.jsonl and manifest.json (counts, label balance, sha256). Train and
val alerts only; the test split is never read. The files are not committed (README rule 7): share them through the
agreed cloud folder with the manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.backtest.engine import BacktestConfig  # noqa: E402
from ftm.backtest.world import load_world  # noqa: E402
from ftm.orchestrator.decision_set import DecisionSetBuilder, write_set  # noqa: E402
from ftm.planners.pool import QuestionPool  # noqa: E402


def synthetic_parts(n: int, seed: int):
    from dataclasses import replace

    from ftm.backtest.synth import make_world
    from ftm.orchestrator.runner import pool_from_backtest

    w = make_world(n, seed)
    alerts = sorted(w.alerts, key=lambda a: a.binding.t0)
    cut = int(len(alerts) * 0.6)
    alerts = [replace(a, split="train" if i < cut else "val") for i, a in enumerate(alerts)]
    pool, _ = pool_from_backtest(w.banks, alerts[:cut], ["n_counterparties(ALERT.receiver, dir=in, window=before(72))",
                                                          "passed_on(ALERT.receiver, window=after(24), min_frac=0.5)"], (1.0, 2.0),
                              BacktestConfig(min_answers=40, min_per_class=6, min_per_bin=10))  # fmt: skip
    return w.ledgers, w.banks, alerts, pool


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--synthetic", type=int, default=None)
    ap.add_argument("--table", default=None, help="lr_table.jsonl; without it the 'stop' decisions are skipped")
    ap.add_argument("--threshold", type=float, default=0.3, help="decision threshold used by the 'stop' labels")
    ap.add_argument("--max-total", type=int, default=None)
    ap.add_argument("--max-alerts", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    pool = QuestionPool.from_jsonl(a.table) if a.table else None
    if a.synthetic:
        ledgers, banks, alerts, spool = synthetic_parts(a.synthetic, a.seed)
        pool = pool or spool
    else:
        ledgers, banks, alerts = {}, {}, []
        for split in ("train", "val"):
            w = load_world(a.data, split, a.seed, max_alerts=a.max_alerts)
            ledgers.update(w.ledgers)
            banks.update(w.banks)
            alerts += w.alerts
    builder = DecisionSetBuilder(ledgers, banks, pool, a.threshold, seed=a.seed)
    out = a.out or f"data/decisions/{a.data if not a.synthetic else 'synthetic'}"
    manifest = write_set(builder.build(alerts, a.max_total), out)
    print(json.dumps({k: manifest[k] for k in ("n", "by_split_decision")}, indent=1))
    return manifest


if __name__ == "__main__":
    main()
