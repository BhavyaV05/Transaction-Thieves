"""Multi-GNN / FraudGT / MEGA-GNN (one-bank and see-everything).

Owner: T1 (Abhishek).  Run from the repo root:
  python scripts/train_gnn.py --repro     # M2: Multi-GNN reproduction on its own split (prints the command)

REPRODUCTION ONLY (tracks.md 3.5): IBM's Multi-GNN code splits HI-Small 60/20/20 by transaction index, which
overlaps our test days, and logs the test F1 every epoch. That run reads only the raw transactions, writes
nothing to data/scores or data/alerts, and nothing from it feeds any later model or decision. Our-split runs
(one-bank, see-everything) are not implemented yet.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.baselines import multi_gnn  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro", action="store_true")
    ap.add_argument("--clone", default="~/Documents/t1_external/Multi-GNN")
    ap.add_argument("--budget-s", type=int, default=8700, help="stop after the epoch that passes this many seconds")
    ap.add_argument("--run", action="store_true", help="launch it (otherwise only print the command)")
    a = ap.parse_args()
    if not a.repro:
        raise SystemExit("only --repro is implemented")
    clone = Path(a.clone).expanduser()
    metrics = Path("data/multignn/pna_metrics.jsonl").resolve()
    print(f"cd {clone} && T1_BUDGET_S={a.budget_s} T1_METRICS={metrics} {shlex.join(multi_gnn.repro_command())}")
    if a.run:
        multi_gnn.launch(clone, metrics, a.budget_s).wait()


if __name__ == "__main__":
    main()
