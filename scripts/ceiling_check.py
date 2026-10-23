"""Ceiling checks on the privacy code (G2). Owner: T4 (Shreyas). CPU only, about a minute.

    python scripts/ceiling_check.py                              # exact + empirical over every kind and eps
    python scripts/ceiling_check.py --backtest-dir <cache dir>   # also the unclipped back-test LRs (ftm/backtest cache)

Writes data/runs/ceiling_check.json and exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.eval import ceiling  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000, help="draws per true value in the empirical check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backtest-dir", default=None, help="directory of back-test cache entries (*.json)")
    ap.add_argument("--out", default="data/runs/ceiling_check.json")
    a = ap.parse_args()
    out = {"exact": [], "empirical": [], "backtest": None}
    for g in ceiling.default_grid():
        kw = {k: v for k, v in g.items() if k != "kind"}
        out["exact"].append(ceiling.exact_check(g["kind"], **kw))
        out["empirical"].append({k: v for k, v in ceiling.empirical_check(g["kind"], n=a.n, seed=a.seed, **kw).items()
                                 if k != "cells"})  # fmt: skip
    if a.backtest_dir:
        entries = [json.loads(p.read_text()) for p in sorted(Path(a.backtest_dir).glob("*.json"))]
        out["backtest"] = ceiling.backtest_check(entries)
    ok = all(r["ok"] is not False for r in out["exact"] + out["empirical"]) and (out["backtest"] is None or out["backtest"]["ok"])
    out["empirical_inconclusive"] = sum(r["ok"] is None for r in out["empirical"])
    out["ok"] = ok
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=str))
    bad = [r for r in out["exact"] + out["empirical"] if r["ok"] is False]
    print(f"exact {sum(r['ok'] for r in out['exact'])}/{len(out['exact'])} ok, "
          f"empirical {sum(r['ok'] is True for r in out['empirical'])}/{len(out['empirical'])} ok "
          f"({out['empirical_inconclusive']} inconclusive: no overlap to sample)"
          + (f", backtest violations {len(out['backtest']['violations'])}" if out["backtest"] else ""))  # fmt: skip
    for r in bad:
        print("FAIL", r)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
