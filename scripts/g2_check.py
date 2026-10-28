"""Gate G2: expert questions improve Brier by >= 0.01 at eps = 2 on validation, under the privacy ceiling.

Owner: T4 (Shreyas).  Spec: plan.md 13, tracks.md 2.2.  Run from the repo root.  The method is in `ftm/eval/g2.py`.

    # smoke test on the synthetic world (never a result): back-test on the first 60% of alerts, check on the rest
    python scripts/g2_check.py --world synth
    # the real check: the val alerts of the eval part, the expert questions back-tested on train
    python scripts/g2_check.py --world real --data hi_medium
    # the same with the back-test cache scripts/backtest.py already wrote (skips loading the train split)
    python scripts/g2_check.py --world real --data hi_medium --backtest-dir data/backtest/hi_medium/<namespace>
    python scripts/g2_check.py --world real --data hi_medium --fusion per_family     # cross-fitted inside val
    # per-family fusion with the ridge and the prior ("naive" or "zero") chosen by inner cross-validation
    python scripts/g2_check.py --world real --data hi_medium --fusion per_family --fusion-select cv
    # with the replication filter (docs/t4_fusion_findings.md section 7): accepted questions that do not keep their
    # direction across time blocks get e_value = 1 before e-BH (m unchanged); descriptive, off by default
    python scripts/g2_check.py --world real --data hi_medium --backtest-dir data/backtest/hi_medium/<ns> --replication-filter outputs/replication/hi_medium.json

The fixed-order expert planner asks up to four questions per alert (eps 0.5 each, budget 2) through the harness (gate,
accountant, clock); the answers go through fusion; the Brier gain over p0 gets a cluster-bootstrap CI. The ceiling check
runs on the back-test cache entries. Without `--backtest-dir` the real run back-tests the expert questions on train
itself (slow, but resumable: it writes the shared cache `data/backtest/<data>/<namespace>/`).

Writes data/runs/g2_<world>.json and prints a one-line verdict. Exit code 0 = pass, 1 = fail, 2 = not run.
The test split is never read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftm.backtest.engine import BacktestCache, BacktestConfig  # noqa: E402
from ftm.eval import bootstrap as bs  # noqa: E402
from ftm.eval import g2  # noqa: E402
from ftm.eval import replication as rep  # noqa: E402

PASS, FAIL, NOT_RUN = 0, 1, 2


def real_inputs(a):
    """(banks, entries, val alerts, attempt_of) of the real world; raises FileNotFoundError if the data is missing."""
    import polars as pl

    from ftm.backtest.world import load_world
    from ftm.data import io

    cfg = io.load_config(a.data)
    val = load_world(a.data, "val", a.seed)
    meta = pl.read_parquet(ROOT / cfg["alerts_dir"] / "val.parquet")  # time-guard: not transactions
    keep = set(meta["alert_id"]) if a.val_part == "all" else set(meta.filter(pl.col("val_part") == a.val_part)["alert_id"])
    alerts = [x for x in val.alerts if x.alert_id in keep]
    attempt_of = dict(zip(meta["alert_id"], meta["attempt_id"], strict=True)) if "attempt_id" in meta.columns else None
    if a.backtest_dir:
        d = Path(a.backtest_dir)
        entries = g2.entries_from_cache(BacktestCache(d.parent, d.name), a.fdr)
    else:
        train = load_world(a.data, "train", a.seed)
        bc = BacktestConfig()
        cache = BacktestCache(ROOT / "data" / "backtest" / Path(a.data).stem, bc.namespace(train.alert_set_id, a.seed))
        entries = g2.backtest_entries(train.banks, train.alerts, g2.expert_program_list(), a.eps_levels, bc, cache, a.fdr)
    return val.banks, entries, alerts, attempt_of


def _train(a):
    """(banks, alerts) of the train world, for --replication-blocks only."""
    from ftm.backtest.world import load_world

    w = load_world(a.data, "train", a.seed)
    return w.banks, w.alerts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", choices=["synth", "real"], default="synth")
    ap.add_argument("--data", default="hi_medium", help="dataset name (real world)")
    ap.add_argument("--val-part", default="eval", choices=["fit", "cal", "eval", "all"], help="val alerts scored (real world)")
    ap.add_argument("--backtest-dir", default=None, help="assembled back-test cache directory (real world)")
    ap.add_argument("--n", type=int, default=360, help="alerts of the synthetic world")
    ap.add_argument("--fusion", choices=g2.FUSIONS, default="naive")
    ap.add_argument("--fusion-select", choices=g2.SELECTS, default="fixed",
                    help="per_family only: 'cv' picks ridge and prior by inner cross-validation (fuse.fit_cv)")
    ap.add_argument("--budget", type=float, default=g2.BUDGET, help="eps per alert")
    ap.add_argument("--ask-eps", type=float, default=g2.ASK_EPS, help="eps level of each question")
    ap.add_argument("--eps-levels", type=float, nargs="+", default=list(g2.EPS_LEVELS))
    ap.add_argument("--fdr", type=float, default=0.10)
    ap.add_argument("--hub-cap", type=int, default=None, help="accounts in more alerts than this do not link clusters")
    ap.add_argument("--n-boot", type=int, default=bs.N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default data/runs/g2_<world>.json")
    rep.add_filter_args(ap)
    a = ap.parse_args(argv)

    try:
        if a.world == "synth":
            banks, entries, val = g2.synthetic_inputs(a.n, a.seed, eps_levels=a.eps_levels)
            attempt_of = None
        else:
            banks, entries, val, attempt_of = real_inputs(a)
    except (FileNotFoundError, KeyError) as e:  # the data, the alert set or the cache is not on this machine
        print(f"G2 NOT RUN: {type(e).__name__}: {e}", file=sys.stderr)
        return NOT_RUN
    if not val or not entries:
        print(f"G2 NOT RUN: {len(val)} validation alerts, {len(entries)} back-test entries", file=sys.stderr)
        return NOT_RUN
    filter_info = None
    if a.replication_filter or a.replication_blocks:
        if a.world != "real" and a.replication_blocks:
            print("G2 NOT RUN: --replication-blocks needs the real world (use --replication-filter)", file=sys.stderr)
            return NOT_RUN
        try:
            entries, filter_info = rep.filter_from_args(a, entries, lambda: _train(a), a.fdr)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"G2 NOT RUN: {type(e).__name__}: {e}", file=sys.stderr)
            return NOT_RUN
    r = g2.run_g2(banks, entries, val, a.world, a.fusion, a.budget, a.ask_eps, a.eps_levels, attempt_of, a.hub_cap,
                  a.n_boot, a.seed, fdr=a.fdr, fusion_select=a.fusion_select)  # fmt: skip
    if a.world == "real":
        r["data"], r["val_part"] = a.data, a.val_part
    if filter_info is not None:
        r["replication_filter"] = filter_info
    out = Path(a.out or ROOT / "data" / "runs" / f"g2_{a.world}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1, default=str))
    print(g2.verdict_line(r))
    if filter_info is not None:
        print(f"replication filter: {filter_info['n_dropped']} accepted questions dropped "
              f"({filter_info['accepted_before']} -> {filter_info['accepted_after']} accepted, m = {filter_info['m']})")
    return PASS if r["passed"] else FAIL


if __name__ == "__main__":
    sys.exit(main())
