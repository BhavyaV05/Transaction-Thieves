"""Replication check: does each accepted question keep its direction inside time blocks of the train back-test?

Owner: T4 (Shreyas).  Spec: GitHub issue #84, docs/t4_fusion_findings.md section 7.  Method: `ftm/eval/replication.py`.
Run from the repo root.

    # smoke test on a planted world (a time-confounded question and a genuine one; no data, never a result)
    python scripts/replication_check.py --synthetic
    # the real check: asks the accepted questions of the cache again on the train alerts (about 30 min of CPU)
    python scripts/replication_check.py --data hi_medium --backtest-dir data/backtest/hi_medium/<namespace>
    # the same plus the validation direction of each cell (val part eval; reporting only, the flag never uses it)
    python scripts/replication_check.py --data hi_medium --backtest-dir data/backtest/hi_medium/<namespace> --val-part eval
    # four blocks must agree instead of two halves
    python scripts/replication_check.py --data hi_medium --backtest-dir ... --blocks 4

Writes outputs/replication/<data>.json (config, summary, one row per accepted question: question, eps, cell, claimed log
LR, pooled, per-block, Mantel-Haenszel, replicates, reason) and a .md table next to it, and prints one summary line.
The JSON is the input of `--replication-filter` in scripts/g2_check.py and scripts/fusion_diagnostics.py.
Exit code 0 = done, 2 = not run (data, alert set or cache missing).  Only the train and val splits are read; the test
split is never touched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftm.eval import replication as rep  # noqa: E402

DONE, NOT_RUN = 0, 2


def real_rows(a, cfg: rep.ReplicationConfig):
    """(rows, entries) of the real world: the accepted entries of the cache re-asked on the train alerts."""
    from ftm.backtest.engine import BacktestCache
    from ftm.backtest.world import load_world
    from ftm.eval import g2

    d = Path(a.backtest_dir)
    entries = g2.entries_from_cache(BacktestCache(d.parent, d.name), a.fdr)
    train = load_world(a.data, "train", a.seed)
    rows = rep.replication_report(train.banks, train.alerts, entries, cfg)
    if a.val_part:
        import polars as pl

        from ftm.data import io

        val = load_world(a.data, "val", a.seed)
        meta = pl.read_parquet(ROOT / io.load_config(a.data)["alerts_dir"] / "val.parquet")  # time-guard: not transactions
        keep = set(meta["alert_id"]) if a.val_part == "all" else set(meta.filter(pl.col("val_part") == a.val_part)["alert_id"])
        rep.add_validation(rows, entries, rep.backtest_collector(val.banks, [x for x in val.alerts if x.alert_id in keep]),
                           cfg.alpha)  # fmt: skip
    return rows, entries


def synthetic_rows(a, cfg: rep.ReplicationConfig):
    entries, train_collect, val_collect = rep.planted_world(a.n, a.seed)
    rows = rep.replication_report(None, None, entries, cfg, collector=train_collect)
    if a.val_part:
        rep.add_validation(rows, entries, val_collect, cfg.alpha)
    return rows, entries


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="planted world with a time confound; needs no data")
    ap.add_argument("--data", default="hi_medium", help="dataset name")
    ap.add_argument("--backtest-dir", default=None, help="assembled back-test cache directory data/backtest/<data>/<ns>")
    ap.add_argument("--val-part", default=None, choices=["fit", "cal", "eval", "all"],
                    help="also compute the val log LR and the direction agreement on this val part (default: none)")  # fmt: skip
    ap.add_argument("--blocks", type=int, choices=[2, 4], default=2, help="time blocks that must agree in sign (2 = halves)")
    ap.add_argument("--mh-blocks", type=int, default=4, help="strata of the Mantel-Haenszel estimate")
    ap.add_argument("--min-block-answers", type=int, default=rep.ReplicationConfig.min_block_answers)
    ap.add_argument("--min-block-class", type=int, default=rep.ReplicationConfig.min_block_class)
    ap.add_argument("--fdr", type=float, default=0.10)
    ap.add_argument("--n", type=int, default=800, help="answers per question of the planted world")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default outputs/replication/<data>.json (synthetic: synthetic.json)")
    a = ap.parse_args(argv)

    cfg = rep.ReplicationConfig(a.blocks, a.mh_blocks, a.min_block_answers, a.min_block_class)
    name = "synthetic" if a.synthetic else a.data
    try:
        if a.synthetic:
            rows, _ = synthetic_rows(a, cfg)
        else:
            if not a.backtest_dir:
                print("replication check NOT RUN: --backtest-dir is required (or use --synthetic)", file=sys.stderr)
                return NOT_RUN
            rows, _ = real_rows(a, cfg)
    except (FileNotFoundError, KeyError) as e:  # the data, the alert set or the cache is not on this machine
        print(f"replication check NOT RUN: {type(e).__name__}: {e}", file=sys.stderr)
        return NOT_RUN
    if not rows:
        print("replication check NOT RUN: no accepted question", file=sys.stderr)
        return NOT_RUN
    summary = rep.summarize(rows)
    out = Path(a.out or ROOT / "outputs" / "replication" / f"{name}.json")
    rep.write_report(out, rows, summary, cfg, data=name, val_part=a.val_part, backtest_dir=a.backtest_dir)
    out.with_suffix(".md").write_text(rep.to_markdown(rows, summary, name))
    print(rep.summary_line(summary, name))
    return DONE


if __name__ == "__main__":
    sys.exit(main())
