"""Fit the p0-conditional fusion (ftm.fusion.conditional) on TRAIN per-alert back-test rows.  CPU only.

Owner: T2.  Run from the repo root, after `scripts/backtest.py ... --assemble` has written the LR table and
`alert_rows.parquet` of a namespace:

    python scripts/fit_conditional_fusion.py --ns data/backtest/hi_medium/<namespace> \\
        --p0-file data/alerts/hi_medium/train.parquet --out outputs/fusion/conditional_<namespace>.json

--p0-file is any parquet with alert_id, p0 (a recalibrated train p0 goes here).  The ridge is chosen by contiguous-block CV on
train.  Optional: --val-rows (an alert_rows.parquet from `scripts/backtest.py --split val`) with --val-alerts
(data/alerts/<ds>/val.parquet: alert_id, p0, label, val_part) choose the shrink factor and base on val fit+cal rows only; eval and
test are never read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from ftm.fusion.conditional import ConditionalFusion, alert_matrix, choose_shrink, fit_ridge_cv  # noqa: E402
from ftm.llm.investigation import LRTable  # noqa: E402


def load_table(path: Path) -> LRTable:
    return LRTable.from_rows(json.loads(line) for line in path.read_text().splitlines() if line.strip())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", required=True, help="back-test namespace dir (lr_table.jsonl, alert_rows.parquet)")
    ap.add_argument("--rows", default=None)
    ap.add_argument("--table", default=None)
    ap.add_argument("--p0-file", default="data/alerts/hi_medium/train.parquet")
    ap.add_argument("--ridge", type=float, default=None, help="fixed ridge (pseudo-observations); default: CV over a grid")
    ap.add_argument("--draws", type=int, default=3, help="eps-level draws per (alert, question) stacked into the fit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-rows", default=None)
    ap.add_argument("--val-alerts", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ns = Path(a.ns)
    table = load_table(Path(a.table or ns / "lr_table.jsonl"))
    rows = pl.read_parquet(a.rows or ns / "alert_rows.parquet").to_dicts()
    p0f = pl.read_parquet(a.p0_file)
    if "timestamp" in p0f.columns:
        p0f = p0f.sort("timestamp")
    p0_of = dict(zip(p0f["alert_id"], p0f["p0"], strict=True))
    p0, X, y, fams, _ = alert_matrix(rows, table, p0_of, draws=a.draws, seed=a.seed)
    if not fams:
        raise SystemExit("no accepted question has rows: nothing to fit")
    if a.ridge is None:
        m = fit_ridge_cv(p0, X, y, fams)
    else:
        m = ConditionalFusion(ridge=a.ridge).fit(p0, X, y, fams)
    m.meta.update(p0_file=str(a.p0_file), table=str(a.table or ns / "lr_table.jsonl"), namespace=ns.name, n_rows=int(len(y)),
                  draws=a.draws, seed=a.seed)  # fmt: skip
    if a.val_rows:
        va = pl.read_parquet(a.val_alerts)
        va = (va.sort("timestamp") if "timestamp" in va.columns else va).filter(pl.col("val_part").is_in(["fit", "cal"]))  # eval is never used to choose anything
        vp0 = dict(zip(va["alert_id"], va["p0"], strict=True))
        vrows = [r for r in pl.read_parquet(a.val_rows).to_dicts() if r["alert_id"] in vp0]
        p, Xv, yv, _, _ = alert_matrix(vrows, table, vp0, families=fams, draws=1, seed=a.seed)
        grid = choose_shrink(m, p, Xv, yv)
        print("shrink grid (val fit+cal Brier):", json.dumps(grid, indent=1), file=sys.stderr)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    h = m.to_json(a.out)
    base_brier = float(np.mean((p0 - y) ** 2))
    fit_brier = float(np.mean((m.predict(p0, X) - y) ** 2))
    print(json.dumps({"out": a.out, "hash": h, "families": fams, "w": m.w.tolist(), "a": m.a, "b": m.b, "ridge": m.ridge,
                      "shrink": m.shrink, "base": m.base, "n": int(len(y)), "train_brier_p0": base_brier,
                      "train_brier_fused": fit_brier}, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
