"""Top-5 feature attributions per alert (hand-off for T4's Sharma port: `alert.summary["top_features"]`).

Owner: T1 (Abhishek).  For every alert of `data/alerts/<ds>/{train,val}.parquet` the home bank's one-bank GFP+LightGBM
model is refit with the exact settings of `scripts/train_gfp.py` (seed 0, threads 4), saved under
`models/gfp_lgbm/<ds>/bank<id>.txt`, and the alert's top-5 LightGBM `pred_contrib` (SHAP) features are written to
`data/alerts/<ds>/top_features_{train,val}.parquet`. Val alerts use the full model, train alerts the out-of-fold model
that produced their p0. The refit is checked against the stored p0 table (`data/runs/top_features_<ds>.json`).

  python scripts/top_features.py --data hi_small
  python scripts/top_features.py --data hi_medium           # bank 70 alone takes about 2 h of GFP features
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.baselines import gfp_gbt as g  # noqa: E402
from ftm.data import io  # noqa: E402

SEED = 0
COLS = ["tx_id", "timestamp", "from_bank", "to_bank", "src_id", "dst_id", "amount_received", "amount_paid",
        "receiving_currency", "payment_currency", "payment_format", "is_laundering", "split"]  # fmt: skip


def commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def one_bank(ds: str, bank: int, tx_ids: np.ndarray, cut, threads: int, mdir: Path, out_dir: Path, stored: pl.DataFrame) -> dict:
    cfg = io.load_config(ds)
    frame = io.scan_view(cfg["views_dir"], bank, split=["train", "val"], columns=COLS).collect()
    frame = frame.with_columns((pl.col("from_bank") == bank).cast(pl.Int8).alias("out"))
    t0 = time.time()
    res = g.score_graph(frame, g.category_codes(frame), cut, seed=SEED, threads=threads, extra_cols=["out"],
                        explain_tx=tx_ids)  # fmt: skip
    if "explain" not in res:
        return {"bank": bank, "status": res["meta"].get("status"), "alerts": len(tx_ids), "explained": 0}
    res["model"].booster_.save_model(str(mdir / f"bank{bank}.txt"))
    # reproduction check against the stored p0 table (same rows, same bank)
    new = pl.DataFrame({"tx_id": res["frame"]["tx_id"], "raw_new": res["raw"], "p0_new": res["p0"]})
    j = stored.filter(pl.col("bank") == bank).join(new, on="tx_id")
    both = j.filter(pl.col("score_raw").is_not_null() & pl.col("raw_new").is_not_nan())
    d_raw = (both["score_raw"] - both["raw_new"]).abs()
    d_p0 = (both["p0"] - both["p0_new"]).abs()
    res["explain"].with_columns(pl.lit(bank).cast(pl.Int64).alias("bank")).write_parquet(out_dir / f"explain_bank{bank}.parquet")
    return {"bank": bank, "status": "ok", "alerts": len(tx_ids), "explained": res["explain"].height, "rows_compared": both.height,
            "max_abs_diff_score_raw": float(d_raw.max()), "max_abs_diff_p0": float(d_p0.max()),
            "rows_raw_diff_gt_1e-9": int((d_raw > 1e-9).sum()), "calibration": res["meta"]["calibration"],
            "seconds": time.time() - t0}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--threads", type=int, default=4, help="the value train_gfp.py used for the stored p0")
    ap.add_argument("--banks", type=int, nargs="*", default=None, help="subset (debugging); default every alert bank")
    ap.add_argument("--assemble-only", action="store_true", help="only merge the per-bank parts into the parquet files")
    a = ap.parse_args()
    cfg = io.load_config(a.data)
    adir = Path(cfg["alerts_dir"])
    alerts = pl.concat([pl.read_parquet(adir / f"{s}.parquet") for s in ("train", "val")])  # time-guard: not transactions
    v = io.scan_transactions(cfg["parquet_dir"], split="val", columns=["timestamp"]).select(
        pl.col("timestamp").min().alias("a"), pl.col("timestamp").max().alias("b")).collect()  # fmt: skip
    cut = (v["a"][0], v["b"][0])
    stored = pl.read_parquet(Path("data/scores") / a.data / "p0_one_bank.parquet")  # time-guard: not transactions
    out_dir = Path("data/runs") / f"top_features_{a.data}"
    out_dir.mkdir(parents=True, exist_ok=True)
    mdir = Path("models/gfp_lgbm") / a.data
    mdir.mkdir(parents=True, exist_ok=True)
    banks = a.banks or sorted(alerts["bank"].unique().to_list())
    log = []
    for b in banks if not a.assemble_only else []:
        if (out_dir / f"explain_bank{b}.parquet").exists():
            continue  # resume
        r = one_bank(a.data, b, alerts.filter(pl.col("bank") == b)["tx_id"].to_numpy(), cut, a.threads, mdir, out_dir, stored)
        (out_dir / f"report_bank{b}.json").write_text(json.dumps(r, indent=1))
        print(r, flush=True)
    parts = [pl.read_parquet(f) for f in sorted(out_dir.glob("explain_bank*.parquet"))]
    ex = pl.concat(parts)
    for s in ("train", "val"):
        al = pl.read_parquet(adir / f"{s}.parquet").select("alert_id", "tx_id", "bank")  # time-guard: not transactions
        out = al.join(ex, on=["tx_id", "bank"], how="left")
        missing = out.filter(pl.col("top_features").is_null()).height
        out.select("alert_id", "tx_id", "bank", "top_features").write_parquet(adir / f"top_features_{s}.parquet")
        log.append({"split": s, "alerts": out.height, "without_top_features": missing})
    reports = [json.loads(f.read_text()) for f in sorted(out_dir.glob("report_bank*.json"))]
    summary = {"dataset": a.data, "commit": commit(), "seed": SEED, "threads": a.threads, "files": log, "banks": reports,
               "max_abs_diff_score_raw": max((r.get("max_abs_diff_score_raw", 0.0) for r in reports), default=None),
               "max_abs_diff_p0": max((r.get("max_abs_diff_p0", 0.0) for r in reports), default=None)}  # fmt: skip
    Path(f"data/runs/top_features_{a.data}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("files", "max_abs_diff_score_raw", "max_abs_diff_p0")}, indent=1))


if __name__ == "__main__":
    main()
