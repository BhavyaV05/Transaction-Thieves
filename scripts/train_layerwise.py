"""Ceydeli et al. layer-wise exchange on the 30 HI-Small bank views (Group B, AML-specific method; secondary).

Owner: T1 (Abhishek).  Method and every deviation from the paper: `ftm/baselines/layerwise.py`.  Train split only for training, val for
scoring, no test row anywhere.  Same bank views, evaluation rows, Platt calibrator on the pooled val `fit` rows and metrics as the
DP-FedAvg numbers (`ftm.baselines.federated.evaluate`).

  python scripts/train_layerwise.py --data hi_small --arms sync_sgd:bwd fedavg:bwd fedprox:bwd local:none --seeds 0 1 2

An arm is `<sync>:<exchange>`, sync in local|fedavg|fedprox|sync_sgd, exchange in none|fwd|bwd. Writes data/runs/layerwise_<ds>.json (one row
per run) and data/scores/<ds>/ceydeli/<arm>_seed<k>.parquet (calibrated p0 of every val view row; val rows only).
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

from ftm.baselines import federated as fed  # noqa: E402
from ftm.baselines import gfp_gbt as g  # noqa: E402
from ftm.baselines import layerwise as lw  # noqa: E402
from ftm.data import io  # noqa: E402

COLS = ["tx_id", "timestamp", "from_bank", "to_bank", "src_id", "dst_id", "amount_received", "amount_paid", "receiving_currency",
        "payment_currency", "payment_format", "is_laundering", "split"]  # fmt: skip


def commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def load(ds: str, banks: list[int] | None, min_train_pos: int):
    cfg = io.load_config(ds)
    order = io.read_banks(cfg["views_dir"]).sort("rank")["bank"].to_list()
    v = io.scan_transactions(cfg["parquet_dir"], split="val", columns=["timestamp"]).select(
        pl.col("timestamp").min().alias("a"), pl.col("timestamp").max().alias("b")).collect()  # fmt: skip
    cut = (v["a"][0], v["b"][0])
    frames = {b: io.scan_view(cfg["views_dir"], b, split=["train", "val"], columns=COLS).collect().sort(["timestamp", "tx_id"])
              for b in (banks or order)}  # fmt: skip
    skipped = {b: int(f.filter(pl.col("split") == "train")["is_laundering"].cast(pl.Int64).sum()) for b, f in frames.items()}
    skipped = {b: n for b, n in skipped.items() if n < min_train_pos}
    frames = {b: f for b, f in frames.items() if b not in skipped}
    codes = fed.category_codes(list(frames.values()))
    t0 = min(f["timestamp"].min() for f in frames.values())
    x = {}
    for b, f in frames.items():
        tfeat = ((f["timestamp"] - t0).dt.total_seconds().to_numpy() / (10 * 86400.0)).astype("f4")[:, None]
        x[b] = np.hstack([fed.tx_features(f, codes), tfeat]).astype("f4")
    parts = {}
    for b, f in frames.items():
        val = f.filter(pl.col("split") == "val")
        parts[b] = g.val_parts(val["timestamp"], *cut)
    return frames, x, parts, skipped, cut


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--arms", nargs="+", default=["sync_sgd:bwd"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--local-steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--banks", type=int, nargs="*", default=None, help="subset of banks (debugging)")
    ap.add_argument("--min-train-pos", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    frames, x, parts, skipped, cut = load(a.data, a.banks, a.min_train_pos)
    print(f"{len(frames)} clients, skipped {skipped}, loaded in {time.time() - t0:.0f}s", flush=True)
    out = Path(a.out or f"data/runs/layerwise_{a.data}.json")
    rows = json.loads(out.read_text())["rows"] if out.exists() else []
    sdir = Path("data/scores") / a.data / "ceydeli"
    sdir.mkdir(parents=True, exist_ok=True)
    for arm in a.arms:
        sync, ex = arm.split(":")
        for seed in a.seeds:
            if any(r["arm"] == arm and r["seed"] == seed and r["steps"] == a.steps for r in rows):
                continue  # resume
            t1 = time.time()
            res = lw.run(frames, x, sync=sync, exchange=ex, steps=a.steps, local_steps=a.local_steps, parts=parts, batch=a.batch, lr=a.lr,
                         seed=seed, device=a.device, log=lambda s: print(s, flush=True))  # fmt: skip
            banks = list(frames)
            clients = [{"bank": b, "yva": frames[b].filter(pl.col("split") == "val")["is_laundering"].cast(pl.Float32).to_numpy(),
                        "part_va": parts[b]} for b in banks]  # fmt: skip
            metrics = fed.evaluate(clients, res["raw"])
            raw = np.concatenate(res["raw"])
            y, part = np.concatenate([c["yva"] for c in clients]), np.concatenate([c["part_va"] for c in clients])
            fit = part == "fit"
            cal, _ = g.fit_calibrator(raw[fit], y[fit], has_signal=False)
            pl.DataFrame({"tx_id": np.concatenate(res["val_idx_tx"]), "bank": np.concatenate([[b] * len(t) for b, t in zip(banks, res["val_idx_tx"], strict=True)]),
                          "mode": "one_bank", "model": f"ceydeli_{sync}_{ex}", "split": "val", "val_part": part, "p0": cal(raw), "score_raw": raw,
                          "label": y.astype(bool), "seed": seed, "commit": commit()}).write_parquet(sdir / f"{sync}_{ex}_seed{seed}.parquet")  # fmt: skip
            rows.append({"arm": arm, "sync": sync, "exchange": ex, "seed": seed, "steps": a.steps, "local_steps": a.local_steps, "lr": a.lr,
                         "batch": a.batch, "best_step": res["best_step"], "best_val_fit_ap": res["best_val_fit_ap"], "metrics": metrics,
                         "history": res["history"], "seconds": time.time() - t1, "commit": commit(), "clients": len(banks), "skipped": skipped})  # fmt: skip
            out.write_text(json.dumps({"dataset": a.data, "val_cut": [str(cut[0]), str(cut[1])], "rows": rows}, indent=1, default=str))
            print(arm, seed, metrics, f"{time.time() - t1:.0f}s", flush=True)


if __name__ == "__main__":
    main()
