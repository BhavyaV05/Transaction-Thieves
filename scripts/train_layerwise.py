"""Ceydeli et al. layer-wise exchange on the 30 HI-Small / HI-Medium bank views (Group B, AML-specific method; secondary).

Owner: T1 (Abhishek).  Method and every deviation from the paper: `ftm/baselines/layerwise.py`.  Train split only for training, val for
scoring, no test row anywhere.  Same bank views, evaluation rows, Platt calibrator on the pooled val `fit` rows and metrics as the
DP-FedAvg numbers (`ftm.baselines.federated.evaluate`).

  python scripts/train_layerwise.py --data hi_small --arms sync_sgd:bwd fedavg:bwd fedprox:bwd local:none --seeds 0 1 2

  python scripts/train_layerwise.py --data hi_medium --steps 480 --lr 6e-3 --arms sync_sgd:bwd --seeds 0     # neighbour-sampled (automatic on HI-Medium)

`--sampled` / `--fanout` / `--eval-batch`: the paper's neighbour-sampled mini-batches (`ftm.baselines.layerwise.Sampler`); HI-Medium is always sampled
(full batch does not fit 8 GB there). `--scratch DIR` redirects every output (run json, scores, final files) for smoke tests.
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

SCORES, RUNS = Path("data/scores"), Path("data/runs")  # `--scratch DIR` moves both
SAMPLED_DATASETS = ("hi_medium",)  # full-batch message passing over a client's whole view needs more than 8 GB there
COLS = ["tx_id", "timestamp", "from_bank", "to_bank", "src_id", "dst_id", "amount_received", "amount_paid", "receiving_currency",
        "payment_currency", "payment_format", "is_laundering", "split"]  # fmt: skip


def commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def load(ds: str, banks: list[int] | None, min_train_pos: int, final: bool = False):
    """`final=True` (week 14, needs FTM_FINAL_RUN=1) returns two more items: the views with the TEST rows appended and their edge features
    (category codes and the time origin come from the train/val rows, so the train/val rows have the same features as in the val run)."""
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
    if not final:
        return frames, x, parts, skipped, cut
    frames_te = {b: io.scan_view(cfg["views_dir"], b, split=["train", "val", "test"], columns=COLS).collect().sort(["timestamp", "tx_id"]) for b in frames}
    x_te = {b: np.hstack([fed.tx_features(f, codes), ((f["timestamp"] - t0).dt.total_seconds().to_numpy() / (10 * 86400.0)).astype("f4")[:, None]]).astype("f4")
            for b, f in frames_te.items()}  # fmt: skip
    return frames, x, parts, skipped, cut, frames_te, x_te


def final_run(a) -> None:
    """Week 14: rerun each arm at the step chosen on val (`best_step` of `data/runs/layerwise_<ds>.json`, no selection) and score the TEST rows.
    Needs FTM_FINAL_RUN=1 (scripts/run_final.py). The val rows of the same run give the Platt calibrator (pooled val `fit` rows); their raw
    scores are compared with the stored val run (GPU scatter is not bit-deterministic: the rank correlation must be above 0.95). Writes
    data/scores/<ds>/final/ceydeli_<sync>_<ex>_seed<k>.parquet (test rows) and data/runs/final_layerwise_<ds>.json."""
    from scipy.stats import spearmanr

    from ftm import guards

    guards.require_final_run("the test split of the layer-wise exchange")
    frames, x, parts, skipped, cut, frames_te, x_te = load(a.data, a.banks, a.min_train_pos, final=True)
    val_rows = json.loads(Path(a.out or RUNS / f"layerwise_{a.data}.json").read_text())["rows"]
    out = SCORES / a.data / "final"
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    for arm in a.arms:
        sync, ex = arm.split(":")
        for seed in a.seeds:
            ref = next(r for r in val_rows if r["arm"] == arm and r["seed"] == seed and r["steps"] == a.steps)
            res = lw.run(frames, x, sync=sync, exchange=ex, steps=a.steps, local_steps=a.local_steps, parts=parts, batch=a.batch, lr=a.lr, seed=seed, device=a.device,
                         log=lambda s: print(s, flush=True), fixed_step=ref["best_step"], frames_test=frames_te, x_edge_test=x_te,
                         sampled=a.sampled, fanout=a.fanout, eval_batch=a.eval_batch)  # fmt: skip
            banks = list(frames)
            y = np.concatenate([frames[b].filter(pl.col("split") == "val")["is_laundering"].cast(pl.Float32).to_numpy() for b in banks])
            part, raw = np.concatenate([parts[b] for b in banks]), np.concatenate(res["raw"])
            fit = part == "fit"
            cal, _ = g.fit_calibrator(raw[fit], y[fit], has_signal=False)
            tx_te = np.concatenate(res["test_idx_tx"])
            te = {b: frames_te[b].filter(pl.col("split") == "test") for b in banks}
            lab = np.concatenate([te[b]["is_laundering"].to_numpy() for b in banks]).astype(bool)
            assert np.array_equal(tx_te, np.concatenate([te[b]["tx_id"].to_numpy() for b in banks]))
            raw_te = np.concatenate(res["raw_test"])
            pl.DataFrame({"tx_id": tx_te, "bank": np.concatenate([[b] * len(te[b]) for b in banks]), "mode": "one_bank", "model": f"ceydeli_{sync}_{ex}", "split": "test",
                          "val_part": None, "p0": cal(raw_te), "score_raw": raw_te, "label": lab, "seed": seed, "commit": commit()},
                         schema_overrides={"val_part": pl.Utf8}).write_parquet(out / f"{sync}_{ex}_seed{seed}.parquet")  # fmt: skip
            old = pl.read_parquet(SCORES / a.data / "ceydeli" / f"{sync}_{ex}_seed{seed}.parquet").sort(["tx_id", "bank"])  # time-guard: not transactions
            nv = pl.DataFrame({"tx_id": np.concatenate(res["val_idx_tx"]), "bank": np.concatenate([[b] * len(t) for b, t in zip(banks, res["val_idx_tx"], strict=True)]), "raw": raw}).sort(["tx_id", "bank"])
            rho = float(spearmanr(old["score_raw"].to_numpy(), nv["raw"].to_numpy())[0])
            recs.append({"arm": arm, "seed": seed, "step": ref["best_step"], "val_spearman_vs_stored": rho, "val_max_abs_diff_vs_stored": float(np.max(np.abs(old["score_raw"].to_numpy() - nv["raw"].to_numpy()))),
                         "test_rows": int(len(tx_te))})  # fmt: skip
            print("final", recs[-1], flush=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"final_layerwise_{a.data}.json").write_text(json.dumps({"commit": commit(), "sampled": a.sampled, "records": recs}, indent=1))
    if any(r["val_spearman_vs_stored"] < 0.95 for r in recs):
        raise SystemExit("the rerun does not reproduce the stored val scores (rank correlation < 0.95): do not use these test scores")


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
    ap.add_argument("--eval-every", type=int, default=40)
    ap.add_argument("--sampled", action="store_true", help="neighbour-sampled mini-batches (the paper's training); automatic on HI-Medium")
    ap.add_argument("--fanout", type=int, default=100, help="neighbours drawn per node and hop when sampled (the paper: 100)")
    ap.add_argument("--eval-batch", type=int, default=8192, help="seed edges per client and chunk when scoring sampled")
    ap.add_argument("--out", default=None)
    ap.add_argument("--scratch", default=None, help="write every output under this directory instead of data/scores and data/runs (smoke tests)")
    ap.add_argument("--final", action="store_true", help="week 14: rerun the arms at their val-chosen step and score the TEST rows (needs FTM_FINAL_RUN=1)")
    a = ap.parse_args()
    global SCORES, RUNS
    if a.scratch:
        SCORES, RUNS = Path(a.scratch) / "scores", Path(a.scratch) / "runs"
    a.sampled = a.sampled or a.data in SAMPLED_DATASETS
    if a.final:
        return final_run(a)
    t0 = time.time()
    frames, x, parts, skipped, cut = load(a.data, a.banks, a.min_train_pos)
    print(f"{len(frames)} clients, skipped {skipped}, loaded in {time.time() - t0:.0f}s", flush=True)
    out = Path(a.out or RUNS / f"layerwise_{a.data}.json")
    rows = json.loads(out.read_text())["rows"] if out.exists() else []
    sdir = SCORES / a.data / "ceydeli"
    sdir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    for arm in a.arms:
        sync, ex = arm.split(":")
        for seed in a.seeds:
            if any(r["arm"] == arm and r["seed"] == seed and r["steps"] == a.steps for r in rows):
                continue  # resume
            t1 = time.time()
            res = lw.run(frames, x, sync=sync, exchange=ex, steps=a.steps, local_steps=a.local_steps, parts=parts, batch=a.batch, lr=a.lr,
                         seed=seed, device=a.device, log=lambda s: print(s, flush=True), eval_every=a.eval_every, sampled=a.sampled, fanout=a.fanout,
                         eval_batch=a.eval_batch)  # fmt: skip
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
                         "history": res["history"], "seconds": time.time() - t1, "commit": commit(), "clients": len(banks), "skipped": skipped,
                         "sampled": a.sampled, "fanout": a.fanout if a.sampled else None, "eval_every": a.eval_every, "eval_batch": a.eval_batch if a.sampled else None})  # fmt: skip
            out.write_text(json.dumps({"dataset": a.data, "val_cut": [str(cut[0]), str(cut[1])], "rows": rows}, indent=1, default=str))
            print(arm, seed, metrics, f"{time.time() - t1:.0f}s", flush=True)


if __name__ == "__main__":
    main()
