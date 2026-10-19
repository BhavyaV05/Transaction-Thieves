"""Gap check with the T4 eval library: one-bank vs see-everything p0 on validation, cluster-bootstrap CIs.

Owner: T4 (Shreyas).  Run from the repo root, e.g. python scripts/eval_report.py --data hi_small --on alerts

--on rows    every scored val `eval` row (what ftm/baselines/README.md reports, there with a row bootstrap)
--on alerts  only the val alerts (data/alerts/<ds>/val.parquet, val_part == eval): the rows the agent sees,
             so this is the denominator of gap closed for our system (plan.md 9.1)

Prints Brier / PR-AUC / ECE per model, Brier(one bank) - Brier(see everything) with a cluster CI and a row CI
side by side, and the cluster sizes for both clusterings (attempt + accounts, attempt + transaction).
Writes data/runs/eval_gap_<ds>_<on>.json. Needs p0 for both modes (hi_small today; hi_medium after the
see-everything run).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.baselines.gfp_gbt import read_p0  # noqa: E402
from ftm.data import io  # noqa: E402
from ftm.eval import bootstrap as bs  # noqa: E402
from ftm.eval import metrics as mt  # noqa: E402


def load_frame(ds: str, on: str) -> pl.DataFrame:
    """One row per (transaction, bank view): y, p_ob, p_se, tx_id, bank, src_id, dst_id, attempt_id."""
    cfg = io.load_config(ds)
    pq = cfg["parquet_dir"]
    p0 = read_p0(ds, split="val").filter(pl.col("val_part") == "eval")
    se = p0.filter(pl.col("mode") == "see_everything").select("tx_id", pl.col("p0").alias("p_se"))
    if se.height == 0:
        raise SystemExit(f"{ds}: no see-everything p0 yet (ftm/baselines/README.md)")
    if on == "alerts":
        ob = pl.read_parquet(f"data/alerts/{ds}/val.parquet")  # time-guard: not transactions
        ob = ob.filter(pl.col("val_part") == "eval").select("tx_id", "bank", pl.col("p0").alias("p_ob"), "label")
    else:
        ob = p0.filter(pl.col("mode") == "one_bank").select("tx_id", "bank", pl.col("p0").alias("p_ob"), "label")
    ob = ob.drop_nulls("p_ob")
    tx = io.scan_transactions(pq, split="val", columns=["tx_id", "src_id", "dst_id"]).collect()
    pat = io.scan_patterns(pq, split="val").select("tx_id", "attempt_id").collect()
    f = ob.join(se, on="tx_id", how="inner").join(tx, on="tx_id", how="left").join(pat, on="tx_id", how="left")
    return f.with_columns(pl.col("label").cast(pl.Int8).alias("y")).sort("tx_id", "bank")


def report(f: pl.DataFrame, n_boot: int, seed: int, hub_cap: int | None = None) -> dict:
    y, pob, pse = f["y"].to_numpy(), f["p_ob"].to_numpy(), f["p_se"].to_numpy()
    att = f["attempt_id"].to_list()
    clusterings = {
        "attempt+accounts": bs.cluster_ids(att, [f["src_id"].to_list(), f["dst_id"].to_list()]),
        "attempt+tx": bs.cluster_ids(att, [f["tx_id"].to_list()]),  # a tx seen by two banks is one cluster
        "row": np.arange(len(y)),
    }
    if hub_cap is not None:  # sensitivity: hub accounts (in more than hub_cap rows) do not link
        clusterings[f"attempt+accounts(hub_cap={hub_cap})"] = bs.cluster_ids(
            att, [f["src_id"].to_list(), f["dst_id"].to_list()], hub_cap=hub_cap
        )
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "n_boot": n_boot,
           "one_bank": mt.detection_summary(y, pob), "see_everything": mt.detection_summary(y, pse),
           "brier_diff_ob_minus_se": {}}  # fmt: skip
    for name, cl in clusterings.items():
        d = bs.brier_diff(y, pob, pse, cl, n_boot=n_boot, seed=seed)
        out["brier_diff_ob_minus_se"][name] = {"diff": d["diff"], "ci95": d["ci"], "clusters": bs.cluster_report(cl)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--on", choices=["rows", "alerts"], default="alerts")
    ap.add_argument("--n-boot", type=int, default=bs.N_BOOT)
    ap.add_argument("--seed", type=int, default=bs.SEED)
    ap.add_argument("--hub-cap", type=int, default=None, help="also report attempt+accounts without hub accounts")
    a = ap.parse_args()
    out = report(load_frame(a.data, a.on), a.n_boot, a.seed, a.hub_cap)
    path = Path("data/runs") / f"eval_gap_{a.data}_{a.on}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
