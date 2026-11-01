"""HI-Medium see-everything on the val `eval` part: MEGA-GNN vs Multi-GNN vs GFP, the same rows and bootstrap as `train_gnn.py --compare` (T3, CPU).

    python scripts/mega_gnn_vs_gfp.py

Reads data/scores/hi_medium/p0_{mega_gnn,multi_gnn}__see_everything.parquet and the GFP scores; writes data/runs/mega_gnn_vs_gfp_hi_medium.json.
Val only. The Multi-GNN and GFP rows reproduce data/runs/multi_gnn_vs_gfp_hi_medium.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.baselines import gfp_gbt as g  # noqa: E402
from ftm.data import io  # noqa: E402
from ftm.eval import bootstrap as bs  # noqa: E402

DS = "hi_medium"


def see_everything(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter((pl.col("mode") == "see_everything") & pl.col("p0").is_not_null() & pl.col("p0").is_not_nan())


def stats(df: pl.DataFrame) -> dict:
    df = see_everything(df)
    fit, ev = df.filter(pl.col("val_part") == "fit"), df.filter(pl.col("val_part") == "eval")
    thr = g.metrics(fit["label"].cast(pl.Int8).to_numpy(), fit["p0"].to_numpy())["thr"]
    m = g.metrics(ev["label"].cast(pl.Int8).to_numpy(), ev["p0"].to_numpy(), thr)
    return {k: m.get(k) for k in ("n", "n_pos", "brier", "pr_auc", "f1")}


def eval_col(df: pl.DataFrame, name: str) -> pl.DataFrame:
    d = see_everything(df).filter(pl.col("val_part") == "eval")
    return d.select("tx_id", "label", pl.col("p0").alias(name)).unique("tx_id")


def main() -> None:
    cfg = io.load_config(DS)
    d = Path("data/scores") / DS
    me = pl.read_parquet(d / "p0_mega_gnn__see_everything.parquet")  # time-guard: not transactions
    mg = pl.read_parquet(d / "p0_multi_gnn__see_everything.parquet")  # time-guard: not transactions
    gf = g.read_p0(DS, split="val")
    out: dict = {"dataset": DS, "rows": {"mega_gnn/see_everything": stats(me), "multi_gnn/see_everything": stats(mg), "gfp_lgbm/see_everything": stats(gf)}}
    a = eval_col(me, "p_mega").join(eval_col(mg, "p_multi").drop("label"), on="tx_id").join(eval_col(gf, "p_gfp").drop("label"), on="tx_id")
    pq = cfg["parquet_dir"]
    tx = io.scan_transactions(pq, split="val", columns=["tx_id", "src_id", "dst_id"]).collect()
    pat = io.scan_patterns(pq, split="val").select("tx_id", "attempt_id").collect()
    j = a.join(tx, on="tx_id", how="left").join(pat, on="tx_id", how="left").sort("tx_id")
    cl = bs.cluster_ids(j["attempt_id"].to_list(), [j["src_id"].to_list(), j["dst_id"].to_list()])
    y = j["label"].cast(pl.Int8).to_numpy()
    out["common_n"], out["common_pos"] = len(y), int(y.sum())
    out["common_brier"] = {c: float(np.mean((j[c].to_numpy() - y) ** 2)) for c in ("p_mega", "p_multi", "p_gfp")}
    for name, x, z in (("gfp_minus_mega", "p_gfp", "p_mega"), ("multi_gnn_minus_mega", "p_multi", "p_mega"), ("gfp_minus_multi_gnn", "p_gfp", "p_multi")):
        r = bs.brier_diff(y, j[x].to_numpy(), j[z].to_numpy(), cl, n_boot=2000)
        out[f"brier_{name}"] = {"diff": r["diff"], "ci95": r["ci"]}
    Path(f"data/runs/mega_gnn_vs_gfp_{DS}.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
