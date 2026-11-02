"""HI-Medium see-everything GNNs on the val `eval` part: look-ahead scoring (stored p0) vs time-respecting scoring (`--causal`, run on ada).  Owner: T3 (Akshat).

    python scripts/gnn_causal_vs_noncausal.py

Reads data/scores/hi_medium/p0_{multi_gnn,mega_gnn}__see_everything.parquet (non-causal: a val edge's sampled neighbourhood may contain LATER val edges),
p0_{multi_gnn,mega_gnn}_causal__see_everything.parquet (PyG `time_attr`: only edges at or before the seed edge) and the GFP scores.  Same checkpoints,
same calibration rule (the calibrator is refit on the val `fit` part of each score file).  Val only.  Writes data/runs/gnn_causal_vs_noncausal_hi_medium.json.
Per arm: n, positives, Brier, PR-AUC, log loss, AUC, ECE, calibration slope, precision/recall/F1 at the arm's own best-`fit`-F1 threshold; paired cluster
bootstrap (attempt + shared accounts, hub cap 20, 2,000 resamples) of Brier(causal) - Brier(non-causal) and of each - GFP on the common rows.
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
from ftm.eval import metrics as mt  # noqa: E402

DS = "hi_medium"
ARMS = {"multi_gnn": "p0_multi_gnn__see_everything.parquet", "multi_gnn_causal": "p0_multi_gnn_causal__see_everything.parquet",
        "mega_gnn": "p0_mega_gnn__see_everything.parquet", "mega_gnn_causal": "p0_mega_gnn_causal__see_everything.parquet"}  # fmt: skip


def usable(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter((pl.col("mode") == "see_everything") & pl.col("p0").is_not_null() & pl.col("p0").is_not_nan())


def full_stats(df: pl.DataFrame) -> dict:
    from sklearn.metrics import log_loss, roc_auc_score

    df = usable(df)
    fit, ev = df.filter(pl.col("val_part") == "fit"), df.filter(pl.col("val_part") == "eval")
    thr = g.metrics(fit["label"].cast(pl.Int8).to_numpy(), fit["p0"].to_numpy())["thr"]
    y, p = ev["label"].cast(pl.Int8).to_numpy(), ev["p0"].to_numpy()
    m = g.metrics(y, p, thr)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    lg = np.log(pc / (1 - pc))
    from sklearn.linear_model import LogisticRegression

    slope = float(LogisticRegression(C=1e6, max_iter=500).fit(lg.reshape(-1, 1), y).coef_[0, 0]) if 0 < y.sum() < len(y) else float("nan")  # logistic slope on the logit (1 = calibrated)
    return {**{k: m.get(k) for k in ("n", "n_pos", "brier", "pr_auc", "f1", "precision", "recall")}, "log_loss": float(log_loss(y, pc, labels=[0, 1])),
            "auc": float(roc_auc_score(y, p)), "ece": mt.ece(y, p), "calibration_slope": slope, "thr_fit_f1": thr}  # fmt: skip


def eval_col(df: pl.DataFrame, name: str) -> pl.DataFrame:
    d = usable(df).filter(pl.col("val_part") == "eval")
    return d.select("tx_id", "label", pl.col("p0").alias(name)).unique("tx_id")


def main() -> None:
    cfg = io.load_config(DS)
    d = Path("data/scores") / DS
    frames = {k: pl.read_parquet(d / f) for k, f in ARMS.items() if (d / f).exists()}  # time-guard: not transactions
    frames["gfp_lgbm"] = g.read_p0(DS, split="val")
    out: dict = {"dataset": DS, "arms": {k: full_stats(v) for k, v in frames.items()}}
    a = None
    for k, v in frames.items():
        c = eval_col(v, "p_" + k)
        a = c if a is None else a.join(c.drop("label"), on="tx_id")
    pq = cfg["parquet_dir"]
    tx = io.scan_transactions(pq, split="val", columns=["tx_id", "src_id", "dst_id"]).collect()
    pat = io.scan_patterns(pq, split="val").select("tx_id", "attempt_id").collect()
    j = a.join(tx, on="tx_id", how="left").join(pat, on="tx_id", how="left").sort("tx_id")
    cl = bs.cluster_ids(j["attempt_id"].to_list(), [j["src_id"].to_list(), j["dst_id"].to_list()])
    y = j["label"].cast(pl.Int8).to_numpy()
    out["common_n"], out["common_pos"] = len(y), int(y.sum())
    out["common_brier"] = {k: float(np.mean((j["p_" + k].to_numpy() - y) ** 2)) for k in frames}
    pairs = [(f"{m}_causal", m) for m in ("multi_gnn", "mega_gnn") if f"{m}_causal" in frames] + [("gfp_lgbm", m) for m in frames if m != "gfp_lgbm"]
    for x, z in pairs:  # Brier(x) - Brier(z); negative = x better
        r = bs.brier_diff(y, j["p_" + x].to_numpy(), j["p_" + z].to_numpy(), cl, n_boot=2000)
        out.setdefault("brier_diff_x_minus_z", {})[f"{x}_minus_{z}"] = {"diff": r["diff"], "ci95": r["ci"]}
    Path("data/runs").mkdir(parents=True, exist_ok=True)
    Path(f"data/runs/gnn_causal_vs_noncausal_{DS}.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
