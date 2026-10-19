"""C11 edge scorer, data prep and a tiny baseline: "is this edge part of a laundering attempt?".

Owner: T1 (Abhishek).  Spec: tracks.md 3.8.  No PCST, no subgraph retrieval here.

Training data comes from the TRAIN split only (ftm.data.io); val is used to report. Label = the
transaction has a row in patterns.parquet (it belongs to a pattern-file attempt). Laundering transactions
that are not in the pattern file get label 0: the scorer predicts attempt membership, not the raw label.
Features use only the edge itself and past-only counts (how many transactions the sender sent and the
receiver received before this one in time order), so they are valid at any `as_of`.
Run: python -m ftm.retrieval.edge_scorer --data hi_small
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from ftm.data import io

FEATURES = ["log_amount", "ratio_paid_received", "cur_recv", "cur_pay", "fmt", "same_bank", "hour",
            "src_prior_out", "dst_prior_in", "src_prior_in", "dst_prior_out"]  # fmt: skip
COLS = ["tx_id", "timestamp", "from_bank", "to_bank", "src_id", "dst_id", "amount_received", "amount_paid",
        "receiving_currency", "payment_currency", "payment_format"]  # fmt: skip


def edge_features(frame: pl.DataFrame, codes: dict | None = None) -> tuple[pl.DataFrame, dict]:
    """Per-edge features for one split's rows (sorted by time); `codes` from the train split for categoricals."""
    f = frame.sort(["timestamp", "tx_id"])
    cats = ("receiving_currency", "payment_currency", "payment_format")
    codes = codes or {c: {v: i for i, v in enumerate(sorted(f[c].unique().to_list()))} for c in cats}
    out = f.select(
        "tx_id",
        pl.col("amount_received").clip(lower_bound=0).log1p().alias("log_amount"),
        (pl.col("amount_paid") / (pl.col("amount_received") + 1e-9)).alias("ratio_paid_received"),
        *[pl.col(c).replace_strict(codes[c], default=-1, return_dtype=pl.Int16).alias(n)
          for c, n in zip(cats, ("cur_recv", "cur_pay", "fmt"), strict=True)],
        (pl.col("from_bank") == pl.col("to_bank")).cast(pl.Int8).alias("same_bank"),
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("tx_id").cum_count().over("src_id").alias("src_prior_out"),
        pl.col("tx_id").cum_count().over("dst_id").alias("dst_prior_in"),
    )  # fmt: skip
    # prior counts of the opposite role: how many times the node appeared as receiver (sender) before this row
    role = pl.concat([f.select("tx_id", pl.col("timestamp"), pl.col("src_id").alias("n"), pl.lit(0).alias("r")),
                      f.select("tx_id", pl.col("timestamp"), pl.col("dst_id").alias("n"), pl.lit(1).alias("r"))])  # fmt: skip
    role = role.sort(["timestamp", "tx_id", "r"]).with_columns(
        (pl.col("r").cum_sum().over("n") - pl.col("r")).alias("n_prior_in"),
        ((1 - pl.col("r")).cum_sum().over("n") - (1 - pl.col("r"))).alias("n_prior_out"),
    )  # fmt: skip
    s = role.filter(pl.col("r") == 0).select("tx_id", pl.col("n_prior_in").alias("src_prior_in"))
    d = role.filter(pl.col("r") == 1).select("tx_id", pl.col("n_prior_out").alias("dst_prior_out"))
    return out.join(s, on="tx_id").join(d, on="tx_id").sort("tx_id"), codes


def load_split(cfg: dict, split: str, codes: dict | None = None):
    """(X float32, y int, tx_id) for a train/val split. Reads only through ftm.data.io."""
    fr = io.scan_transactions(cfg["parquet_dir"], split=split, columns=COLS).collect()
    feats, codes = edge_features(fr, codes)
    lab = io.scan_patterns(cfg["parquet_dir"], split=split).select("tx_id").with_columns(pl.lit(1).alias("y")).collect()
    feats = feats.join(lab, on="tx_id", how="left").with_columns(pl.col("y").fill_null(0))
    return feats.select(FEATURES).to_numpy().astype("f4"), feats["y"].to_numpy(), feats["tx_id"].to_numpy(), codes


def baseline(data: str = "hi_small", neg_frac: float = 0.05, seed: int = 0) -> dict:
    """Tiny MLP (one hidden layer of 32) on train, average precision and ROC AUC on val."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    cfg = io.load_config(data)
    Xtr, ytr, _, codes = load_split(cfg, "train")
    Xva, yva, _, _ = load_split(cfg, "val", codes)
    rng = np.random.default_rng(seed)
    keep = (ytr == 1) | (rng.random(len(ytr)) < neg_frac)
    sc = StandardScaler().fit(np.log1p(np.abs(Xtr[keep])) * np.sign(Xtr[keep]))
    t = lambda X: sc.transform(np.log1p(np.abs(X)) * np.sign(X))  # noqa: E731
    m = MLPClassifier((32,), max_iter=60, early_stopping=True, random_state=seed).fit(t(Xtr[keep]), ytr[keep])
    p = m.predict_proba(t(Xva))[:, 1]
    return {"data": data, "n_train": int(len(ytr)), "train_pos": int(ytr.sum()), "n_val": int(len(yva)),
            "val_pos": int(yva.sum()), "val_ap": float(average_precision_score(yva, p)),
            "val_auc": float(roc_auc_score(yva, p)), "val_base_rate": float(yva.mean())}  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hi_small")
    print(json.dumps(baseline(ap.parse_args().data), indent=1))
