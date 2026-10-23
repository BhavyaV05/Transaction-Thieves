"""C11 edge scorer, data prep and a tiny baseline: "is this edge part of a laundering attempt?".

Owner: T1 (Abhishek).  Spec: tracks.md 3.8.  No PCST, no subgraph retrieval here.

Training data comes from the TRAIN split only (ftm.data.io); val is used to report. Label = the
transaction has a row in patterns.parquet (it belongs to a pattern-file attempt). Laundering transactions
that are not in the pattern file get label 0: the scorer predicts attempt membership, not the raw label.
Features use only the edge itself and past-only counts (how many transactions the sender sent and the
receiver received before this one in time order), so they are valid at any `as_of`.
Run: python -m ftm.retrieval.edge_scorer --data hi_small            # the tiny sklearn baseline
     python -m ftm.retrieval.edge_scorer --data hi_small --train    # M3: torch MLP over edge + node-neighbourhood features

M3 (`train_scorer`): features are the edge's own attributes, past-only prior counts, and **past-24h aggregates of the two
endpoints and of their neighbours' flows** (what the sender received and sent, what the receiver received and sent in
[t - 24h, t), strictly before the edge). They depend only on edges before the edge's own timestamp, so they are the same
at any `as_of` and can be computed once over the train+val timeline. The model is trained on train edges only and selected
on the val `fit` part (AP); val `eval` is for reporting. Scores of train edges are in-sample.
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


AGG = ["src_out_n", "src_out_amt", "src_in_n", "src_in_amt", "dst_out_n", "dst_out_amt", "dst_in_n", "dst_in_amt", "src_passthru", "dst_passthru"]  # fmt: skip
FEATURES2 = FEATURES + AGG


def timeline_features(frame: pl.DataFrame, codes: dict | None = None, window: str = "24h") -> tuple[pl.DataFrame, dict]:
    """`FEATURES2` for a train+val frame (any order; sorted here). Every feature uses only edges strictly before the row's timestamp
    (prior counts: before it in (timestamp, tx_id) order; 24h aggregates: timestamps in [t - 24h, t)), so no later edge is read."""
    base, codes = edge_features(frame, codes)
    f = frame.sort(["timestamp", "tx_id"])
    ev = pl.concat([
        f.select("tx_id", "timestamp", pl.col("src_id").alias("n"), pl.lit(0).alias("role"), pl.col("amount_paid").alias("a")),
        f.select("tx_id", "timestamp", pl.col("dst_id").alias("n"), pl.lit(1).alias("role"), pl.col("amount_received").alias("a")),
    ]).sort(["timestamp", "tx_id", "role"])  # fmt: skip
    # per node: past outgoing (role 0) and incoming (role 1) count and amount in the window
    out = ev.with_columns(
        pl.when(pl.col("role") == 0).then(1.0).otherwise(0.0).alias("o_n"), pl.when(pl.col("role") == 0).then(pl.col("a")).otherwise(0.0).alias("o_a"),
        pl.when(pl.col("role") == 1).then(1.0).otherwise(0.0).alias("i_n"), pl.when(pl.col("role") == 1).then(pl.col("a")).otherwise(0.0).alias("i_a"),
    ).with_columns(
        *[pl.col(c).rolling_sum_by("timestamp", window_size=window, closed="left").over("n").alias(c + "_w") for c in ("o_n", "o_a", "i_n", "i_a")]
    )  # fmt: skip
    cols = ["o_n_w", "o_a_w", "i_n_w", "i_a_w"]
    s = out.filter(pl.col("role") == 0).select("tx_id", *[pl.col(c).alias("s_" + c) for c in cols])
    d = out.filter(pl.col("role") == 1).select("tx_id", *[pl.col(c).alias("d_" + c) for c in cols])
    j = base.join(s, on="tx_id").join(d, on="tx_id")
    lg = lambda c: pl.col(c).clip(lower_bound=0).log1p()  # noqa: E731
    j = j.with_columns(
        lg("s_o_n_w").alias("src_out_n"), lg("s_o_a_w").alias("src_out_amt"), lg("s_i_n_w").alias("src_in_n"), lg("s_i_a_w").alias("src_in_amt"),
        lg("d_o_n_w").alias("dst_out_n"), lg("d_o_a_w").alias("dst_out_amt"), lg("d_i_n_w").alias("dst_in_n"), lg("d_i_a_w").alias("dst_in_amt"),
        (pl.col("s_i_a_w") / (pl.col("s_o_a_w") + pl.col("s_i_a_w") + 1.0)).alias("src_passthru"),
        (pl.col("d_o_a_w") / (pl.col("d_o_a_w") + pl.col("d_i_a_w") + 1.0)).alias("dst_passthru"),
    )  # fmt: skip
    return j.select("tx_id", *FEATURES2).sort("tx_id"), codes


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


def train_scorer(data: str = "hi_small", seed: int = 0, epochs: int = 30, neg_frac: float = 0.1, device: str = "cpu") -> dict:
    """M3: MLP (2 x 128) on `FEATURES2`. Train edges only (all attempt edges + `neg_frac` of the rest, weights restore the base rate);
    epoch picked by AP on val `fit`; metrics on val `eval`. Writes models/edge_scorer_<data>.pt and
    data/retrieval/<data>/edge_scores.parquet (tx_id, score for every train+val edge; `score` of train edges is in-sample)."""
    from pathlib import Path

    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score

    from ftm.baselines.gfp_gbt import val_parts

    cfg = io.load_config(data)
    pq = cfg["parquet_dir"]
    fr = io.scan_transactions(pq, split=["train", "val"], columns=COLS + ["split"]).collect()
    feats, codes = timeline_features(fr)
    lab = io.scan_patterns(pq, split=["train", "val"]).select("tx_id").with_columns(pl.lit(1).alias("y")).collect()
    meta = fr.select("tx_id", "split", "timestamp").join(lab, on="tx_id", how="left").with_columns(pl.col("y").fill_null(0)).sort("tx_id")
    assert meta["tx_id"].to_list() == feats["tx_id"].to_list()
    X = feats.select(FEATURES2).to_numpy().astype("f4")
    y, sp, ts = meta["y"].to_numpy(), meta["split"].to_numpy(), meta["timestamp"]
    v = ts.filter(pl.Series(sp == "val"))
    part = np.full(len(y), "", dtype="U4")
    part[sp == "val"] = val_parts(v, v.min(), v.max())
    X = np.sign(X) * np.log1p(np.abs(X))  # heavy tails
    tr = np.flatnonzero(sp == "train")
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = ((X - mu) / sd).astype("f4")
    rng = np.random.default_rng(seed)
    keep = tr[(y[tr] == 1) | (rng.random(len(tr)) < neg_frac)]
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 128), torch.nn.ReLU(), torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    xt, yt = torch.tensor(Xn[keep]).to(device), torch.tensor(y[keep], dtype=torch.float32).to(device)
    w = torch.where(yt == 1, 1.0, 1.0 / neg_frac).to(device)
    w = w / w.mean()

    def predict(idx):
        net.eval()
        with torch.no_grad():
            return torch.cat([net(torch.tensor(Xn[idx[i : i + 500_000]]).to(device)).squeeze(1).cpu() for i in range(0, len(idx), 500_000)]).numpy()

    fit, ev = np.flatnonzero(part == "fit"), np.flatnonzero(part == "eval")
    best, hist = (-1.0, None), []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(keep), device=device)
        for i in range(0, len(perm), 4096):
            b = perm[i : i + 4096]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt[b]).squeeze(1), yt[b], weight=w[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        ap = float(average_precision_score(y[fit], predict(fit)))
        hist.append({"epoch": ep, "val_fit_ap": ap})
        if ap > best[0]:
            best = (ap, {k: t.clone() for k, t in net.state_dict().items()}, ep)
    net.load_state_dict(best[1])
    s_all = predict(np.arange(len(y)))
    Path("models").mkdir(exist_ok=True)
    torch.save({"state": best[1], "mu": mu, "sd": sd, "features": FEATURES2, "codes": codes, "best_epoch": best[2]}, f"models/edge_scorer_{data}.pt")
    out = Path("data/retrieval") / data
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"tx_id": meta["tx_id"], "score": 1 / (1 + np.exp(-s_all.astype("f8"))), "split": meta["split"]}).write_parquet(out / "edge_scores.parquet")
    return {"data": data, "best_epoch": best[2], "val_fit_ap": best[0], "val_eval_ap": float(average_precision_score(y[ev], s_all[ev])),
            "val_eval_auc": float(roc_auc_score(y[ev], s_all[ev])), "val_eval_pos": int(y[ev].sum()), "val_eval_n": int(len(ev)),
            "val_eval_base_rate": float(y[ev].mean()), "n_train_used": int(len(keep)), "history": hist, "seed": seed, "features": FEATURES2}  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=120)
    a = ap.parse_args()
    print(json.dumps(train_scorer(a.data, epochs=a.epochs, device=a.device) if a.train else baseline(a.data), indent=1))
