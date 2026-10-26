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


def chunked_timeline_features(scan: pl.LazyFrame, codes: dict, n_nodes: int, window: str = "24h"):
    """Yield `(day, FEATURES2 frame of that day's rows)`: the same values as `timeline_features` over the whole frame, one day at
    a time, for datasets that do not fit in memory (HI-Medium). `scan` has COLS + `day`, train/val rows only.

    A day's 24 h windows reach back at most into the previous day, so the slab is {day - 1, day}; the prior counts (how often a
    node appeared before) are carried across slabs in two dense counters over the node ids (`n_nodes`), which hold the days up to
    day - 2. Equality with the whole-frame computation is tested (tests/test_retrieval_scorer.py).
    """
    days = sorted(scan.select(pl.col("day").unique()).collect()["day"].to_list())
    out_cnt, in_cnt = np.zeros(n_nodes, np.int64), np.zeros(n_nodes, np.int64)
    pending: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # day -> (src, dst) not yet in the counters
    for d in days:
        for old in [x for x in pending if x <= d - 2]:
            src, dst = pending.pop(old)
            out_cnt += np.bincount(src, minlength=n_nodes)
            in_cnt += np.bincount(dst, minlength=n_nodes)
        slab = scan.filter(pl.col("day").is_in([d - 1, d])).collect()
        feats, _ = timeline_features(slab, codes, window)
        meta = slab.sort("tx_id").select("tx_id", "src_id", "dst_id", "day")
        assert feats["tx_id"].to_list() == meta["tx_id"].to_list()
        s, t = meta["src_id"].to_numpy(), meta["dst_id"].to_numpy()
        feats = feats.with_columns(
            pl.col("src_prior_out") + pl.Series(out_cnt[s]), pl.col("dst_prior_in") + pl.Series(in_cnt[t]),
            pl.col("src_prior_in") + pl.Series(in_cnt[s]), pl.col("dst_prior_out") + pl.Series(out_cnt[t]),
        )  # fmt: skip
        today = (meta["day"] == d).to_numpy()
        pending[d] = (s[today], t[today])
        yield d, feats.filter(pl.Series(today))
        del slab, feats


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


def train_scorer_streaming(data: str = "hi_medium", seed: int = 0, epochs: int = 120, neg_frac: float = 0.1, device: str = "cpu",
                           threads: int = 6, keep_chunks: bool = False, suffix: str = "") -> dict:
    """`train_scorer` for datasets that do not fit in memory (HI-Medium): features one day at a time (`chunked_timeline_features`,
    equal to the whole-frame values), negatives subsampled while streaming (all attempt edges + `neg_frac` of the other train edges
    are kept in memory, plus the val `fit` rows for the epoch choice), then every train+val edge is scored chunk by chunk.
    Same model, loss, epoch rule and outputs as `train_scorer` (`suffix` is added to the output file names, for checks);
    `peak_rss_mb` is the process high-water mark (ru_maxrss)."""
    import resource
    import shutil
    import time
    from pathlib import Path

    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score

    from ftm.baselines.gfp_gbt import val_parts

    t0 = time.time()
    rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # noqa: E731  (MB on Linux)
    torch.set_num_threads(threads)
    cfg = io.load_config(data)
    pq = cfg["parquet_dir"]
    scan = io.scan_transactions(pq, split=["train", "val"], columns=COLS + ["split", "day"])
    cats = ("receiving_currency", "payment_currency", "payment_format")
    codes = {c: {v: i for i, v in enumerate(sorted(scan.select(pl.col(c).unique()).collect()[c].to_list()))} for c in cats}
    mx = scan.select(pl.max_horizontal(pl.col("src_id").max(), pl.col("dst_id").max())).collect().item()
    vt = scan.filter(pl.col("split") == "val").select(pl.col("timestamp").min().alias("a"), pl.col("timestamp").max().alias("b")).collect()
    att = io.scan_patterns(pq, split=["train", "val"]).select("tx_id").collect()["tx_id"].to_numpy()
    out = Path("data/retrieval") / data
    chunks = out / f"chunks{suffix}"
    chunks.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_tr = 0
    sx = sxx = 0.0
    keep_x, keep_y, fit_x, fit_y, stages = [], [], [], [], {}
    for d, feats in chunked_timeline_features(scan, codes, int(mx) + 1):
        m = scan.filter(pl.col("day") == d).select("tx_id", "split", "timestamp").collect().sort("tx_id")
        assert m["tx_id"].to_list() == feats["tx_id"].to_list()
        X = feats.select(FEATURES2).to_numpy().astype("f4")
        X = np.sign(X) * np.log1p(np.abs(X))  # heavy tails
        y = np.isin(m["tx_id"].to_numpy(), att).astype("i1")
        sp = np.array(m["split"].to_list(), dtype="U5")
        part = np.full(len(y), "", dtype="U4")
        isv = sp == "val"
        if isv.any():
            part[isv] = val_parts(m["timestamp"].filter(pl.Series(isv)), vt["a"][0], vt["b"][0])
        np.savez(chunks / f"day{d:02d}.npz", X=X, y=y, tx=m["tx_id"].to_numpy(), sp=sp, part=part)
        tr = sp == "train"
        n_tr += int(tr.sum())
        sx = sx + X[tr].astype("f8").sum(0)
        sxx = sxx + (X[tr].astype("f8") ** 2).sum(0)
        k = tr & ((y == 1) | (rng.random(len(y)) < neg_frac))
        keep_x.append(X[k])
        keep_y.append(y[k])
        f = part == "fit"
        fit_x.append(X[f])
        fit_y.append(y[f])
        stages[f"features_day{d}"] = {"rows": len(y), "peak_rss_mb": rss(), "seconds": time.time() - t0}
        del feats, X
    mu = sx / n_tr
    sd = np.sqrt(np.maximum(sxx / n_tr - mu**2, 0)) + 1e-6
    norm = lambda a: ((a - mu) / sd).astype("f4")  # noqa: E731
    xt, yt = torch.tensor(norm(np.vstack(keep_x))).to(device), torch.tensor(np.concatenate(keep_y), dtype=torch.float32).to(device)
    xf, yf = norm(np.vstack(fit_x)), np.concatenate(fit_y)
    del keep_x, fit_x
    w = torch.where(yt == 1, 1.0, 1.0 / neg_frac).to(device)
    w = w / w.mean()
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], 128), torch.nn.ReLU(), torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)

    def predict(Xn):
        net.eval()
        with torch.no_grad():
            return torch.cat([net(torch.tensor(Xn[i : i + 500_000]).to(device)).squeeze(1).cpu() for i in range(0, len(Xn), 500_000)]).numpy()

    stages["streamed_features"] = {"train_rows": n_tr, "kept_train": int(len(yt)), "kept_pos": int(yt.sum().item()), "val_fit_rows": int(len(yf)),
                                   "peak_rss_mb": rss(), "seconds": time.time() - t0}  # fmt: skip
    best, hist = (-1.0, None, -1), []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(yt), device=device)
        for i in range(0, len(perm), 4096):
            b = perm[i : i + 4096]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt[b]).squeeze(1), yt[b], weight=w[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        ap = float(average_precision_score(yf, predict(xf)))
        hist.append({"epoch": ep, "val_fit_ap": ap})
        if ap > best[0]:
            best = (ap, {k: t.clone() for k, t in net.state_dict().items()}, ep)
    net.load_state_dict(best[1])
    stages["trained"] = {"peak_rss_mb": rss(), "seconds": time.time() - t0}
    del xt, yt, xf
    ev_y, ev_s, frames = [], [], []
    for f in sorted(chunks.glob("day*.npz")):  # score every train+val edge, one day at a time
        z = np.load(f)
        s_all = predict(norm(z["X"]))
        frames.append(pl.DataFrame({"tx_id": z["tx"], "score": 1 / (1 + np.exp(-s_all.astype("f8"))), "split": z["sp"]}))
        e = z["part"] == "eval"
        ev_y.append(z["y"][e])
        ev_s.append(s_all[e])
    Path("models").mkdir(exist_ok=True)
    torch.save({"state": best[1], "mu": mu, "sd": sd, "features": FEATURES2, "codes": codes, "best_epoch": best[2]}, f"models/edge_scorer_{data}{suffix}.pt")
    pl.concat(frames).sort("tx_id").write_parquet(out / f"edge_scores{suffix}.parquet")
    ev_y, ev_s = np.concatenate(ev_y), np.concatenate(ev_s)
    stages["scored"] = {"peak_rss_mb": rss(), "seconds": time.time() - t0}
    if not keep_chunks:
        shutil.rmtree(chunks)
    return {"data": data, "streaming": True, "best_epoch": best[2], "val_fit_ap": best[0], "val_eval_ap": float(average_precision_score(ev_y, ev_s)),
            "val_eval_auc": float(roc_auc_score(ev_y, ev_s)), "val_eval_pos": int(ev_y.sum()), "val_eval_n": int(len(ev_y)),
            "val_eval_base_rate": float(ev_y.mean()), "neg_frac": neg_frac, "seed": seed, "features": FEATURES2, "peak_rss_mb": rss(),
            "seconds": time.time() - t0, "stages": stages, "history": hist}  # fmt: skip


def score_final(data: str, threads: int = 6) -> dict:
    """Week 14: score the TEST edges with the val run's model (`models/edge_scorer_<data>.pt`; nothing is trained or tuned).

    Features are the past-only `FEATURES2` over the train+val+test timeline, one day at a time (`chunked_timeline_features`, equal to the
    whole-frame values), with the category codes, mean and sd stored in the checkpoint (so appended test rows cannot shift them). The val
    edges of the same pass must reproduce `data/retrieval/<data>/edge_scores.parquet` (max difference recorded; must be below 1e-5).
    Writes data/retrieval/<data>/final/edge_scores_test.parquet (tx_id, score, split) and data/runs/final_edge_scorer_<data>.json.
    Locked: needs FTM_FINAL_RUN=1 (scripts/run_final.py)."""
    import time
    from pathlib import Path

    import torch

    from ftm import guards

    guards.require_final_run("the test edges of the edge scorer")
    t0 = time.time()
    torch.set_num_threads(threads)
    ck = torch.load(f"models/edge_scorer_{data}.pt", weights_only=False)
    cfg = io.load_config(data)
    scan = io.scan_transactions(cfg["parquet_dir"], split=["train", "val", "test"], columns=COLS + ["split", "day"])
    mx = scan.select(pl.max_horizontal(pl.col("src_id").max(), pl.col("dst_id").max())).collect().item()
    net = torch.nn.Sequential(torch.nn.Linear(len(ck["features"]), 128), torch.nn.ReLU(), torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1))
    net.load_state_dict(ck["state"])
    net.eval()
    mu, sd = ck["mu"], ck["sd"]
    frames = []
    for d, feats in chunked_timeline_features(scan, ck["codes"], int(mx) + 1):
        m = scan.filter(pl.col("day") == d).select("tx_id", "split").collect().sort("tx_id")
        assert m["tx_id"].to_list() == feats["tx_id"].to_list()
        sp = np.array(m["split"].to_list(), dtype="U5")
        keep = sp != "train"  # val rows are only a regression check; train rows are not scored again
        if not keep.any():
            continue
        X = feats.select(FEATURES2).to_numpy().astype("f4")[keep]
        X = np.sign(X) * np.log1p(np.abs(X))
        with torch.no_grad():
            z = net(torch.tensor(((X - mu) / sd).astype("f4"))).squeeze(1).numpy()
        frames.append(pl.DataFrame({"tx_id": m["tx_id"].to_numpy()[keep], "score": 1 / (1 + np.exp(-z.astype("f8"))), "split": sp[keep]}))
    new = pl.concat(frames).sort("tx_id")
    old = pl.read_parquet(Path("data/retrieval") / data / "edge_scores.parquet").filter(pl.col("split") == "val")  # time-guard: not transactions
    j = old.join(new.filter(pl.col("split") == "val"), on="tx_id", suffix="_new")
    diff = float((j["score"] - j["score_new"]).abs().max())
    out = Path("data/retrieval") / data / "final"
    out.mkdir(parents=True, exist_ok=True)
    new.filter(pl.col("split") == "test").write_parquet(out / "edge_scores_test.parquet")
    rec = {"data": data, "test_edges": int((new["split"] == "test").sum()), "val_edges_compared": j.height, "max_abs_val_score_diff": diff, "seconds": time.time() - t0}
    Path(f"data/runs/final_edge_scorer_{data}.json").write_text(json.dumps(rec, indent=1))
    if j.height != old.height or diff > 1e-5:
        raise SystemExit(f"the final pass does not reproduce the stored val scores ({rec}): do not use these test scores")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--streaming", action="store_true", help="day-by-day features and streamed negative subsampling (HI-Medium)")
    ap.add_argument("--suffix", default="", help="output file name suffix (streaming only; use it for checks)")
    ap.add_argument("--final", action="store_true", help="week 14: score the TEST edges with the val run's model (needs FTM_FINAL_RUN=1)")
    a = ap.parse_args()
    if a.final:
        print(json.dumps(score_final(a.data), indent=1))
    elif a.train and a.streaming:
        res = train_scorer_streaming(a.data, epochs=a.epochs, device=a.device, suffix=a.suffix)
    else:
        res = train_scorer(a.data, epochs=a.epochs, device=a.device) if a.train else baseline(a.data)
    print(json.dumps(res, indent=1))
