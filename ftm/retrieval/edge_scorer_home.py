"""C11 edge scorer v2 (finding 18): per-home-bank features, out-of-fold train scores.

Owner: T1 (Abhishek).  Spec: tracks.md 3.8.

v1 (`edge_scorer.py`) computes the 24 h endpoint aggregates and the prior counts over ALL banks' transactions, but the evidence view of an alert is the
home bank's own ledger, so the scorer saw flows the home bank never sees; and its train-edge scores are in-sample (the model was fit on them).
v2 scores an edge FROM A HOME BANK'S VIEW: the features are `FEATURES2` computed on View(bank) alone (edges with from_bank or to_bank == bank; an
account at another bank shows only its flows with the home bank) plus `home_side` (1 the home bank sends, 2 it receives, 3 both). One pooled MLP over
all banks' rows, same architecture / loss / negative subsampling as v1. Scores are keyed (tx_id, bank): a transaction between two top-30 banks has a
score in each view. Train edges get out-of-fold scores (3 day blocks of the train split, each scored by a model fit on the other two for the same
number of epochs as the full model); val edges the full model, whose epoch is picked on val `fit` AP (as v1). Nothing after the edge enters its features.

  python -m ftm.retrieval.edge_scorer_home --data hi_medium --threads 6      # CPU, ~15 min HI-Medium, writes edge_scores_home_v2.parquet
Evidence views use it with ftm.retrieval.evidence_views `scorer_version="v2"` (or FTM_EDGE_SCORER=v2).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from ftm.data import io
from ftm.retrieval.edge_scorer import COLS, FEATURES2, timeline_features

FEATURES_HOME = FEATURES2 + ["home_side"]


def home_features(frame: pl.DataFrame, bank: int, codes: dict | None = None) -> tuple[pl.DataFrame, dict]:
    """`FEATURES_HOME` of every edge of View(bank) (`frame` = that view only, any order), from the view alone."""
    feats, codes = timeline_features(frame, codes)
    side = frame.sort("tx_id").select("tx_id", ((pl.col("from_bank") == bank).cast(pl.Int8) + 2 * (pl.col("to_bank") == bank).cast(pl.Int8)).alias("home_side"))
    return feats.join(side, on="tx_id").sort("tx_id"), codes


def banks_of(cfg: dict) -> list[int]:
    return sorted(int(f.stem.split("=")[1]) for f in Path(cfg["views_dir"]).glob("bank=*.parquet"))


def build_table(cfg: dict, banks: list[int] | None = None, splits=("train", "val"), codes: dict | None = None):
    """Stack the per-bank feature tables: (X float32 [n, F], meta frame with tx_id, bank, split, timestamp, day, y), codes."""
    pq = cfg["parquet_dir"]
    pat = io.scan_patterns(pq, split=list(splits)).select("tx_id").with_columns(pl.lit(1).alias("y")).collect()
    xs, metas = [], []
    for b in banks or banks_of(cfg):
        fr = io.scan_view(cfg["views_dir"], b, split=list(splits), columns=COLS + ["split", "day"]).collect()
        if fr.is_empty():
            continue
        if codes is None:
            cats = ("receiving_currency", "payment_currency", "payment_format")
            codes = {c: {v: i for i, v in enumerate(sorted(io.scan_transactions(pq, split="train", columns=[c]).select(pl.col(c).unique()).collect()[c].to_list()))} for c in cats}
        feats, _ = home_features(fr, b, codes)
        meta = fr.sort("tx_id").select("tx_id", pl.lit(b).cast(pl.Int64).alias("bank"), "split", "timestamp", "day").join(pat, on="tx_id", how="left").with_columns(pl.col("y").fill_null(0))
        assert meta["tx_id"].to_list() == feats["tx_id"].to_list()
        xs.append(feats.select(FEATURES_HOME).to_numpy().astype("f4"))
        metas.append(meta)
    return np.vstack(xs), pl.concat(metas), codes


def day_folds(days: np.ndarray, d0: int, d1: int, k: int = 3) -> np.ndarray:
    """Block index 0..k-1 of train days d0..d1 (contiguous, as equal as possible); -1 for days outside."""
    edges = np.linspace(d0 - 1, d1, k + 1)
    f = np.clip(np.searchsorted(edges, days - 0.5), 1, k) - 1
    return np.where((days >= d0) & (days <= d1), f, -1)


def _net(n_in: int):
    import torch

    return torch.nn.Sequential(torch.nn.Linear(n_in, 128), torch.nn.ReLU(), torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1))


def _predict(net, Xn: np.ndarray) -> np.ndarray:
    import torch

    net.eval()
    with torch.no_grad():
        return torch.cat([net(torch.tensor(Xn[i : i + 500_000])).squeeze(1) for i in range(0, len(Xn), 500_000)]).numpy()


def fit_net(Xn, y, epochs: int, neg_frac: float, seed: int, select=None):
    """Train the MLP on the rows (all positives + `neg_frac` of the negatives, weights restore the base rate). `select` = (X, y) of the val fit part:
    the epoch with the best AP there is kept (returns net, epoch, history); without it all `epochs` are run."""
    import torch
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    keep = np.flatnonzero((y == 1) | (rng.random(len(y)) < neg_frac))
    torch.manual_seed(seed)
    net = _net(Xn.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    xt, yt = torch.tensor(Xn[keep]), torch.tensor(y[keep], dtype=torch.float32)
    w = torch.where(yt == 1, 1.0, 1.0 / neg_frac)
    w = w / w.mean()
    best, hist = (-1.0, None, epochs - 1), []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(keep))
        for i in range(0, len(perm), 4096):
            b = perm[i : i + 4096]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt[b]).squeeze(1), yt[b], weight=w[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if select is not None:
            ap = float(average_precision_score(select[1], _predict(net, select[0])))
            hist.append({"epoch": ep, "val_fit_ap": ap})
            if ap > best[0]:
                best = (ap, {k: t.clone() for k, t in net.state_dict().items()}, ep)
    if select is not None:
        net.load_state_dict(best[1])
    return net, best[2], hist


def train_home(data: str = "hi_medium", seed: int = 0, epochs: int = 120, neg_frac: float = 0.1, threads: int = 6, banks: list[int] | None = None, suffix: str = "") -> dict:
    """Features per home bank, OOF train scores, full-model val scores. Writes data/retrieval/<data>/edge_scores_home_v2<suffix>.parquet,
    models/edge_scorer_home_<data><suffix>.pt and returns the report (also data/runs/edge_scorer_home_<data><suffix>.json)."""
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score

    from ftm.baselines.gfp_gbt import val_parts

    t0 = time.time()
    torch.set_num_threads(threads)
    cfg = io.load_config(data)
    X, meta, codes = build_table(cfg, banks)
    t_feat = time.time() - t0
    X = np.sign(X) * np.log1p(np.abs(X))
    sp, y, day = meta["split"].to_numpy(), meta["y"].to_numpy(), meta["day"].to_numpy()
    tr = sp == "train"
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = ((X - mu) / sd).astype("f4")
    vts = io.scan_transactions(cfg["parquet_dir"], split="val", columns=["timestamp"]).collect()["timestamp"]
    part = np.full(len(y), "", dtype="U4")
    vm = sp == "val"
    part[vm] = val_parts(meta["timestamp"].filter(pl.Series(vm)), vts.min(), vts.max())
    d0, d1 = cfg["split_days"]["train"]
    fold = day_folds(day, d0, d1)
    fit, ev = np.flatnonzero(part == "fit"), np.flatnonzero(part == "eval")
    net, best_ep, hist = fit_net(Xn[tr], y[tr], epochs, neg_frac, seed, select=(Xn[fit], y[fit]))
    score = np.full(len(y), np.nan)
    score[vm] = _predict(net, Xn[vm])
    for f in range(3):
        m_f = fold == f
        n_f, *_ = fit_net(Xn[tr & ~m_f], y[tr & ~m_f], best_ep + 1, neg_frac, seed + 1 + f)
        score[m_f] = _predict(n_f, Xn[m_f])
    prob = 0.5 * (1 + np.tanh(0.5 * score))  # stable sigmoid
    out = Path("data/retrieval") / data
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"tx_id": meta["tx_id"], "bank": meta["bank"], "score": prob, "split": meta["split"], "oof": tr}).write_parquet(out / f"edge_scores_home_v2{suffix}.parquet")
    Path("models").mkdir(exist_ok=True)
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "features": FEATURES_HOME, "codes": codes, "best_epoch": best_ep}, f"models/edge_scorer_home_{data}{suffix}.pt")
    rep = {"data": data, "banks": len(set(meta["bank"].to_list())), "rows": len(y), "best_epoch": best_ep, "val_fit_ap": max(h["val_fit_ap"] for h in hist), "seed": seed,
           "val_eval_ap": float(average_precision_score(y[ev], score[ev])), "val_eval_auc": float(roc_auc_score(y[ev], score[ev])), "val_eval_pos": int(y[ev].sum()), "val_eval_n": int(len(ev)),
           "train_oof_ap": float(average_precision_score(y[tr], score[tr])), "train_oof_auc": float(roc_auc_score(y[tr], score[tr])), "train_base_rate": float(y[tr].mean()),
           "feature_seconds": t_feat, "seconds": time.time() - t0, "features": FEATURES_HOME, "history": hist}  # fmt: skip
    v1 = out / "edge_scores.parquet"
    if v1.exists():  # v1 on the same (tx, bank) rows: global features, train rows in-sample
        old = pl.read_parquet(v1, columns=["tx_id", "score"]).rename({"score": "s1"})  # time-guard: not transactions
        j = meta.select("tx_id", "bank", "split", "y").with_columns(pl.Series("part", part)).join(old, on="tx_id", how="left")
        for name, m in (("val_eval", j["part"] == "eval"), ("train", j["split"] == "train")):
            jj = j.filter(m)
            rep[f"v1_{name}_ap"] = float(average_precision_score(jj["y"], jj["s1"]))
            rep[f"v1_{name}_auc"] = float(roc_auc_score(jj["y"], jj["s1"]))
    Path("data/runs").mkdir(exist_ok=True)
    Path(f"data/runs/edge_scorer_home_{data}{suffix}.json").write_text(json.dumps(rep, indent=1))
    return rep


def score_final_home(data: str, threads: int = 6) -> dict:
    """Week 14: score the TEST edges of every home bank's view with the val run's v2 model (nothing trained or tuned; mu/sd/codes from the checkpoint).
    The view's train+val rows are in the same pass (the features of a test edge need the earlier days) and the val scores must reproduce
    `edge_scores_home_v2.parquet` (max diff < 1e-5). Writes data/retrieval/<data>/final/edge_scores_home_v2_test.parquet. Locked: FTM_FINAL_RUN=1."""
    import torch

    from ftm import guards

    guards.require_final_run("the test edges of the home-bank edge scorer")
    torch.set_num_threads(threads)
    ck = torch.load(f"models/edge_scorer_home_{data}.pt", weights_only=False)
    cfg = io.load_config(data)
    net = _net(len(ck["features"]))
    net.load_state_dict(ck["state"])
    X, meta, _ = build_table(cfg, splits=("train", "val", "test"), codes=ck["codes"])
    X = np.sign(X) * np.log1p(np.abs(X))
    s = _predict(net, ((X - ck["mu"]) / ck["sd"]).astype("f4"))
    new = pl.DataFrame({"tx_id": meta["tx_id"], "bank": meta["bank"], "score": 0.5 * (1 + np.tanh(0.5 * s)), "split": meta["split"]})
    old = pl.read_parquet(Path("data/retrieval") / data / "edge_scores_home_v2.parquet").filter(pl.col("split") == "val")  # time-guard: not transactions
    j = old.join(new.filter(pl.col("split") == "val"), on=["tx_id", "bank"], suffix="_new")
    diff = float((j["score"] - j["score_new"]).abs().max())
    rec = {"data": data, "test_rows": int((new["split"] == "test").sum()), "val_rows_compared": j.height, "max_abs_val_score_diff": diff}
    if j.height != old.height or diff > 1e-5:
        raise SystemExit(f"the final pass does not reproduce the stored val scores ({rec}): do not use these test scores")
    out = Path("data/retrieval") / data / "final"
    out.mkdir(parents=True, exist_ok=True)
    new.filter(pl.col("split") == "test").write_parquet(out / "edge_scores_home_v2_test.parquet")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--banks", type=int, nargs="*", default=None, help="subset of banks (debugging)")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--final", action="store_true", help="week 14: TEST edges (needs FTM_FINAL_RUN=1)")
    a = ap.parse_args()
    r = score_final_home(a.data, a.threads) if a.final else train_home(a.data, epochs=a.epochs, threads=a.threads, banks=a.banks, suffix=a.suffix)
    print(json.dumps({k: v for k, v in r.items() if k != "history"}, indent=1))
