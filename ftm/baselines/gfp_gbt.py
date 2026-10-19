"""C1: GFP features + LightGBM, one-bank (per bank view) and see-everything. Check F1 63.23 on HI-Small first.

Owner: T1 (Abhishek). See tracks.md section 3 and ftm/baselines/README.md.

Leakage rules (the README repeats them):
  - snapml GraphFeaturePreprocessor is a streaming graph. Edges in earlier `transform` calls are
    history (past only); edges inside ONE call are inserted together, so an edge also sees the
    other edges of its own batch, including later ones (checked on a toy graph). We therefore feed
    time-sorted batches of 128 edges (the GFP paper's inference batch size) and never mix splits in
    one batch. An edge sees all earlier batches plus at most 127 later edges of the same split.
  - features of val rows use train history (earlier split); features of train rows never use val.
    Test rows are never passed in: `build_features` raises on them without FTM_FINAL_RUN=1.
  - GFP columns that depend on the absolute time (mean/sum of the timestamp column) are dropped.
Heavy imports (snapml, lightgbm, sklearn) are inside functions so CI does not need them.
"""

from __future__ import annotations

import time
from functools import lru_cache

import numpy as np
import polars as pl

from ftm import guards

BATCH = 128
HOUR, DAY = 3600, 86400
# Settings of the GFP paper (Blanusa et al., KDD 2024, arXiv:2402.08593, section 3): one-day window
# for the vertex statistics and for fan, degree, temporal cycle and simple cycle features, six hours
# for scatter-gather, cycle length constraint 10, vertex statistics on the Timestamp and Amount
# columns. Bins (2, 3) are the library example's; the paper does not list them.
GFP_PARAMS = {
    "num_threads": 4,
    "vertex_stats": True, "vertex_stats_tw": DAY, "vertex_stats_cols": [3, 4],
    "vertex_stats_feats": [0, 1, 2, 3, 4, 8, 9, 10],
    "fan": True, "fan_tw": DAY, "fan_bins": [2, 3],
    "degree": True, "degree_tw": DAY, "degree_bins": [2, 3],
    "scatter-gather": True, "scatter-gather_tw": 6 * HOUR, "scatter-gather_bins": [2, 3],
    "temp-cycle": True, "temp-cycle_tw": DAY, "temp-cycle_bins": [2, 3],
    "lc-cycle": True, "lc-cycle_tw": DAY, "lc-cycle_len": 10, "lc-cycle_bins": [2, 3],
}  # fmt: skip
N_ECHO = 5  # snapml echoes the 5 input columns (edge id, src, dst, timestamp, amount) first
BASIC = ["amount_received", "amount_paid", "receiving_currency", "payment_currency",
         "payment_format", "same_bank", "hour"]  # fmt: skip
LGBM_GRID = [
    {"num_leaves": nl, "scale_pos_weight": w, "learning_rate": 0.05, "min_child_samples": 20}
    for nl in (31, 127) for w in (1.0, 8.0)
]  # fmt: skip


def _gfp(threads: int):
    from snapml import GraphFeaturePreprocessor

    g = GraphFeaturePreprocessor()
    g.set_params({**GFP_PARAMS, "num_threads": threads})
    return g


@lru_cache(maxsize=1)
def time_invariant_columns() -> np.ndarray:
    """Indices (after the 5 echo columns) of GFP outputs that do not change when all timestamps shift.

    Mean and sum of the raw timestamp are absolute-time features; the model would learn the date.
    Found empirically on a random sparse graph so it does not depend on the library's column layout
    (sparse on purpose: a dense 40-node graph makes the cycle enumeration run for minutes).
    """
    rng = np.random.default_rng(0)
    n = 400
    e = np.column_stack([np.arange(n), rng.integers(0, 2000, n), rng.integers(0, 2000, n),
                         np.sort(rng.integers(0, 3 * HOUR, n)), rng.uniform(1, 100, n)]).astype("f8")  # fmt: skip
    a = _gfp(1).fit_transform(e)[:, N_ECHO:]
    e2 = e.copy()
    e2[:, 3] += 10 * DAY
    b = _gfp(1).fit_transform(e2)[:, N_ECHO:]
    return np.flatnonzero(np.all(np.isclose(a, b, rtol=1e-6, atol=1e-6), axis=0))


def graph_features(tx_id, ts, src, dst, amount, grp, batch: int = BATCH, threads: int = 4):
    """GFP features for time-sorted edges; `grp` (non-decreasing ints, e.g. 0 train, 1 val) splits batches."""
    g = _gfp(threads)
    edges = np.column_stack([tx_id, src, dst, ts, amount]).astype("f8")
    keep = time_invariant_columns()
    out = np.empty((len(edges), len(keep)), dtype="f4")
    bounds = np.flatnonzero(np.diff(grp)) + 1
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(edges)], strict=True):
        for s in range(lo, hi, batch):
            e = min(s + batch, hi)
            out[s:e] = np.clip(g.transform(edges[s:e])[:, N_ECHO:][:, keep], -1e30, 1e30)  # kurtosis can overflow f4
    return out


def category_codes(frame: pl.DataFrame) -> dict[str, dict[str, int]]:
    """String -> int codes for the three categorical columns, from the given (train/val) rows only."""
    cols = ("receiving_currency", "payment_currency", "payment_format")
    return {c: {v: i for i, v in enumerate(sorted(frame[c].unique().to_list()))} for c in cols}


def build_features(frame: pl.DataFrame, codes: dict, threads: int = 4):
    """(X float32, names, sorted frame) for train/val rows; GFP is run over the whole frame in time order.

    `frame` is one graph: the full train+val data (see-everything) or one bank view. Its rows must be
    train/val only; a test row raises unless FTM_FINAL_RUN=1.
    """
    if (frame["split"] == "test").any():
        guards.require_final_run("test rows in build_features")
    f = frame.sort(["timestamp", "tx_id"])
    # "paper_test" is only used by the --repro run on the published split of the raw file
    order = {"train": 0, "val": 1, "test": 2, "paper_test": 2}
    grp = f["split"].replace_strict(order, return_dtype=pl.Int8).to_numpy()
    if (np.diff(grp) < 0).any():
        raise ValueError("a later split starts before an earlier one ends; cannot batch without leakage")
    t0 = f["timestamp"].min()
    ts = (f["timestamp"] - t0).dt.total_seconds().to_numpy()
    gf = graph_features(f["tx_id"].to_numpy(), ts, f["src_id"].to_numpy(), f["dst_id"].to_numpy(),
                        f["amount_received"].to_numpy(), grp, threads=threads)  # fmt: skip
    basic = np.column_stack([
        f["amount_received"].to_numpy(), f["amount_paid"].to_numpy(),
        *[f[c].replace_strict(codes[c], default=-1, return_dtype=pl.Int16).to_numpy()
          for c in ("receiving_currency", "payment_currency", "payment_format")],
        (f["from_bank"] == f["to_bank"]).cast(pl.Int8).to_numpy(), f["timestamp"].dt.hour().to_numpy(),
    ]).astype("f4")  # fmt: skip
    names = BASIC + [f"gfp_{i}" for i in time_invariant_columns()]
    return np.hstack([basic, gf]), names, f


def fit_lgbm(Xtr, ytr, Xva, yva, grid=LGBM_GRID, seed: int = 0, threads: int = 4):
    """Fit one LightGBM per grid point with early stopping on (Xva, yva); keep the best by validation AP.

    Returns (model, chosen_params, log) where log lists (params, AP, best_iteration) for every point.
    """
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score

    best, log = None, []
    for p in grid:
        m = lgb.LGBMClassifier(n_estimators=600, random_state=seed, n_jobs=threads, verbose=-1,
                               colsample_bytree=0.8, subsample=0.8, subsample_freq=1, **p)  # fmt: skip
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(30, verbose=False)])  # fmt: skip
        ap = float(average_precision_score(yva, m.predict_proba(Xva)[:, 1]))
        log.append({**p, "val_ap": ap, "best_iteration": int(m.best_iteration_ or 0)})
        if best is None or ap > best[0]:
            best = (ap, m, p)
    return best[1], best[2], log


def fit_calibrator(raw, y, min_pos: int = 30):
    """Isotonic regression raw -> P(laundering); Platt scaling when there are fewer than `min_pos` positives.

    Returns (callable on raw scores, name). Isotonic is the plan's choice; with a handful of positives
    it is a step function with large plateaus, so the sparse banks fall back to a 2-parameter logistic.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    raw, y = np.asarray(raw, "f8"), np.asarray(y)
    if y.sum() >= min_pos:
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, y)
        return iso.predict, "isotonic"
    z = np.log(np.clip(raw, 1e-9, 1 - 1e-9) / (1 - np.clip(raw, 1e-9, 1 - 1e-9)))[:, None]
    lr = LogisticRegression(C=1e6, max_iter=500).fit(z, y)

    def platt(r):
        r = np.clip(np.asarray(r, "f8"), 1e-9, 1 - 1e-9)
        return lr.predict_proba(np.log(r / (1 - r))[:, None])[:, 1]

    return platt, "platt"


def metrics(y, p, thr: float | None = None) -> dict:
    """Brier, PR-AUC, and F1/precision/recall at `thr` (the best-F1 threshold of (y, p) when None)."""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    y, p = np.asarray(y), np.asarray(p, "f8")
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "brier": float(np.mean((p - y) ** 2))}
    if y.sum() == 0:
        return out | {"pr_auc": float("nan"), "f1": float("nan"), "precision": float("nan"),
                      "recall": float("nan"), "thr": thr}  # fmt: skip
    out["pr_auc"] = float(average_precision_score(y, p))
    if thr is None:
        pr, rc, th = precision_recall_curve(y, p)
        f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
        thr = float(th[int(np.argmax(f1[:-1]))])
    pred = p >= thr
    tp = float((pred & (y == 1)).sum())
    prec, rec = tp / max(pred.sum(), 1), tp / y.sum()
    return out | {"f1": 2 * prec * rec / max(prec + rec, 1e-12), "precision": prec, "recall": rec, "thr": thr}


def val_parts(timestamps: pl.Series, first, last) -> np.ndarray:
    """Label each val row 'fit' (first half of the val time span), 'cal' (next quarter) or 'eval' (last quarter).

    `fit` trains the early stopping and the calibrator, `cal` is the conformal calibration part (it is
    not used for isotonic), `eval` is where every reported val metric and achieved miss rate is measured.
    The cut points come from the val time range only, so they are identical for every bank and mode.
    """
    span = (last - first).total_seconds()
    frac = ((timestamps - first).dt.total_seconds() / span).to_numpy()
    return np.where(frac < 0.5, "fit", np.where(frac < 0.75, "cal", "eval"))


def score_graph(frame: pl.DataFrame, codes: dict, parts_cut: tuple, seed: int = 0, threads: int = 4,
                grid=LGBM_GRID, n_folds: int = 3, extra_cols=None) -> dict:  # fmt: skip
    """Features -> LightGBM on train -> calibration on val 'fit' -> scores for every train/val row.

    Train rows get out-of-fold raw scores (`n_folds` contiguous time blocks, each scored by a model fit
    on the other blocks with the chosen hyper-parameters), val rows the model fit on all of train. The
    same calibrator (fit on val 'fit' only) maps both to p0. Returns arrays aligned with `frame` sorted
    by (timestamp, tx_id) plus run metadata.
    """
    t0 = time.time()
    X, names, f = build_features(frame, codes, threads=threads)
    if extra_cols is not None:
        X = np.hstack([X, np.column_stack([f[c].to_numpy() for c in extra_cols]).astype("f4")])
        names = names + list(extra_cols)
    t_feat = time.time() - t0
    y = f["is_laundering"].to_numpy().astype(int)
    sp = f["split"].to_numpy()
    tr, va = sp == "train", sp == "val"
    part = np.full(len(f), "", dtype="U4")
    part[va] = val_parts(f["timestamp"].filter(pl.Series(va)), *parts_cut)
    fit = va & (part == "fit")
    meta = {"n_rows": int(len(f)), "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_features": X.shape[1],
            "feature_seconds": t_feat, "train_pos": int(y[tr].sum()), "val_fit_pos": int(y[fit].sum())}  # fmt: skip
    raw = np.full(len(f), np.nan)
    if y[tr].sum() < 5 or y[fit].sum() < 1:
        meta["status"] = "skipped: too few positives"
        return {"frame": f, "raw": raw, "p0": raw.copy(), "part": part, "meta": meta}
    t1 = time.time()
    model, params, log = fit_lgbm(X[tr], y[tr], X[fit], y[fit], grid=grid, seed=seed, threads=threads)
    raw[va] = model.predict_proba(X[va])[:, 1]
    # out-of-fold train scores: contiguous time blocks of the (time-sorted) train rows
    idx = np.flatnonzero(tr)
    blocks = np.array_split(idx, n_folds)
    import lightgbm as lgb

    for b in blocks:
        rest = np.setdiff1d(idx, b, assume_unique=True)
        if y[rest].sum() < 5:
            raw[b] = model.predict_proba(X[b])[:, 1]  # ponytail: in-sample fallback for tiny banks
            continue
        m = lgb.LGBMClassifier(n_estimators=max(model.best_iteration_ or 50, 20), random_state=seed,
                               n_jobs=threads, verbose=-1, colsample_bytree=0.8, subsample=0.8,
                               subsample_freq=1, **params)  # fmt: skip
        m.fit(X[rest], y[rest])
        raw[b] = m.predict_proba(X[b])[:, 1]
    cal, cal_name = fit_calibrator(raw[fit], y[fit])
    p0 = np.full(len(f), np.nan)
    ok = ~np.isnan(raw)
    p0[ok] = cal(raw[ok])
    meta |= {"params": params, "grid": log, "calibration": cal_name, "model_seconds": time.time() - t1,
             "status": "ok"}  # fmt: skip
    return {"frame": f, "raw": raw, "p0": p0, "part": part, "meta": meta, "y": y,
            "importances": dict(zip(names, model.feature_importances_.tolist(), strict=True))}  # fmt: skip


def p0_table(res: dict, mode: str, bank: int | None, cm: str, seed: int = 0) -> pl.DataFrame:
    """Rows of the p0 hand-off table for one graph (schema in ftm/baselines/README.md)."""
    f = res["frame"]
    n = f.height
    return pl.DataFrame({
        "tx_id": f["tx_id"], "bank": pl.Series([bank] * n, dtype=pl.Int64), "mode": [mode] * n,
        "model": ["gfp_lgbm"] * n, "split": f["split"], "val_part": res["part"], "p0": res["p0"],
        "score_raw": res["raw"], "label": f["is_laundering"], "seed": [seed] * n, "commit": [cm] * n,
    }).with_columns(pl.col("val_part").replace("", None))  # fmt: skip
