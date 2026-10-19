"""GFP+LightGBM per bank and full; calibrated probabilities per transaction.

Owner: T1 (Abhishek).  Run from the repo root:
  python scripts/train_gfp.py --repro                      # E1/G1: reproduce F1 63.23 on the paper split
  python scripts/train_gfp.py --data hi_small --mode both  # p0 hand-off (#5, #17), see ftm/baselines/README.md
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

from ftm.baselines import gfp_gbt as g  # noqa: E402

SEED = 0


def commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def repro(csv: str, out: str, threads: int) -> None:
    """E1/G1 reproduction on the published split. REPRODUCTION ONLY, behind --repro because the
    published 60/20/20 split overlaps our test days: it reads the raw CSV, uses only the paper's
    hyper-parameter families (fixed small grid, tuned on its own val part), writes nothing to
    data/scores or data/alerts, and nothing from it feeds any later model or decision (tracks.md 3.5).
    """
    import xgboost as xgb
    from sklearn.metrics import average_precision_score

    t0 = time.time()
    raw = pl.read_csv(csv, infer_schema_length=10000, schema_overrides={"Account": pl.Utf8, "Account_duplicated_0": pl.Utf8})
    cols = raw.columns  # duplicated header "Account" is renamed by polars
    acc = [c for c in cols if c.startswith("Account")]
    df = raw.rename({
        "Timestamp": "timestamp_s", "From Bank": "from_bank", "To Bank": "to_bank", acc[0]: "from_account",
        acc[1]: "to_account", "Amount Received": "amount_received", "Amount Paid": "amount_paid",
        "Receiving Currency": "receiving_currency", "Payment Currency": "payment_currency",
        "Payment Format": "payment_format", "Is Laundering": "is_laundering",
    }).with_row_index("tx_id")  # fmt: skip
    df = df.with_columns(
        pl.col("tx_id").cast(pl.Int64),
        pl.col("timestamp_s").str.to_datetime("%Y/%m/%d %H:%M").alias("timestamp"),
        pl.col("from_bank").cast(pl.Int64), pl.col("to_bank").cast(pl.Int64),
        (pl.col("from_bank").cast(pl.Utf8) + "_" + pl.col("from_account")).alias("fk"),
        (pl.col("to_bank").cast(pl.Utf8) + "_" + pl.col("to_account")).alias("tk"),
    )  # fmt: skip
    nodes = pl.concat([df["fk"], df["tk"]]).unique().sort().to_frame("k").with_row_index("node")
    df = df.join(nodes.rename({"k": "fk", "node": "src_id"}), on="fk").join(nodes.rename({"k": "tk", "node": "dst_id"}), on="tk")
    df = df.sort(["timestamp", "tx_id"])
    n = df.height
    split = np.where(np.arange(n) < 0.6 * n, "train", np.where(np.arange(n) < 0.8 * n, "val", "paper_test"))
    df = df.with_columns(pl.Series("split", split), pl.col("src_id").cast(pl.Int64), pl.col("dst_id").cast(pl.Int64))
    print(f"loaded {n} rows in {time.time() - t0:.0f}s", flush=True)
    codes = g.category_codes(df.filter(pl.col("split") != "paper_test"))
    t1 = time.time()
    cache = Path("data/runs/cache_repro.npz")  # intermediate, deleted at the end of the session
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        X, y, sp, t_feat = z["X"], z["y"], z["sp"], float(z["t"])
    else:
        X, names, f = g.build_features(df, codes, threads=threads)
        t_feat = time.time() - t1
        y = f["is_laundering"].to_numpy().astype(int)
        sp = f["split"].to_numpy()
        np.savez(cache, X=X, y=y, sp=sp, t=t_feat)
    tr, va, te = sp == "train", sp == "val", sp == "paper_test"
    print(f"features {X.shape} in {t_feat:.0f}s", flush=True)
    res = {"commit": commit(), "seed": SEED, "n": int(n), "feature_seconds": t_feat, "n_features": X.shape[1],
           "published": {"GFP+XGBoost batch128": 63.23, "GFP+LightGBM batch128": 62.86}, "models": {}}  # fmt: skip

    t2 = time.time()
    m, p, log = g.fit_lgbm(X[tr], y[tr], X[va], y[va], seed=SEED, threads=threads)
    sv, st = m.predict_proba(X[va])[:, 1], m.predict_proba(X[te])[:, 1]
    mv = g.metrics(y[va], sv)
    res["models"]["lightgbm"] = {"params": p, "grid": log, "val": mv, "test": g.metrics(y[te], st, mv["thr"]),
                                  "test_f1_at_0.5": g.metrics(y[te], st, 0.5)["f1"], "seconds": time.time() - t2}  # fmt: skip
    t = res["models"]["lightgbm"]["test"]
    print("lightgbm", f"test F1 {100 * t['f1']:.2f} P {t['precision']:.3f} R {t['recall']:.3f} PR-AUC {t['pr_auc']:.3f}", flush=True)
    best = None
    t2 = time.time()
    xlog = []
    for depth in (6, 10):
        for w in (1.0, 8.0):
            xm = xgb.XGBClassifier(n_estimators=600, max_depth=depth, learning_rate=0.05, scale_pos_weight=w,
                                   tree_method="hist", subsample=0.8, colsample_bytree=0.8, n_jobs=threads,
                                   random_state=SEED, eval_metric="aucpr", early_stopping_rounds=30)  # fmt: skip
            xm.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
            ap = float(average_precision_score(y[va], xm.predict_proba(X[va])[:, 1]))
            xlog.append({"max_depth": depth, "scale_pos_weight": w, "val_ap": ap})
            if best is None or ap > best[0]:
                best = (ap, xm, {"max_depth": depth, "scale_pos_weight": w})
    xm = best[1]
    sv, st = xm.predict_proba(X[va])[:, 1], xm.predict_proba(X[te])[:, 1]
    mv = g.metrics(y[va], sv)
    res["models"]["xgboost"] = {"params": best[2], "grid": xlog, "val": mv, "test": g.metrics(y[te], st, mv["thr"]),
                                 "test_f1_at_0.5": g.metrics(y[te], st, 0.5)["f1"], "seconds": time.time() - t2}  # fmt: skip
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    for k, v in res["models"].items():
        t = v["test"]
        print(k, f"test F1 {100 * t['f1']:.2f} P {t['precision']:.3f} R {t['recall']:.3f} PR-AUC {t['pr_auc']:.3f}", flush=True)


COLS = ["tx_id", "timestamp", "from_bank", "to_bank", "src_id", "dst_id", "amount_received", "amount_paid",
        "receiving_currency", "payment_currency", "payment_format", "is_laundering", "split"]  # fmt: skip


def run_dataset(ds: str, mode: str, threads: int, banks: list[int] | None, neg_frac: float = 1.0, tag: str = "", exclude=()) -> None:
    from ftm.data import io

    cfg = io.load_config(ds)
    pq = cfg["parquet_dir"]
    v = io.scan_transactions(pq, split="val", columns=["timestamp"]).select(
        pl.col("timestamp").min().alias("a"), pl.col("timestamp").max().alias("b")).collect()  # fmt: skip
    cut = (v["a"][0], v["b"][0])
    out_dir = Path("data/scores") / ds
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("data/runs").mkdir(parents=True, exist_ok=True)
    cm = commit()
    meta = {"dataset": ds, "mode": mode, "commit": cm, "seed": SEED, "val_cut": [str(cut[0]), str(cut[1])], "graphs": {}}
    parts = []
    t0 = time.time()
    if mode == "see_everything":
        frame = io.scan_transactions(pq, split=["train", "val"], columns=COLS).sort(["timestamp", "tx_id"]).collect()
        codes = g.category_codes(frame)
        frame = g.encode_categoricals(frame, codes)  # memory: strings -> Int16
        res = g.score_graph(frame, codes, cut, seed=SEED, threads=threads, neg_frac=neg_frac, presorted=True)
        meta["graphs"]["all"] = res["meta"] | {"importances": res.get("importances")}
        parts.append(g.p0_table(res, mode, None, cm))
        del frame, res
    else:
        todo = [b for b in (banks or io.read_banks(cfg["views_dir"])["bank"].to_list()) if b not in exclude]
        for b in todo:
            frame = io.scan_view(cfg["views_dir"], b, split=["train", "val"], columns=COLS).collect()
            frame = frame.with_columns((pl.col("from_bank") == b).cast(pl.Int8).alias("out"))  # bank knows its direction
            res = g.score_graph(frame, g.category_codes(frame), cut, seed=SEED, threads=threads, extra_cols=["out"])
            meta["graphs"][str(b)] = res["meta"]
            parts.append(g.p0_table(res, mode, b, cm))
            print(f"bank {b}: {res['meta'].get('status')} {time.time() - t0:.0f}s", flush=True)
    df = pl.concat(parts)
    sfx = f"__{tag}" if tag else ""
    df.write_parquet(out_dir / f"p0_{mode}{sfx}.parquet")
    meta["seconds"] = time.time() - t0
    Path(f"data/runs/p0_{ds}_{mode}{sfx}.json").write_text(json.dumps(meta, indent=1, default=str))
    print("done", ds, mode, f"{time.time() - t0:.0f}s", flush=True)
    if not tag:
        combine(ds)


def recalibrate(ds: str) -> None:
    """Refit the calibrator of every (mode, bank) from the stored raw scores (val 'fit' part) and rewrite p0.

    Same rule as score_graph (isotonic only with enough positives and signal, else Platt); no model is refit.
    """
    d = Path("data/scores") / ds
    for mode in ("one_bank", "see_everything"):
        f = d / f"p0_{mode}.parquet"
        if not f.exists():
            continue
        df = pl.read_parquet(f)
        outs = []
        for (bank,), grp in df.group_by(["bank"], maintain_order=True):
            ok = grp.filter(pl.col("score_raw").is_not_null() & pl.col("score_raw").is_not_nan())
            fit = ok.filter(pl.col("val_part") == "fit")
            if fit.height == 0 or fit["label"].sum() == 0:
                outs.append(grp)
                continue
            trn = ok.filter(pl.col("split") == "train")
            sig = g.has_signal(trn["label"].cast(pl.Int8).to_numpy(), trn["score_raw"].to_numpy()) if trn.height else False
            cal, name = g.fit_calibrator(fit["score_raw"].to_numpy(), fit["label"].cast(pl.Int8).to_numpy(), has_signal=sig)
            p0 = np.where(grp["score_raw"].is_null() | grp["score_raw"].is_nan(), np.nan, cal(grp["score_raw"].fill_null(0.0).fill_nan(0.0).to_numpy()))
            outs.append(grp.with_columns(pl.Series("p0", p0).fill_nan(None)))
            print(mode, bank, name, flush=True)
        pl.concat(outs).write_parquet(f)
    combine(ds)


def combine(ds: str) -> None:
    """p0_one_bank.parquet = untagged + tagged parts (parallel runs over bank subsets); p0.parquet = both modes."""
    d = Path("data/scores") / ds
    for mode in ("one_bank", "see_everything"):
        parts = sorted(d.glob(f"p0_{mode}__*.parquet"))
        if parts:
            base = [d / f"p0_{mode}.parquet"] if (d / f"p0_{mode}.parquet").exists() else []
            pl.concat([pl.read_parquet(x) for x in base + parts]).write_parquet(d / f"p0_{mode}.parquet")
            for x in parts:
                x.unlink()  # merged; the per-run json files in data/runs keep the metadata
    both = [d / f"p0_{m}.parquet" for m in ("one_bank", "see_everything")]
    if all(x.exists() for x in both):
        pl.concat([pl.read_parquet(x) for x in both]).write_parquet(d / "p0.parquet")
        print("combined", ds, flush=True)


def report(ds: str) -> None:
    """Per-bank and pooled one-bank vs see-everything table on the val 'eval' part (the gap)."""
    d = Path("data/scores") / ds
    ob = pl.read_parquet(d / "p0_one_bank.parquet")  # time-guard: not transactions
    se = pl.read_parquet(d / "p0_see_everything.parquet").select("tx_id", pl.col("p0").alias("p0_se"))  # time-guard: not transactions
    v = ob.filter((pl.col("split") == "val") & pl.col("p0").is_not_null()).join(se, on="tx_id", how="left")
    rows = []
    for b in sorted(v["bank"].unique().to_list()):
        vb = v.filter(pl.col("bank") == b)
        fit, ev = vb.filter(pl.col("val_part") == "fit"), vb.filter(pl.col("val_part") == "eval")
        y = ev["label"].cast(pl.Int8).to_numpy()
        const = float(fit["label"].cast(pl.Float64).mean())
        allb = ob.filter(pl.col("bank") == b)
        r = {"bank": b, "n_laundering_train_val": int(allb["label"].sum()), "n_eval": len(y), "n_pos": int(y.sum()), "brier_const": float(np.mean((const - y) ** 2))}
        for tag, col in (("ob", "p0"), ("se", "p0_se")):
            thr = g.metrics(fit["label"].cast(pl.Int8).to_numpy(), fit[col].to_numpy())["thr"]
            m = g.metrics(y, ev[col].to_numpy(), thr)
            r |= {f"brier_{tag}": m["brier"], f"prauc_{tag}": m.get("pr_auc"), f"f1_{tag}": m.get("f1")}
        rows.append(r)
    tab = pl.DataFrame(rows)
    ev = v.filter(pl.col("val_part") == "eval")
    y = ev["label"].cast(pl.Int8).to_numpy()
    const = float(v.filter(pl.col("val_part") == "fit")["label"].cast(pl.Float64).mean())
    pooled = {"n_eval": len(y), "n_pos": int(y.sum()), "brier_const": float(np.mean((const - y) ** 2))}
    for tag, col in (("ob", "p0"), ("se", "p0_se")):
        fit = v.filter(pl.col("val_part") == "fit")
        thr = g.metrics(fit["label"].cast(pl.Int8).to_numpy(), fit[col].to_numpy())["thr"]
        m = g.metrics(y, ev[col].to_numpy(), thr)
        pooled |= {f"brier_{tag}": m["brier"], f"prauc_{tag}": m["pr_auc"], f"f1_{tag}": m["f1"]}
    # sum of squared errors weighted pooled Brier is the number above; the macro mean over banks is below
    pooled["bss_ob"] = 1 - pooled["brier_ob"] / pooled["brier_const"]
    pooled["bss_se"] = 1 - pooled["brier_se"] / pooled["brier_const"]
    # paired bootstrap over eval rows (not over banks/days: the CI is about row noise only)
    rng = np.random.default_rng(SEED)
    se_, ob_ = (ev["p0_se"].to_numpy() - y) ** 2, (ev["p0"].to_numpy() - y) ** 2
    diffs = [(ob_[i] - se_[i]).mean() for i in (rng.integers(0, len(y), len(y)) for _ in range(300))]
    pooled["brier_gap_ob_minus_se"] = float((ob_ - se_).mean())
    pooled["brier_gap_ci95"] = [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]
    pooled["macro_brier_ob"] = float(tab["brier_ob"].mean())
    pooled["macro_brier_se"] = float(tab["brier_se"].mean())
    out = Path("data/runs")
    tab.write_csv(out / f"gap_{ds}_per_bank.csv")
    (out / f"gap_{ds}_pooled.json").write_text(json.dumps(pooled, indent=1))
    md = ["| bank | laundering (train+val) | eval rows | eval pos | Brier const | Brier one-bank | Brier see-all | PR-AUC one-bank | PR-AUC see-all | F1 one-bank | F1 see-all |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    f = lambda x: "n/a" if x is None or x != x else f"{x:.4f}"  # noqa: E731
    for r in tab.iter_rows(named=True):
        md.append(f"| {r['bank']} | {r['n_laundering_train_val']} | {r['n_eval']} | {r['n_pos']} | {r['brier_const'] * 1e3:.3f}e-3 | {r['brier_ob'] * 1e3:.3f}e-3 | {r['brier_se'] * 1e3:.3f}e-3 | {f(r['prauc_ob'])} | {f(r['prauc_se'])} | {f(r['f1_ob'])} | {f(r['f1_se'])} |")  # fmt: skip
    (out / f"gap_{ds}_per_bank.md").write_text("\n".join(md) + "\n")
    with pl.Config(tbl_rows=40, tbl_cols=12, float_precision=5):
        print(tab)
    print(json.dumps(pooled, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro", action="store_true", help="reproduction on the published split (raw CSV)")
    ap.add_argument("--csv", default="data/raw/hi_small/HI-Small_Trans.csv")
    ap.add_argument("--out", default="data/runs/gfp_repro_hi_small.json")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--data", default=None)
    ap.add_argument("--mode", choices=["one_bank", "see_everything"], default=None)
    ap.add_argument("--report", action="store_true", help="one-bank vs see-everything table for --data")
    ap.add_argument("--neg-frac", type=float, default=1.0, help="share of clean train rows used for fitting")
    ap.add_argument("--tag", default="", help="write p0_<mode>__<tag>.parquet (parallel runs); merge with --combine")
    ap.add_argument("--exclude", type=int, nargs="*", default=[], help="banks to skip")
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--recalibrate", action="store_true", help="refit calibrators from stored raw scores")
    ap.add_argument("--banks", type=int, nargs="*", default=None, help="subset of banks (debugging)")
    a = ap.parse_args()
    if a.repro:
        repro(a.csv, a.out, a.threads)
    elif a.recalibrate and a.data:
        recalibrate(a.data)
    elif a.combine and a.data:
        combine(a.data)
    elif a.report and a.data:
        report(a.data)
    elif a.data and a.mode:
        run_dataset(a.data, a.mode, a.threads, a.banks, a.neg_frac, a.tag, a.exclude)
    else:
        raise SystemExit("give --repro, or --data hi_small --mode one_bank|see_everything")


if __name__ == "__main__":
    main()
