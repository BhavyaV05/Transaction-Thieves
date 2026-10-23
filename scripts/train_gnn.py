"""Multi-GNN / FraudGT / MEGA-GNN (one-bank and see-everything).

Owner: T1 (Abhishek).  Run from the repo root:
  python scripts/train_gnn.py --repro     # M2: Multi-GNN reproduction on its own split (prints the command)

REPRODUCTION ONLY (tracks.md 3.5): IBM's Multi-GNN code splits HI-Small 60/20/20 by transaction index, which
overlaps our test days, and logs the test F1 every epoch. That run reads only the raw transactions, writes
nothing to data/scores or data/alerts, and nothing from it feeds any later model or decision.

OUR split (train/val only, through ftm.data.io; no test edge ever enters a graph):
  python scripts/train_gnn.py --our-split --data hi_small --mode see_everything --budget-s 4800
  python scripts/train_gnn.py --our-split --data hi_small --mode one_bank --budget-s 4800 [--banks 70 10 12]
  python scripts/train_gnn.py --compare --data hi_small        # Brier / PR-AUC / F1 vs GFP on val `eval`
Writes data/scores/<ds>/p0_multi_gnn.parquet (val rows; GFP's files are untouched) and data/runs/p0_<ds>_multi_gnn_<mode>.json.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.baselines import gfp_gbt as g  # noqa: E402
from ftm.baselines import multi_gnn  # noqa: E402
from ftm.data import io  # noqa: E402

MIN_TRAIN_POS = 5  # one-bank: skip views with fewer train positives (logged with the count); same rule as GFP
TOP_BANKS = 30  # every view
EXTRA = ["from_bank", "to_bank"]


def commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def log(msg) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def val_cut(pq) -> tuple:
    v = io.scan_transactions(pq, split="val", columns=["timestamp"]).select(
        pl.col("timestamp").min().alias("a"), pl.col("timestamp").max().alias("b")).collect()  # fmt: skip
    return v["a"][0], v["b"][0]


def run_graph(frame: pl.DataFrame, cut, mode: str, bank, budget_s: float, max_epochs: int, clone: Path, cm: str, workers: int, ckpt, batch_size: int = 2048, resume: bool = False):
    """Train + score one graph, calibrate on the val `fit` part, return (p0 table of the val rows, metadata)."""
    frame = frame.sort(["timestamp", "tx_id"])
    sp = frame["split"].to_numpy()
    part = np.full(frame.height, "", dtype="U4")
    part[sp == "val"] = g.val_parts(frame["timestamp"].filter(pl.Series(sp == "val")), *cut)
    res = multi_gnn.train_and_score(frame, clone, budget_s, max_epochs, part == "fit", num_workers=workers, log=log, ckpt=ckpt, batch_size=batch_size, resume=resume)
    v = frame.filter(pl.col("split") == "val")
    raw, y, vpart = res["raw"], v["is_laundering"].cast(pl.Int8).to_numpy(), part[sp == "val"]
    ok, fit = ~np.isnan(raw), vpart == "fit"
    p0 = np.full(len(raw), np.nan)
    if y[fit & ok].sum() < 1:  # no laundering row in the val fit part: no calibrator can be fit, p0 stays null (GFP skips such views too)
        sig, name = False, "skipped: no val-fit positives"
    else:
        sig = g.has_signal(y[fit & ok], raw[fit & ok])  # signal check on the fit part itself (the part the calibrator sees)
        cal, name = g.fit_calibrator(raw[fit & ok], y[fit & ok], has_signal=sig)
        p0[ok] = cal(raw[ok])
    tab = pl.DataFrame({"tx_id": v["tx_id"], "bank": pl.Series([bank] * v.height, dtype=pl.Int64), "mode": [mode] * v.height,
                        "model": ["multi_gnn"] * v.height, "split": ["val"] * v.height, "val_part": vpart,
                        "p0": pl.Series(p0).fill_nan(None), "score_raw": pl.Series(raw.astype("f8")).fill_nan(None),
                        "label": v["is_laundering"], "seed": [1] * v.height, "commit": [cm] * v.height})  # fmt: skip
    res.pop("raw"), res.pop("tx_id")
    return tab, res | {"calibration": name, "fit_signal": bool(sig)}


def our_split(a) -> None:
    cfg = io.load_config(a.data)
    pq, clone = cfg["parquet_dir"], Path(a.clone).expanduser()
    cut, cm = val_cut(pq), commit()
    cols = multi_gnn.EDGE_COLS
    out = Path("data/scores") / a.data
    out.mkdir(parents=True, exist_ok=True)
    Path("data/runs").mkdir(exist_ok=True)
    meta = {"dataset": a.data, "mode": a.mode, "commit": cm, "seed": 1, "batch_size": a.batch_size, "neighbours": list(multi_gnn.N_NEIGHS), "val_cut": [str(cut[0]), str(cut[1])], "graphs": {}, "skipped": {}}
    t0, parts = time.time(), []
    f_tab, f_meta = out / f"p0_multi_gnn__{a.mode}.parquet", Path(f"data/runs/p0_{a.data}_multi_gnn_{a.mode}.json")
    base = None  # --banks resumes: the other banks' rows already on disk stay (written back with every checkpoint)
    if a.mode == "one_bank" and a.banks and f_tab.exists():
        base = pl.read_parquet(f_tab).filter(~pl.col("bank").is_in(list(a.banks)))  # time-guard: not transactions

    def write_table(ps):
        pl.concat(([base] if base is not None else []) + ps).write_parquet(f_tab)
    ckdir = Path("models/multi_gnn")  # gitignored
    ckdir.mkdir(parents=True, exist_ok=True)
    if a.mode == "see_everything":
        frame = io.scan_transactions(pq, split=["train", "val"], columns=cols).collect()
        tab, m = run_graph(frame, cut, a.mode, None, a.budget_s, a.max_epochs, clone, cm, a.workers, ckdir / f"{a.data}_see.pt", a.batch_size, a.resume)
        meta["graphs"]["all"] = m
        parts.append(tab)
    else:
        banks = io.read_banks(cfg["views_dir"]).sort("rank")
        todo = a.banks or banks["bank"].head(TOP_BANKS).to_list()
        frames = {}
        for b in todo:
            f = io.scan_view(cfg["views_dir"], b, split=["train", "val"], columns=cols + EXTRA).collect()
            npos = int(f.filter(pl.col("split") == "train")["is_laundering"].sum())
            if npos < MIN_TRAIN_POS:
                meta["skipped"][str(b)] = f"{npos} train positives < {MIN_TRAIN_POS}"
                log(f"bank {b}: skipped, {npos} train positives")
                continue
            frames[b] = f.drop(EXTRA)
        size = {b: f.filter(pl.col("split") == "train").height for b, f in frames.items()}
        for b, f in frames.items():
            share = a.budget_s * size[b] / sum(size.values())
            tab, m = run_graph(f, cut, a.mode, b, max(share, 90.0), a.max_epochs, clone, cm, a.workers, ckdir / f"{a.data}_bank{b}.pt", a.batch_size)
            meta["graphs"][str(b)] = m
            parts.append(tab)
            log(f"bank {b} done, {time.time() - t0:.0f}s")
            write_table(parts)  # checkpoint of the table
    write_table(parts)
    if a.mode == "one_bank" and a.banks:  # keep the other banks' metadata if it exists
        if f_meta.exists():
            old = json.loads(f_meta.read_text())
            meta["graphs"] = old["graphs"] | meta["graphs"]
            meta["skipped"] = old["skipped"] | meta["skipped"]
    meta["seconds"] = time.time() - t0
    f_meta.write_text(json.dumps(meta, indent=1, default=str))
    combine(a.data)
    log(f"done {a.mode} {meta['seconds']:.0f}s")


def combine(ds: str) -> None:
    """p0_multi_gnn.parquet = whichever of the two per-mode files exist (GFP's p0 files are never touched)."""
    d = Path("data/scores") / ds
    fs = [d / f"p0_multi_gnn__{m}.parquet" for m in ("one_bank", "see_everything") if (d / f"p0_multi_gnn__{m}.parquet").exists()]
    pl.concat([pl.read_parquet(f) for f in fs]).write_parquet(d / "p0_multi_gnn.parquet")  # time-guard: not transactions


def compare(ds: str) -> None:
    """Brier / PR-AUC / F1 on the val `eval` part: Multi-GNN vs GFP, one-bank vs see-everything.

    F1 uses the best-F1 threshold on the same model's val `fit` part. One-bank rows are restricted to the banks
    Multi-GNN scored, for both models, so the comparison is on the same rows. Paired cluster bootstrap
    (ftm.eval, attempt + accounts clusters) for the see-everything gap between the two models.
    """
    from ftm.eval import bootstrap as bs

    cfg = io.load_config(ds)
    d = Path("data/scores") / ds
    mg = pl.read_parquet(d / "p0_multi_gnn.parquet")  # time-guard: not transactions
    gf = g.read_p0(ds, split="val")
    out: dict = {"dataset": ds}
    banks = sorted(mg.filter((pl.col("mode") == "one_bank") & pl.col("p0").is_not_null())["bank"].unique().to_list())
    out["one_bank_banks"] = banks

    def stats(df: pl.DataFrame) -> dict:
        df = df.filter(pl.col("p0").is_not_null() & pl.col("p0").is_not_nan())
        fit, ev = df.filter(pl.col("val_part") == "fit"), df.filter(pl.col("val_part") == "eval")
        thr = g.metrics(fit["label"].cast(pl.Int8).to_numpy(), fit["p0"].to_numpy())["thr"]
        m = g.metrics(ev["label"].cast(pl.Int8).to_numpy(), ev["p0"].to_numpy(), thr)
        const = float(fit["label"].cast(pl.Float64).mean())
        y = ev["label"].cast(pl.Int8).to_numpy()
        return {k: m.get(k) for k in ("n", "n_pos", "brier", "pr_auc", "f1")} | {"brier_const": float(np.mean((const - y) ** 2))}

    rows = {}
    for model, df in (("multi_gnn", mg), ("gfp_lgbm", gf)):
        for mode in ("one_bank", "see_everything"):
            x = df.filter(pl.col("mode") == mode)
            if mode == "one_bank":
                x = x.filter(pl.col("bank").is_in(banks))
            if x.height:
                rows[f"{model}/{mode}"] = stats(x)
    out["rows"] = rows
    # same transactions for the see-everything pair; paired bootstrap on the Brier difference
    pm = mg.filter((pl.col("mode") == "see_everything") & (pl.col("val_part") == "eval")).select("tx_id", pl.col("p0").alias("p_mg"), "label")
    pg = gf.filter((pl.col("mode") == "see_everything") & (pl.col("val_part") == "eval")).select("tx_id", pl.col("p0").alias("p_gfp"))
    j = pm.join(pg, on="tx_id").drop_nulls()
    if j.height:
        pq = cfg["parquet_dir"]
        tx = io.scan_transactions(pq, split="val", columns=["tx_id", "src_id", "dst_id"]).collect()
        pat = io.scan_patterns(pq, split="val").select("tx_id", "attempt_id").collect()
        j = j.join(tx, on="tx_id", how="left").join(pat, on="tx_id", how="left").sort("tx_id")
        cl = bs.cluster_ids(j["attempt_id"].to_list(), [j["src_id"].to_list(), j["dst_id"].to_list()])
        y = j["label"].cast(pl.Int8).to_numpy()
        r = bs.brier_diff(y, j["p_gfp"].to_numpy(), j["p_mg"].to_numpy(), cl, n_boot=2000)
        out["see_everything_brier_gfp_minus_multi_gnn"] = {"diff": r["diff"], "ci95": r["ci"], "clusters": bs.cluster_report(cl), "n": int(len(y))}
    # the project's gap: one-bank minus see-everything Brier on the same val `eval` rows of the scored banks, per model
    pq = cfg["parquet_dir"]
    tx = io.scan_transactions(pq, split="val", columns=["tx_id", "src_id", "dst_id"]).collect()
    pat = io.scan_patterns(pq, split="val").select("tx_id", "attempt_id").collect()
    out["one_bank_minus_see_everything_brier"] = {}
    for model, df in (("multi_gnn", mg), ("gfp_lgbm", gf)):
        ob = df.filter((pl.col("mode") == "one_bank") & pl.col("bank").is_in(banks) & (pl.col("val_part") == "eval")).select("tx_id", pl.col("p0").alias("p_one"), "label")
        se = df.filter((pl.col("mode") == "see_everything") & (pl.col("val_part") == "eval")).select("tx_id", pl.col("p0").alias("p_see"))
        j = ob.join(se, on="tx_id").drop_nulls().join(tx, on="tx_id", how="left").join(pat, on="tx_id", how="left").sort("tx_id")
        cl = bs.cluster_ids(j["attempt_id"].to_list(), [j["src_id"].to_list(), j["dst_id"].to_list()])
        r = bs.brier_diff(j["label"].cast(pl.Int8).to_numpy(), j["p_one"].to_numpy(), j["p_see"].to_numpy(), cl, n_boot=2000)
        out["one_bank_minus_see_everything_brier"][model] = {"diff": r["diff"], "ci95": r["ci"], "n": j.height}
    Path(f"data/runs/multi_gnn_vs_gfp_{ds}.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro", action="store_true")
    ap.add_argument("--our-split", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--mode", choices=["see_everything", "one_bank"], default="see_everything")
    ap.add_argument("--banks", type=int, nargs="*", default=None)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2048, help="2048 = reproduction config; 1024 if VRAM is short")
    ap.add_argument("--resume", action="store_true", help="see_everything: continue from models/multi_gnn/<ds>_see.pt (best epoch so far)")
    ap.add_argument("--workers", type=int, default=0, help="LinkNeighborLoader workers")
    ap.add_argument("--clone", default="~/Documents/t1_external/Multi-GNN")
    ap.add_argument("--budget-s", type=int, default=8700, help="stop after the epoch that passes this many seconds")
    ap.add_argument("--run", action="store_true", help="--repro: launch it (otherwise only print the command)")
    a = ap.parse_args()
    if a.our_split:
        return our_split(a)
    if a.compare:
        return compare(a.data)
    if not a.repro:
        raise SystemExit("give --repro, --our-split or --compare")
    clone = Path(a.clone).expanduser()
    metrics = Path("data/multignn/pna_metrics.jsonl").resolve()
    print(f"cd {clone} && T1_BUDGET_S={a.budget_s} T1_METRICS={metrics} {shlex.join(multi_gnn.repro_command())}")
    if a.run:
        multi_gnn.launch(clone, metrics, a.budget_s).wait()


if __name__ == "__main__":
    main()
