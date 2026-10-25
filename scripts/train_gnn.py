"""Multi-GNN / MEGA-GNN (one-bank and see-everything); FraudGT has its own patch in ftm/baselines.

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

MEGA-GNN (secondary SOTA, week 10) is the same pipeline with `--model mega_gnn` (own process; clone ~/Documents/t1_external/MEGA-GNN):
  python scripts/train_gnn.py --repro --model mega_gnn --run          # reproduction on its own split (reads the raw CSV, reproduction only)
  python scripts/train_gnn.py --our-split --model mega_gnn --data hi_small --mode one_bank --max-epochs 100 --budget-s 7200
Writes data/scores/<ds>/p0_mega_gnn__<mode>.parquet, p0_mega_gnn.parquet and data/runs/p0_<ds>_mega_gnn_<mode>.json.
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
from ftm.baselines import mega_gnn, multi_gnn  # noqa: E402
from ftm.data import io  # noqa: E402

MODELS = {"multi_gnn": (multi_gnn, "~/Documents/t1_external/Multi-GNN"), "mega_gnn": (mega_gnn, "~/Documents/t1_external/MEGA-GNN")}
SCORES, RUNS, MODEL_DIR = Path("data/scores"), Path("data/runs"), Path("models")  # `--scratch DIR` moves all three (smoke tests never touch the real files)
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


def run_graph(frame: pl.DataFrame, cut, mode: str, bank, budget_s: float, max_epochs: int, clone: Path, cm: str, workers: int, ckpt, batch_size: int = 2048, resume: bool = False, model_name: str = "multi_gnn"):
    """Train + score one graph, calibrate on the val `fit` part, return (p0 table of the val rows, metadata)."""
    frame = frame.sort(["timestamp", "tx_id"])
    sp = frame["split"].to_numpy()
    part = np.full(frame.height, "", dtype="U4")
    part[sp == "val"] = g.val_parts(frame["timestamp"].filter(pl.Series(sp == "val")), *cut)
    res = MODELS[model_name][0].train_and_score(frame, clone, budget_s, max_epochs, part == "fit", num_workers=workers, log=log, ckpt=ckpt, batch_size=batch_size, resume=resume)
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
                        "model": [model_name] * v.height, "split": ["val"] * v.height, "val_part": vpart,
                        "p0": pl.Series(p0).fill_nan(None), "score_raw": pl.Series(raw.astype("f8")).fill_nan(None),
                        "label": v["is_laundering"], "seed": [1] * v.height, "commit": [cm] * v.height})  # fmt: skip
    res.pop("raw"), res.pop("tx_id")
    return tab, res | {"calibration": name, "fit_signal": bool(sig)}


def our_split(a) -> None:
    cfg = io.load_config(a.data)
    pq, clone = cfg["parquet_dir"], Path(a.clone or MODELS[a.model][1]).expanduser()
    cut, cm = val_cut(pq), commit()
    cols = multi_gnn.EDGE_COLS
    out = SCORES / a.data
    out.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    meta = {"dataset": a.data, "model": a.model, "mode": a.mode, "commit": cm, "seed": 1, "batch_size": a.batch_size, "neighbours": list(multi_gnn.N_NEIGHS), "val_cut": [str(cut[0]), str(cut[1])], "graphs": {}, "skipped": {}}
    t0, parts = time.time(), []
    f_tab, f_meta = out / f"p0_{a.model}__{a.mode}.parquet", RUNS / f"p0_{a.data}_{a.model}_{a.mode}.json"
    base = None  # --banks resumes: the other banks' rows already on disk stay (written back with every checkpoint)
    if a.mode == "one_bank" and a.banks and f_tab.exists():
        base = pl.read_parquet(f_tab).filter(~pl.col("bank").is_in(list(a.banks)))  # time-guard: not transactions

    def write_table(ps):
        pl.concat(([base] if base is not None else []) + ps).write_parquet(f_tab)
    ckdir = MODEL_DIR / a.model  # gitignored
    ckdir.mkdir(parents=True, exist_ok=True)
    if a.mode == "see_everything":
        frame = io.scan_transactions(pq, split=["train", "val"], columns=cols).collect()
        tab, m = run_graph(frame, cut, a.mode, None, a.budget_s, a.max_epochs, clone, cm, a.workers, ckdir / f"{a.data}_see.pt", a.batch_size, a.resume, a.model)
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
            tab, m = run_graph(f, cut, a.mode, b, max(share, 90.0), a.max_epochs, clone, cm, a.workers, ckdir / f"{a.data}_bank{b}.pt", a.batch_size, model_name=a.model)
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
    combine(a.data, a.model)
    log(f"done {a.mode} {meta['seconds']:.0f}s")


def final_graph(frame: pl.DataFrame, cut, mode: str, bank, clone: Path, cm: str, workers: int, ckpt: Path, batch_size: int, model_name: str, stored: pl.DataFrame):
    """Final run for one graph: the checkpoint of the val run scores the TEST rows; the calibrator is refit from the stored val raw scores.

    Nothing is trained or tuned. `frame` holds train, val and test rows of the graph (a bank view or the whole graph). The calibrator is the
    val run's (`run_graph`): fit on the val `fit` rows of `stored` (the p0 table of the val run: raw score, label), with the same signal rule;
    its p0 on the stored val rows must equal the stored p0 (recorded, must be 0). Test seeds are scored on the train+val+test graph, in `eval()`
    mode, with the seed of the val run. Returns (p0 table of the test rows, record)."""
    frame = frame.sort(["timestamp", "tx_id"])
    sp = frame["split"].to_numpy()
    part = np.full(frame.height, "", dtype="U4")
    part[sp == "val"] = g.val_parts(frame["timestamp"].filter(pl.Series(sp == "val")), *cut)
    test = frame.filter(pl.col("split") == "test")
    raw_v, y_v = stored["score_raw"].to_numpy(), stored["label"].cast(pl.Int8).to_numpy()
    ok_v, fit_v = ~np.isnan(raw_v) & stored["p0"].is_not_null().to_numpy(), (stored["val_part"] == "fit").to_numpy()
    rec: dict = {"graph": "all" if bank is None else int(bank), "test_rows": test.height, "val_rows_stored": stored.height}
    if y_v[fit_v & ok_v].sum() < 1:  # the val run skipped this view (no positive in the val fit part): p0 stays null
        p0, rec["calibration"] = np.full(test.height, np.nan), "skipped: no val-fit positives"
        tx, raw_t = test["tx_id"].to_numpy(), np.full(test.height, np.nan)
    else:
        sig = g.has_signal(y_v[fit_v & ok_v], raw_v[fit_v & ok_v])
        cal, rec["calibration"] = g.fit_calibrator(raw_v[fit_v & ok_v], y_v[fit_v & ok_v], has_signal=sig)
        back = cal(raw_v[ok_v])
        rec["val_p0_max_abs_diff"] = float(np.max(np.abs(back - stored["p0"].to_numpy()[ok_v]))) if ok_v.any() else None
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"{ckpt}: the checkpoint of the val run is missing; the final run never trains")
        if test.height:
            res = MODELS[model_name][0].train_and_score(frame, clone, 0.0, 0, part == "fit", num_workers=workers, log=log, ckpt=ckpt, batch_size=batch_size, resume=True, score_splits=("test",))
            tx, raw_t = res["tx_id"], res["raw"]
        else:  # no test row (a check of the calibrator on a val-only frame)
            tx, raw_t = test["tx_id"].to_numpy(), np.zeros(0)
        p0 = np.full(len(raw_t), np.nan)
        okt = ~np.isnan(raw_t)
        p0[okt] = cal(raw_t[okt])
        rec["test_not_scored"] = int((~okt).sum())
    tab = pl.DataFrame({"tx_id": tx, "bank": pl.Series([bank] * len(tx), dtype=pl.Int64), "mode": [mode] * len(tx), "model": [model_name] * len(tx),
                        "split": ["test"] * len(tx), "val_part": pl.Series([None] * len(tx), dtype=pl.Utf8), "p0": pl.Series(p0).fill_nan(None),
                        "score_raw": pl.Series(raw_t.astype("f8")).fill_nan(None), "label": test["is_laundering"], "seed": [1] * len(tx), "commit": [cm] * len(tx)})  # fmt: skip
    return tab, rec


def final_run(a) -> None:
    """Week 14: score the test rows of every graph with the checkpoints of the val run. Needs FTM_FINAL_RUN=1 (scripts/run_final.py)."""
    from ftm import guards

    guards.require_final_run("the test split of the GNN baselines")
    cfg = io.load_config(a.data)
    pq, clone = cfg["parquet_dir"], Path(a.clone or MODELS[a.model][1]).expanduser()
    cut, cm, cols = val_cut(pq), commit(), multi_gnn.EDGE_COLS
    stored_all = pl.read_parquet(SCORES / a.data / f"p0_{a.model}__{a.mode}.parquet")  # time-guard: not transactions
    ckdir = MODEL_DIR / a.model
    splits, parts, recs, t0 = ["train", "val", "test"], [], [], time.time()
    if a.mode == "see_everything":
        frame = io.scan_transactions(pq, split=splits, columns=cols).collect()
        tab, rec = final_graph(frame, cut, a.mode, None, clone, cm, a.workers, ckdir / f"{a.data}_see.pt", a.batch_size, a.model, stored_all)
        parts.append(tab), recs.append(rec)
    else:
        banks = io.read_banks(cfg["views_dir"]).sort("rank")
        for b in a.banks or banks["bank"].head(TOP_BANKS).to_list():
            stored = stored_all.filter(pl.col("bank") == b)
            if stored.height == 0:  # skipped in the val run (too few train positives)
                continue
            f = io.scan_view(cfg["views_dir"], b, split=splits, columns=cols + EXTRA).collect().drop(EXTRA)
            tab, rec = final_graph(f, cut, a.mode, b, clone, cm, a.workers, ckdir / f"{a.data}_bank{b}.pt", a.batch_size, a.model, stored)
            parts.append(tab), recs.append(rec)
            log(f"bank {b}: {rec} {time.time() - t0:.0f}s")
    d = SCORES / a.data / "final"
    d.mkdir(parents=True, exist_ok=True)
    pl.concat(parts).write_parquet(d / f"p0_{a.model}__{a.mode}.parquet")
    bad = [r for r in recs if (r.get("val_p0_max_abs_diff") or 0) > 1e-9]
    RUNS / f"final_p0_{a.data}_{a.model}_{a.mode}.json".write_text(json.dumps({"commit": cm, "seconds": time.time() - t0, "graphs": recs}, indent=1, default=str))
    if bad:
        raise SystemExit(f"the calibrator does not reproduce the stored val p0 for {[r['graph'] for r in bad]}: do not use these test scores")
    log(f"final {a.model} {a.mode} done {time.time() - t0:.0f}s")


def combine(ds: str, name: str = "multi_gnn") -> None:
    """p0_<name>.parquet = whichever of the two per-mode files exist (GFP's p0 files are never touched)."""
    d = SCORES / ds
    fs = [d / f"p0_{name}__{m}.parquet" for m in ("one_bank", "see_everything") if (d / f"p0_{name}__{m}.parquet").exists()]
    pl.concat([pl.read_parquet(f) for f in fs]).write_parquet(d / f"p0_{name}.parquet")  # time-guard: not transactions


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
    ap.add_argument("--scratch", default=None, help="write every output (scores, run json, checkpoints) under this directory instead of data/scores, data/runs, models/: use it for smoke tests")
    ap.add_argument("--our-split", action="store_true")
    ap.add_argument("--final", action="store_true", help="week 14: score the TEST rows with the val run's checkpoints (needs FTM_FINAL_RUN=1; trains nothing)")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--mode", choices=["see_everything", "one_bank"], default="see_everything")
    ap.add_argument("--banks", type=int, nargs="*", default=None)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2048, help="2048 = reproduction config; 1024 if VRAM is short")
    ap.add_argument("--resume", action="store_true", help="see_everything: continue from models/multi_gnn/<ds>_see.pt (best epoch so far)")
    ap.add_argument("--workers", type=int, default=0, help="LinkNeighborLoader workers")
    ap.add_argument("--model", choices=sorted(MODELS), default="multi_gnn")
    ap.add_argument("--clone", default=None, help="default: the model's clone under ~/Documents/t1_external")
    ap.add_argument("--budget-s", type=int, default=8700, help="stop after the epoch that passes this many seconds")
    ap.add_argument("--run", action="store_true", help="--repro: launch it (otherwise only print the command)")
    a = ap.parse_args()
    if a.scratch:
        global SCORES, RUNS, MODEL_DIR
        SCORES, RUNS, MODEL_DIR = (Path(a.scratch) / n for n in ("scores", "runs", "models"))
        log(f"scratch run: outputs under {a.scratch}")
    if a.final:
        return final_run(a)
    if a.our_split:
        return our_split(a)
    if a.compare:
        if a.model != "multi_gnn":
            raise SystemExit("--compare is Multi-GNN vs GFP; the MEGA-GNN rows are in scripts/baselines_val_table.py")
        return compare(a.data)
    if not a.repro:
        raise SystemExit("give --repro, --our-split or --compare")
    mod, default_clone = MODELS[a.model]
    clone = Path(a.clone or default_clone).expanduser()
    if a.model == "mega_gnn":
        metrics = Path("data/mega_gnn/repro_metrics.jsonl").resolve()
        print(f"cd {clone} && PYTHONPATH=.:genagg T1_BUDGET_S={a.budget_s} T1_METRICS={metrics} {shlex.join(mod.repro_command())}")
    else:
        metrics = Path("data/multignn/pna_metrics.jsonl").resolve()
        print(f"cd {clone} && T1_BUDGET_S={a.budget_s} T1_METRICS={metrics} {shlex.join(mod.repro_command())}")
    if a.run:
        mod.launch(clone, metrics, a.budget_s).wait()


if __name__ == "__main__":
    main()
