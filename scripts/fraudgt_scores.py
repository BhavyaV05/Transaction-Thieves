"""FraudGT predictions (the 5070 Ti run, HI-Small) into T1's p0 pipeline: calibrate, score, compare.

Owner: T1 (Abhishek).  FraudGT runs in its own checkout (`ftm/baselines/README.md`, FraudGT section); its patch leaves in `<dir>`
`best.pt`, `pred_val.csv` (`tx_id,label,score`, uncalibrated sigmoid), `pred_meta.json`, and in the final run `pred_test.csv`.

  python scripts/fraudgt_scores.py --data hi_small --dump <dir>              # val: calibrate on the val `fit` part, p0 + metrics
  python scripts/fraudgt_scores.py --data hi_small --compare                 # paired Brier differences to GFP / Multi-GNN see-everything
  python scripts/fraudgt_scores.py --data hi_small --final --dump <dir>      # week 14 (FTM_FINAL_RUN=1): pred_test.csv through the stored calibrator

It is a see-everything (full graph) model. The `label` column of the dump is ignored: labels and the fit/cal/eval parts come from `ftm.data.io`
(train/val only), as in `scripts/train_gnn.py`. Every tx_id of the dump must be a val row of ours (anything else is an error). The calibrator is the
Multi-GNN see-everything one (`gfp_gbt.fit_calibrator` on the val `fit` rows, same signal rule). Rows FraudGT's neighbour sampling does not score
(about 0.25 %) get the val `fit` base rate (the constant predictor of `baselines_val_table.py`) in p0 and `scored = False`; their count and share are
in the json. `--scratch DIR` moves every output (smoke tests). Writes data/scores/<ds>/p0_fraudgt.parquet and data/runs/p0_<ds>_fraudgt_see_everything.json
(`--final`: data/scores/<ds>/final/p0_fraudgt.parquet and data/runs/final_p0_<ds>_fraudgt.json).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.baselines import gfp_gbt as g  # noqa: E402
from ftm.data import io  # noqa: E402

SCORES, RUNS = Path("data/scores"), Path("data/runs")  # `--scratch DIR` moves both
MODEL, MODE, SEED = "fraudgt", "see_everything", 42  # FraudGT's own cfg.seed (neighbour sampling is seeded with it)
TOL = 1e-6  # the stored calibrator must reproduce the stored val p0


def commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def read_dump(path: Path, rows: pl.DataFrame) -> pl.DataFrame:
    """The dump's (tx_id, score), checked against `rows` (our rows of that split): no duplicates, every tx_id one of ours. The dump's label is dropped."""
    d = pl.read_csv(path, schema_overrides={"tx_id": pl.Int64})  # time-guard: not transactions
    if not {"tx_id", "score"} <= set(d.columns):
        raise SystemExit(f"{path}: needs the columns tx_id and score, has {d.columns}")
    d = d.select("tx_id", pl.col("score").cast(pl.Float64))
    if d["tx_id"].n_unique() != d.height:
        raise SystemExit(f"{path}: duplicate tx_ids")
    if not d["score"].is_between(0.0, 1.0).all():
        raise SystemExit(f"{path}: scores outside [0, 1] (the dump is the uncalibrated sigmoid)")
    unknown = d.join(rows.select("tx_id"), on="tx_id", how="anti")
    if unknown.height:
        raise SystemExit(f"{path}: {unknown.height} tx_ids are not rows of ours (first: {unknown['tx_id'][0]}); wrong dataset or split?")
    return d


def val_rows(ds: str) -> pl.DataFrame:
    """Val rows (tx_id, label y, part fit/cal/eval) exactly as `scripts/train_gnn.py` derives them (time-sorted, parts from the val span)."""
    v = io.scan_transactions(io.load_config(ds)["parquet_dir"], split="val", columns=["tx_id", "timestamp", "is_laundering"]).collect().sort(["timestamp", "tx_id"])
    cut = (v["timestamp"].min(), v["timestamp"].max())
    return v.select("tx_id", pl.col("is_laundering").cast(pl.Int8).alias("y"), pl.Series("val_part", g.val_parts(v["timestamp"], *cut)))


def calibrate(raw: np.ndarray, y: np.ndarray, fit: np.ndarray):
    """The Multi-GNN see-everything calibrator (`scripts/train_gnn.py: run_graph`) on the scored val `fit` rows. Returns (callable, name, has_signal)."""
    ok = ~np.isnan(raw)
    f = fit & ok
    if y[f].sum() < 1:
        raise SystemExit("no laundering row among the scored val fit rows: cannot calibrate")
    sig = g.has_signal(y[f], raw[f])
    cal, name = g.fit_calibrator(raw[f], y[f], has_signal=sig)
    return cal, name, sig


def p0_table(rows: pl.DataFrame, dump: pl.DataFrame, cal, fill: float, split: str, cm: str) -> pl.DataFrame:
    """One row per row of ours of the split: calibrated p0 where FraudGT scored the edge, `fill` where it did not (`scored` False, score_raw null)."""
    t = rows.join(dump.rename({"score": "score_raw"}), on="tx_id", how="left")
    raw = t["score_raw"].fill_null(float("nan")).to_numpy()
    ok = ~np.isnan(raw)
    p0 = np.full(len(raw), fill)
    p0[ok] = cal(raw[ok])
    n = t.height
    part = t["val_part"] if split == "val" else pl.Series([None] * n, dtype=pl.Utf8)
    return pl.DataFrame({"tx_id": t["tx_id"], "bank": pl.Series([None] * n, dtype=pl.Int64), "mode": [MODE] * n, "model": [MODEL] * n, "split": [split] * n,
                         "val_part": part, "p0": p0, "score_raw": t["score_raw"], "label": t["y"].cast(pl.Boolean), "seed": [SEED] * n, "commit": [cm] * n,
                         "scored": ok})  # fmt: skip


def val_run(a, ds: str) -> None:
    dump = Path(a.dump)
    rows = val_rows(ds)
    d = read_dump(dump / "pred_val.csv", rows)
    t = rows.join(d.rename({"score": "raw"}), on="tx_id", how="left")
    raw, y, part = t["raw"].fill_null(float("nan")).to_numpy(), t["y"].to_numpy(), t["val_part"].to_numpy()
    fit = part == "fit"
    cal, name, sig = calibrate(raw, y, fit)
    fill = float(y[fit].mean())  # the val fit base rate (`baselines_val_table.py`: the constant predictor)
    cm = commit()
    tab = p0_table(rows, d, cal, fill, "val", cm)
    out = SCORES / ds
    out.mkdir(parents=True, exist_ok=True)
    tab.write_parquet(out / "p0_fraudgt.parquet")
    ev, ft = tab.filter(pl.col("val_part") == "eval"), tab.filter(pl.col("val_part") == "fit")
    thr = g.metrics(ft["label"].cast(pl.Int8).to_numpy(), ft["p0"].to_numpy())["thr"]
    ye = ev["label"].cast(pl.Int8).to_numpy()
    m = g.metrics(ye, ev["p0"].to_numpy(), thr)
    sc = ev.filter(pl.col("scored"))  # the same metrics on the scored rows only, for reference
    ms = g.metrics(sc["label"].cast(pl.Int8).to_numpy(), sc["p0"].to_numpy(), thr)
    n_un = int((~tab["scored"]).sum())
    meta = {"dataset": ds, "model": MODEL, "mode": MODE, "commit": cm, "seed": SEED, "dump": str(dump), "pred_meta": json.loads((dump / "pred_meta.json").read_text()) if (dump / "pred_meta.json").exists() else None,
            "val_rows": tab.height, "scored": int(tab["scored"].sum()), "unscored": n_un, "unscored_share": n_un / tab.height, "unscored_fill": fill, "unscored_fill_is": "val fit base rate",
            "calibration": name, "fit_signal": bool(sig), "fit_positives": int(y[fit & ~np.isnan(raw)].sum()),
            "eval": {k: m.get(k) for k in ("n", "n_pos", "brier", "pr_auc", "f1", "precision", "recall", "thr")} | {"brier_const": float(np.mean((fill - ye) ** 2))},
            "eval_scored_only": {k: ms.get(k) for k in ("n", "n_pos", "brier", "pr_auc", "f1")},
            "note": "val look-ahead as the other GNNs (val edges scored on the graph of train + val); the dump's label column was ignored"}  # fmt: skip
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"p0_{ds}_fraudgt_see_everything.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({k: meta[k] for k in ("val_rows", "scored", "unscored", "unscored_share", "calibration", "eval", "eval_scored_only")}, indent=1, default=str))


def final_run(a, ds: str) -> None:
    """Week 14: the stored calibrator (refit from the stored val raw scores, as `train_gnn.py --final`; no new fit) on `pred_test.csv`, the same fill."""
    from ftm import guards

    guards.require_final_run("the test split of FraudGT")
    dump = Path(a.dump)
    stored = pl.read_parquet(SCORES / ds / "p0_fraudgt.parquet")  # time-guard: not transactions
    meta = json.loads((RUNS / f"p0_{ds}_fraudgt_see_everything.json").read_text())
    raw_v, y_v, fit_v = stored["score_raw"].fill_null(float("nan")).to_numpy(), stored["label"].cast(pl.Int8).to_numpy(), (stored["val_part"] == "fit").to_numpy()
    cal, name, _ = calibrate(raw_v, y_v, fit_v)
    fill = float(meta["unscored_fill"])
    if name != meta["calibration"] or abs(float(y_v[fit_v].mean()) - fill) > 1e-12:
        raise SystemExit("the stored val table does not give the stored calibrator or fill: do not use these test scores")
    # 1. the stored calibrator on the dump's pred_val.csv must give the stored val p0 (this also checks that the dump did not change)
    rv = val_rows(ds)
    dv = read_dump(dump / "pred_val.csv", rv)
    back = p0_table(rv, dv, cal, fill, "val", "")
    j = stored.select("tx_id", pl.col("p0").alias("p_stored")).join(back.select("tx_id", "p0", "scored"), on="tx_id", how="full", coalesce=True)
    diff = float(np.max(np.abs(j["p_stored"].to_numpy() - j["p0"].to_numpy()))) if j.height == stored.height and not j["p0"].has_nulls() else float("inf")
    rec = {"commit": commit(), "val_p0_max_abs_diff": diff, "calibration": name, "unscored_fill": fill}
    # 2. the test rows
    rt = io.scan_transactions(io.load_config(ds)["parquet_dir"], split="test", columns=["tx_id", "timestamp", "is_laundering"]).collect().sort(["timestamp", "tx_id"])
    rt = rt.select("tx_id", pl.col("is_laundering").cast(pl.Int8).alias("y"), pl.lit(None, dtype=pl.Utf8).alias("val_part"))
    dt = read_dump(dump / "pred_test.csv", rt)
    tab = p0_table(rt, dt, cal, fill, "test", rec["commit"])
    out = SCORES / ds / "final"
    out.mkdir(parents=True, exist_ok=True)
    tab.write_parquet(out / "p0_fraudgt.parquet")
    rec |= {"test_rows": tab.height, "scored": int(tab["scored"].sum()), "unscored": int((~tab["scored"]).sum()), "unscored_share": float((~tab["scored"]).mean())}
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"final_p0_{ds}_fraudgt.json").write_text(json.dumps(rec, indent=1, default=str))
    if diff > TOL:
        raise SystemExit(f"the stored calibrator does not reproduce the stored val p0 (max abs diff {diff}): do not use these test scores")
    print(f"final fraudgt: {rec}")


def compare(ds: str) -> None:
    """Paired cluster bootstrap (attempt + shared accounts, hub cap 20 as `baselines_val_table.py`) of the Brier difference FraudGT minus GFP /
    Multi-GNN see-everything on the val `eval` rows all three scored (FraudGT's unscored rows carry the fill). Val only."""
    from ftm.eval import bootstrap as bs

    pq = io.load_config(ds)["parquet_dir"]
    fr = pl.read_parquet(SCORES / ds / "p0_fraudgt.parquet").filter(pl.col("val_part") == "eval")  # time-guard: not transactions
    ref = {"gfp_lgbm": g.read_p0(ds, mode=MODE, split="val"), "multi_gnn": pl.read_parquet(Path("data/scores") / ds / "p0_multi_gnn.parquet").filter((pl.col("mode") == MODE) & (pl.col("split") == "val"))}  # fmt: skip
    tx = io.scan_transactions(pq, split="val", columns=["tx_id", "src_id", "dst_id"]).collect()
    pat = io.scan_patterns(pq, split="val").select("tx_id", "attempt_id").collect()
    j = fr.select("tx_id", pl.col("p0").alias("p_fraudgt"), pl.col("label").cast(pl.Int8).alias("y"))
    for k, d in ref.items():
        j = j.join(d.filter(pl.col("val_part") == "eval").select("tx_id", pl.col("p0").alias(f"p_{k}")).drop_nulls(), on="tx_id")
    j = j.join(tx, on="tx_id", how="left").join(pat, on="tx_id", how="left").sort("tx_id")
    cl = bs.cluster_ids(j["attempt_id"].to_list(), [j["src_id"].to_list(), j["dst_id"].to_list()], hub_cap=20)
    y = j["y"].to_numpy()
    out = {"dataset": ds, "n": int(len(y)), "positives": int(y.sum()), "clusters": bs.cluster_report(cl), "hub_cap": 20, "brier_fraudgt": float(np.mean((j["p_fraudgt"].to_numpy() - y) ** 2))}
    for k in ref:
        r = bs.brier_diff(y, j["p_fraudgt"].to_numpy(), j[f"p_{k}"].to_numpy(), cl, n_boot=2000)  # diff = fraudgt minus the reference
        out[f"brier_fraudgt_minus_{k}_see_everything"] = {"diff": r["diff"], "ci95": r["ci"], "brier_ref": float(np.mean((j[f"p_{k}"].to_numpy() - y) ** 2))}
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"fraudgt_vs_see_everything_{ds}.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--dump", default=None, help="the FraudGT run directory (best.pt, pred_val.csv, pred_meta.json; pred_test.csv after the final run)")
    ap.add_argument("--compare", action="store_true", help="paired Brier differences to GFP and Multi-GNN see-everything (needs the val p0 of all three)")
    ap.add_argument("--final", action="store_true", help="week 14: score pred_test.csv with the stored calibrator (needs FTM_FINAL_RUN=1)")
    ap.add_argument("--scratch", default=None, help="write every output under this directory instead of data/scores and data/runs (smoke tests)")
    a = ap.parse_args()
    global SCORES, RUNS
    if a.scratch:
        SCORES, RUNS = Path(a.scratch) / "scores", Path(a.scratch) / "runs"
    if a.data != "hi_small":
        raise SystemExit("FraudGT is HI-Small only (the full-graph model does not fit HI-Medium)")
    if a.compare:
        return compare(a.data)
    if not a.dump:
        raise SystemExit("--dump <FraudGT run directory> is needed")
    return final_run(a, a.data) if a.final else val_run(a, a.data)


if __name__ == "__main__":
    main()
