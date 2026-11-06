"""p0 v2 (findings 16-17): recalibrate the one-bank p0 on val `cal` (frozen, hashed map) and cross-fit a train map; alert sets stay frozen.

  python scripts/recal_p0.py --data hi_medium            # writes data/scores/<ds>/p0_one_bank_v2.{parquet,json}, data/alerts/<ds>_v2/, configs/t1_final/p0_recal_<ds>.json
  python scripts/recal_p0.py --data hi_medium --apply-test   # LOCKED (final run only): apply the frozen map to the test p0

Val: maps (identity, intercept shift, Platt, isotonic on logit/p0), each fit on all `cal` rows or on the cal alerts only; chosen by 5-block time CV inside `cal`: the lowest
Brier on held-out ALERTS (the rows downstream systems see) among candidates that do not worsen the all-rows Brier. `eval` is never used to fit or choose; its Brier is reported for every variant (descriptive).
Train: the OOF train p0 is mapped by a map fit on the other time folds (3 day blocks) of train, kind chosen by the same fold-CV on train alerts.
Nothing here changes which rows are alerts: the alert key is (tx_id, bank) of data/alerts/<ds>/{train,val}.parquet, and every map is monotone in p0_v1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.baselines import p0_recal as R  # noqa: E402
from ftm.data import io  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load(ds: str, cfg: dict) -> pl.DataFrame:
    s = pl.read_parquet(f"data/scores/{ds}/p0_one_bank.parquet").filter(pl.col("p0").is_not_null() & pl.col("score_raw").is_not_null())
    tx = io.scan_transactions(cfg["parquet_dir"], split=["train", "val"], columns=["tx_id", "timestamp", "day"]).collect()
    al = pl.concat([pl.read_parquet(Path(cfg["alerts_dir"]) / f"{x}.parquet", columns=["tx_id", "bank"]) for x in ("train", "val")]).with_columns(pl.lit(True).alias("is_alert"))
    return s.join(tx, on="tx_id").join(al, on=["tx_id", "bank"], how="left").with_columns(pl.col("is_alert").fill_null(False)).sort("timestamp", "tx_id", "bank")


def _arr(d: pl.DataFrame, c: str = "p0"):
    return d[c].to_numpy(), d["label"].cast(pl.Float64).to_numpy()


def val_tables(s: pl.DataFrame, maps: dict) -> dict:
    """Eval (and cal/fit) Brier and mean p0 vs label rate per variant, on alerts and on all rows."""
    out = {}
    for part in ("fit", "cal", "eval"):
        for pop, sel in (("alerts", pl.col("is_alert")), ("rows", pl.lit(True))):
            d = s.filter((pl.col("val_part") == part) & sel)
            p, y = _arr(d)
            row = {"n": d.height, "label_rate": float(y.mean()), "raw": {"brier": float(np.mean((d["score_raw"].to_numpy() - y) ** 2)), "mean": float(d["score_raw"].mean())}}
            for name, m in maps.items():
                q = R.apply_map(m, p)
                row[name] = {"brier": float(np.mean((q - y) ** 2)), "mean": float(q.mean())}
            out[f"{part}/{pop}"] = row
    return out


def select(p, y, al, groups):
    """Candidates = kind x fit population (all rows | alerts only); CV Brier on held-out alerts and on all held-out rows; pick by `select_candidate`."""
    cv, cvr = {}, {}
    for pop, fm in (("rows", None), ("alerts", al)):
        for k in R.KINDS:
            if pop == "alerts" and k == "identity":
                continue
            cv[f"{k}_{pop}"] = R.cv_brier(k, p, y, groups=groups, score_mask=al, fit_mask=fm)
            cvr[f"{k}_{pop}"] = R.cv_brier(k, p, y, groups=groups, fit_mask=fm)
    return cv, cvr, R.select_candidate(cv, cvr)


def run(ds: str, args) -> dict:
    cfg = io.load_config(ds)
    s = load(ds, cfg)
    val, train = s.filter(pl.col("split") == "val"), s.filter(pl.col("split") == "train")
    cal = val.filter(pl.col("val_part") == "cal")
    p, y = _arr(cal)
    al = cal["is_alert"].to_numpy()
    cv, cvr, kind = select(p, y, al, None)
    maps_row = {f"{k}_rows": R.fit_map(k, p, y) for k in R.KINDS}  # fit on all cal rows
    maps_alert = {f"{k}_alerts": R.fit_map(k, p[al], y[al]) for k in R.KINDS if k != "identity"}  # fit on the cal alerts only
    tables = val_tables(val, {"v1": {"kind": "identity"}} | maps_row | maps_alert)
    chosen = (maps_row | maps_alert)[kind]
    cfg_out = ROOT / "configs" / "t1_final" / f"p0_recal_{ds}.json"
    saved = R.save_map(chosen, cfg_out, dataset=ds, fit_parts=["cal"], population="all one-bank view rows of val cal" if kind.endswith("rows") else "alerts of val cal", n_fit=cal.height, n_pos=int(y.sum()),
                       selection_cv_brier_on_cal_alerts=cv, selection_cv_brier_on_cal_rows=cvr, selected=kind, applies_to="p0 v1 of one-bank rows (train, val, and test in the final run)")  # fmt: skip

    # train: cross-fitted by day blocks of the train split
    d0, d1 = cfg["split_days"]["train"]
    edges = np.linspace(d0 - 1, d1, 4)
    fold = np.clip(np.searchsorted(edges, train["day"].to_numpy() - 0.5), 1, 3) - 1
    pt, yt = _arr(train)
    alt = train["is_alert"].to_numpy()
    cvt, cvtr, kt = select(pt, yt, alt, fold)
    p2t = np.empty(len(pt))
    fold_maps = {}
    for f in range(3):
        te = fold == f
        kk, pop = kt.split("_")
        m = R.fit_map(kk, pt[~te & (alt if pop == "alerts" else True)], yt[~te & (alt if pop == "alerts" else True)])
        fold_maps[f] = {k: v for k, v in m.items() if k not in ("x", "y")} | ({"n_knots": len(m["x"])} if "x" in m else {})
        p2t[te] = R.apply_map(m, pt[te])
    tr_rep = {"selected": kt, "cv_brier_on_train_alerts": cvt, "cv_brier_on_train_rows": cvtr, "fold_maps": fold_maps,
              "alerts": {"n": int(alt.sum()), "label_rate": float(yt[alt].mean()), "mean_p0_v1": float(pt[alt].mean()), "mean_p0_v2": float(p2t[alt].mean()),
                         "brier_v1": float(np.mean((pt[alt] - yt[alt]) ** 2)), "brier_v2": float(np.mean((p2t[alt] - yt[alt]) ** 2))},
              "rows": {"n": len(pt), "label_rate": float(yt.mean()), "mean_p0_v1": float(pt.mean()), "mean_p0_v2": float(p2t.mean()),
                       "brier_v1": float(np.mean((pt - yt) ** 2)), "brier_v2": float(np.mean((p2t - yt) ** 2))},
              "alerts_by_fold": {int(f): {"label_rate": float(yt[alt & (fold == f)].mean()), "mean_p0_v1": float(pt[alt & (fold == f)].mean()), "mean_p0_v2": float(p2t[alt & (fold == f)].mean())} for f in range(3)}}  # fmt: skip

    # v2 tables. train: cross-fitted; val: the frozen cal map (fit and cal rows are in-sample for it, eval is clean). rows without an alert-table key keep the same map.
    v2 = s.with_columns(pl.col("p0").alias("p0_v1")).drop("is_alert")
    new = np.where((s["split"] == "train").to_numpy(), np.nan, R.apply_map(chosen, s["p0"].to_numpy()))
    ti = np.flatnonzero((s["split"] == "train").to_numpy())  # train rows of s are in the same time order as `train`
    new[ti] = p2t
    v2 = v2.with_columns(pl.Series("p0", new))
    full = pl.read_parquet(f"data/scores/{ds}/p0_one_bank.parquet")
    keep = full.join(v2.select("tx_id", "bank", "p0", "p0_v1"), on=["tx_id", "bank"], how="left", suffix="_v2")
    keep = keep.with_columns(pl.col("p0").alias("p0_v1"), pl.col("p0_v2").alias("p0")).drop("p0_v2").with_columns(pl.col("p0").fill_null(pl.col("p0_v1") * float("nan")).fill_nan(None))
    out_p = Path(f"data/scores/{ds}/p0_one_bank_v2.parquet")
    keep.select([c for c in full.columns] + ["p0_v1"]).write_parquet(out_p)
    meta = {"calibrator_parts": ["cal"], "val_map": str(cfg_out.relative_to(ROOT)), "val_map_sha256": saved["sha256"], "train": "cross-fitted by 3 day blocks, own map per fold", "p0_v1": "column p0_v1"}
    out_p.with_suffix(".json").write_text(json.dumps(meta, indent=1))

    # alert tables v2: same alert rows, p0 replaced
    adir = Path(cfg["alerts_dir"] + "_v2")
    adir.mkdir(parents=True, exist_ok=True)
    for x in ("train", "val"):
        a = pl.read_parquet(Path(cfg["alerts_dir"]) / f"{x}.parquet").rename({"p0": "p0_v1"})
        a = a.join(keep.select("tx_id", "bank", "p0"), on=["tx_id", "bank"], how="left")
        assert a["p0"].null_count() == 0 and a.height == pl.read_parquet(Path(cfg["alerts_dir"]) / f"{x}.parquet").height
        a.select(["alert_id", "tx_id", "bank", "timestamp", "p0", "label", *[c for c in a.columns if c not in ("alert_id", "tx_id", "bank", "timestamp", "p0", "label")]]).write_parquet(adir / f"{x}.parquet")
    for f in Path(cfg["alerts_dir"]).glob("*.parquet"):
        if f.stem not in ("train", "val", "test") and not (adir / f.name).exists():
            (adir / f.name).symlink_to(f.resolve())  # top_features etc. are p0-free
    rep = {"dataset": ds, "selected_val": kind, "val_cv_brier_on_cal_alerts": cv, "val_cv_brier_on_cal_rows": cvr, "val_map": {k: v for k, v in saved.items() if k not in ("x", "y")}, "val_tables": tables, "train": tr_rep}
    Path(f"data/runs/p0_recal_{ds}.json").write_text(json.dumps(rep, indent=1))
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="hi_medium")
    ap.add_argument("--apply-test", action="store_true", help="LOCKED: final run only")
    a = ap.parse_args()
    if a.apply_test:
        from ftm import guards

        guards.require_final_run("applying the frozen p0 map to test p0")
        raise SystemExit("final run: use ftm.baselines.p0_recal.load_map(configs/t1_final/p0_recal_<ds>.json) and apply_map to the test p0")
    rep = run(a.data, a)
    print(json.dumps({"selected_val": rep["selected_val"], "cv": rep["val_cv_brier_on_cal_alerts"], "train": {k: rep["train"][k] for k in ("selected", "alerts", "rows", "alerts_by_fold")}}, indent=1))
    for k, v in rep["val_tables"].items():
        print(k, v["n"], round(v["label_rate"], 5), {n: (round(c["brier"], 6), round(c["mean"], 5)) for n, c in v.items() if isinstance(c, dict)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
