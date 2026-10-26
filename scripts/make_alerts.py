"""Conformal intake -> train/val/test alert sets and the frozen 1,000-alert subset.

Owner: T1 (Abhishek).  Run from the repo root, e.g. python scripts/make_alerts.py --data hi_small

Reads data/scores/<ds>/p0_one_bank.parquet (scripts/train_gfp.py), writes data/alerts/<ds>/{train,val}.parquet
and data/runs/alerts_<ds>.json (miss rates vs targets, sweep, ablation, test-count projection).
The test alert set (`--final`, week 14, needs FTM_FINAL_RUN=1 and the test p0 of `scripts/train_gfp.py --final`) and the 1,000-alert subset (T4's run_final.py,
`ftm.intake.alerts.stratified_subset`) are not built here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.data import io  # noqa: E402
from ftm.intake import alerts as al  # noqa: E402
from ftm.intake import conformal as cf  # noqa: E402


def thresholds(cal: pl.DataFrame, a_l: float, a_c: float, cfg: dict) -> dict:
    p, y = cal["p0"].to_numpy(), cal["label"].cast(pl.Int8).to_numpy()
    d = (cal["timestamp"] - cal["timestamp"].min()).dt.total_seconds().to_numpy() / 86400  # fractional days: the cal part is < 1 day
    tl, tc = cf.mondrian_thresholds(p, y, a_l, a_c, days=d, half_life_days=cfg["half_life_days"])
    lam = cf.selective_clear_threshold(p, y, cfg["scrc_beta"], cfg["scrc_delta"]) if cfg.get("scrc_beta") else None
    # alert <=> launder in the prediction set <=> p >= t_launder. With apply_scrc the auto-clear region is also
    # capped at lam (alerts widen down to lam, never narrow); otherwise the bound is only reported.
    t_alert = min(tl, lam) if (lam is not None and cfg.get("apply_scrc")) else tl
    ub = cf.selective_risk_upper_bound(p, y, t_alert, cfg["scrc_delta"])
    return {"t_launder": tl, "t_clean": tc, "scrc_lambda": lam, "t_alert": t_alert, "cleared_risk_ub": ub["upper_bound"]}


def summarize(df: pl.DataFrame, t_alert: float, days: float) -> dict:
    p, y = df["p0"].to_numpy(), df["label"].cast(pl.Int8).to_numpy()
    a = p >= t_alert
    return {"alerts": int(a.sum()), "positives": int(y[a].sum()), "alerts_per_day": a.sum() / days,
            "positives_per_day": float(y[a].sum() / days), "recall": float(y[a].sum() / max(y.sum(), 1)),
            "precision": float(y[a].sum() / max(a.sum(), 1)), "laundering_rows": int(y.sum())}  # fmt: skip


def part_report(df: pl.DataFrame, th: dict, a_l: float, a_c: float) -> dict:
    """Alerts, positives, empty sets, achieved vs target miss and the intake miss rate for one set of p0 rows.

    Empty set = neither label in the prediction set (t_clean < p0 < t_launder); not an alert, counted as an
    intake miss (decision 5). Intake miss (end-to-end, decision 2) = share of laundering rows not alerted,
    by view row and by unique transaction (a transaction is alerted if any bank view alerts it).
    """
    p, y = df["p0"].to_numpy(), df["label"].cast(pl.Int8).to_numpy()
    hl, hc = cf.prediction_sets(p, th["t_launder"], th["t_clean"])
    alert = p >= th["t_alert"]
    miss = cf.achieved_miss_rates(p, y, th["t_launder"], th["t_clean"])
    tx = df.with_columns(pl.Series("a", alert), pl.col("label").cast(pl.Int8).alias("y")).group_by("tx_id").agg(pl.col("a").any(), pl.col("y").max())
    pos_tx, hit_tx = int(tx["y"].sum()), int(tx.filter(pl.col("a"))["y"].sum())
    return {"rows": df.height, "laundering_rows": int(y.sum()), "alerts": int(alert.sum()), "positives": int(y[alert].sum()),
            "empty_set_rows": int((~hl & ~hc).sum()), "empty_set_laundering": int(y[~hl & ~hc].sum()),
            "launder_miss_target": a_l, "launder_miss_achieved": miss["launder_miss"],
            "clean_miss_target": a_c, "clean_miss_achieved": miss["clean_miss"],
            "intake_miss_rows": 1 - float(y[alert].sum()) / max(int(y.sum()), 1),
            "laundering_tx": pos_tx, "laundering_tx_alerted": hit_tx, "intake_miss_tx": 1 - hit_tx / max(pos_tx, 1)}  # fmt: skip


def final_run(ds: str, cfg: dict) -> None:
    """Week 14: the TEST alert set = the frozen val threshold applied to the test one-bank p0 (data/scores/<ds>/final/, written by
    `scripts/train_gfp.py --final`). Locked: needs FTM_FINAL_RUN=1 (scripts/run_final.py). Nothing is calibrated or tuned on test.
    Writes data/alerts/<ds>/test.parquet (schema of train/val, `val_part` null, plus `test_tail`: the sparse days after the bulk, to be
    reported separately, decision 4) and data/runs/alerts_<ds>_test.json (counts, threshold, the val projection for comparison)."""
    from ftm import guards
    from ftm.baselines import gfp_gbt as g

    guards.require_final_run("the test alert set")
    dcfg = io.load_config(ds)
    pq = dcfg["parquet_dir"]
    val_run = json.loads(Path(f"data/runs/alerts_{ds}.json").read_text())
    t_alert = val_run["thresholds"]["t_alert"]
    p0 = g.read_p0(ds, mode="one_bank", split="test").filter(pl.col("p0").is_not_null() & pl.col("p0").is_not_nan())
    tx = io.scan_transactions(pq, split="test", columns=["tx_id", "timestamp", "day", "from_bank", "to_bank"]).collect()
    pat = io.scan_patterns(pq, split="test").select("tx_id", "attempt_id", "typology").collect()
    att = io.scan_attempts(pq, split="test").select("attempt_id", "spans_split").collect()
    d = al.frozen_alerts(p0.join(tx, on="tx_id"), t_alert).join(pat, on="tx_id", how="left")
    d = al.test_alert_ids(d)
    d = al.add_direction(al.add_span_flag(d, att)).with_columns((pl.col("day") > cfg["bulk_days"][ds]).alias("test_tail"))
    cols = ["alert_id", "tx_id", "bank", "timestamp", "p0", "label", "attempt_id", "typology", "split", "val_part", "spans_split", "direction", "counterparty_bank", "test_tail"]
    d = d.with_columns(pl.col("val_part").cast(pl.Utf8))  # null for test rows
    adir = Path(dcfg["alerts_dir"])
    d.select(cols).write_parquet(adir / "test.parquet")
    bulk = d.filter(~pl.col("test_tail"))
    out = {"dataset": ds, "t_alert_from_val": t_alert, "alpha_launder": val_run["alpha_launder_used"], "alerts": d.height, "positives": int(d["label"].sum()),
           "bulk_alerts": bulk.height, "bulk_positives": int(bulk["label"].sum()), "tail_alerts": d.height - bulk.height, "unique_tx": d["tx_id"].n_unique(),
           "projection_from_val": next(s for s in val_run["sweep"] if s["alpha_launder"] == val_run["alpha_launder_used"])["proj_test_alerts"], "targets": cfg["targets"]}  # fmt: skip
    Path(f"data/runs/alerts_{ds}_test.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--config", default="configs/intake/conformal.yaml")
    ap.add_argument("--alpha-launder", type=float, default=None)
    ap.add_argument("--final", action="store_true", help="week 14: the test alert set from the frozen val threshold (needs FTM_FINAL_RUN=1)")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.final:
        return final_run(a.data, cfg)
    a_l = a.alpha_launder if a.alpha_launder is not None else cfg["alpha_launder"][a.data]
    a_c = cfg["alpha_clean"]
    dcfg = io.load_config(a.data)
    pq = dcfg["parquet_dir"]
    p0 = pl.read_parquet(Path("data/scores") / a.data / "p0_one_bank.parquet").filter(pl.col("p0").is_not_null() & pl.col("p0").is_not_nan())
    tx = io.scan_transactions(pq, split=["train", "val"], columns=["tx_id", "timestamp", "day", "from_bank", "to_bank"]).collect()
    p0 = p0.join(tx, on="tx_id")
    pat = io.scan_patterns(pq, split=["train", "val"]).select("tx_id", "attempt_id", "typology").collect()
    att = io.scan_attempts(pq, split=["train", "val"]).select("attempt_id", "spans_split").collect()
    val_days = dcfg["split_days"]["val"]
    n_val_days = val_days[1] - val_days[0] + 1
    test_bulk_days = cfg["bulk_days"][a.data] - val_days[1]

    val = p0.filter(pl.col("split") == "val")
    cal = val.filter(pl.col("val_part") == "cal")
    ev = val.filter(pl.col("val_part") == "eval")
    day_frac = 0.25  # share of the val time span covered by each of cal and eval
    out: dict = {"dataset": a.data, "config": cfg, "alpha_launder_used": a_l, "n_cal": cal.height,
                 "n_cal_pos": int(cal["label"].sum()), "n_eval": ev.height, "n_eval_pos": int(ev["label"].sum())}  # fmt: skip

    th = thresholds(cal, a_l, a_c, cfg)
    miss = cf.achieved_miss_rates(ev["p0"].to_numpy(), ev["label"].cast(pl.Int8).to_numpy(), th["t_launder"], th["t_clean"])
    cleared = ev.filter(pl.col("p0") < th["t_alert"])
    out["thresholds"] = th
    hl, hc = cf.prediction_sets(ev["p0"].to_numpy(), th["t_launder"], th["t_clean"])
    yy = ev["label"].cast(pl.Int8).to_numpy()
    out["val_parts"] = {part: part_report(val.filter(pl.col("val_part") == part), th, a_l, a_c) for part in ("fit", "cal", "eval")}
    out["val_parts"]["val_all"] = part_report(val, th, a_l, a_c)
    out["set_types_eval"] = {name: {"rows": int(m.sum()), "laundering": int(yy[m].sum())} for name, m in (
        ("launder_and_clean", hl & hc), ("launder_only", hl & ~hc), ("clean_only", ~hl & hc), ("empty", ~hl & ~hc))}  # fmt: skip
    # time-weighting ablation: same target, different half-lives; achieved miss on the untouched eval part
    tw = []
    for hl in dict.fromkeys((None, cfg["half_life_days"], 0.25, 0.1)):
        c2 = {**cfg, "half_life_days": hl}
        t2 = thresholds(cal, a_l, a_c, c2)
        m2 = cf.achieved_miss_rates(ev["p0"].to_numpy(), ev["label"].cast(pl.Int8).to_numpy(), t2["t_launder"], t2["t_clean"])
        tw.append({"half_life_days": hl, "t_launder": t2["t_launder"], "t_clean": t2["t_clean"],
                   "launder_miss": m2["launder_miss"], "clean_miss": m2["clean_miss"],
                   "alerts_eval": int((ev["p0"].to_numpy() >= t2["t_launder"]).sum())})  # fmt: skip
    out["time_weighting_ablation"] = tw
    out["achieved_vs_target"] = {
        "launder_miss_target": a_l, "launder_miss_achieved": miss["launder_miss"], "n_launder_eval": miss["n_launder"],
        "clean_miss_target": a_c, "clean_miss_achieved": miss["clean_miss"], "n_clean_eval": miss["n_clean"],
        "selective_risk_target": cfg["scrc_beta"],
        "selective_risk_achieved": float(cleared["label"].cast(pl.Float64).mean()) if cleared.height else None,
        "n_auto_cleared_eval": cleared.height,
    }  # fmt: skip
    # eval-part summary and the fixed-window ablation on the same rows
    days_eval = n_val_days * day_frac
    out["eval_conformal"] = summarize(ev, th["t_alert"], days_eval)
    fw = cfg["fixed_window"]
    fa = cf.fixed_window(ev["p0"].to_numpy(), *fw)
    y = ev["label"].cast(pl.Int8).to_numpy()
    out["eval_fixed_window"] = {"window": fw, "alerts": int(fa.sum()), "positives": int(y[fa].sum()),
                                "recall": float(y[fa].sum() / max(y.sum(), 1))}  # fmt: skip
    # alpha sweep: alerts per val day on cal+eval, and the projection to test (val rates only)
    sweep = []
    ce = pl.concat([cal, ev])
    for al_ in sorted({0.05, 0.2, 0.4, 0.5, 0.55, 0.6, 0.62, 0.64, 0.66, 0.7, 0.8, 0.9, 0.95, a_l}):
        t = thresholds(cal, al_, a_c, cfg)
        s = summarize(ce, t["t_alert"], n_val_days * 2 * day_frac)
        tr = summarize(p0.filter(pl.col("split") == "train"), t["t_alert"], 1.0)
        sweep.append({"alpha_launder": al_, "t_alert": t["t_alert"], "cleared_risk_ub": t["cleared_risk_ub"], "val_recall": s["recall"], "val_precision": s["precision"],
                      "proj_test_alerts": al.project_test_count(s["alerts_per_day"], test_bulk_days),
                      "proj_test_positives": al.project_test_count(s["positives_per_day"], test_bulk_days),
                      "train_alerts": tr["alerts"], "train_positives": tr["positives"], "val_alerts_all": summarize(val, t["t_alert"], 1)["alerts"]})  # fmt: skip
    out["sweep"] = sweep
    out["test_bulk_days"] = test_bulk_days
    out["projection_note"] = "PROJECTION from val alert rates per day x bulk test days; no test row was read."

    # alert sets for train and val at the chosen alpha
    adir = Path(dcfg["alerts_dir"])
    adir.mkdir(parents=True, exist_ok=True)
    cols = ["alert_id", "tx_id", "bank", "timestamp", "p0", "label", "attempt_id", "typology", "split", "val_part",
            "spans_split", "direction", "counterparty_bank"]  # fmt: skip
    counts = {}
    for split in ("train", "val"):
        d = p0.filter((pl.col("split") == split) & (pl.col("p0") >= th["t_alert"])).join(pat, on="tx_id", how="left")
        d = d.sort(["timestamp", "tx_id", "bank"]).with_columns(
            (pl.lit(split[0].upper()) + pl.int_range(pl.len()).cast(pl.Utf8).str.zfill(6)).alias("alert_id"))  # fmt: skip
        d = al.add_span_flag(d, att)
        d = al.add_direction(d)
        d.select(cols).write_parquet(adir / f"{split}.parquet")
        # attempt-level recall: attempts with >= 1 laundering tx in this split that have >= 1 alerted tx (any bank view)
        att_all = (pat.join(p0.filter(pl.col("split") == split).select("tx_id").unique(), on="tx_id")["attempt_id"].n_unique())
        att_hit = d.filter(pl.col("attempt_id").is_not_null())["attempt_id"].n_unique()
        counts[split] = {"alerts": d.height, "positives": int(d["label"].sum()),
                         "spans_split_flagged": int(d["spans_split"].sum()),  # sensitivity analysis only, never a filter
                         "kept_if_span_rule": int((~d["spans_split"]).sum()), "kept_if_span_rule_pos": int(d.filter(~pl.col("spans_split"))["label"].sum()),
                         "unique_tx": d["tx_id"].n_unique(), "attempts_in_split_views": att_all, "attempts_with_alert": att_hit}  # fmt: skip
    out["alert_sets"] = counts
    Path("data/runs").mkdir(parents=True, exist_ok=True)
    Path(f"data/runs/alerts_{a.data}.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("thresholds", "val_parts", "set_types_eval", "achieved_vs_target", "eval_conformal", "eval_fixed_window", "alert_sets")}, indent=1, default=str))
    for s in sweep:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})


if __name__ == "__main__":
    main()
