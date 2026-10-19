"""Conformal intake -> train/val/test alert sets and the frozen 1,000-alert subset.

Owner: T1 (Abhishek).  Run from the repo root, e.g. python scripts/make_alerts.py --data hi_small

Reads data/scores/<ds>/p0_one_bank.parquet (scripts/train_gfp.py), writes data/alerts/<ds>/{train,val}.parquet
and data/runs/alerts_<ds>.json (miss rates vs targets, sweep, ablation, test-count projection).
The test alert set and the 1,000-alert subset are NOT built: they need the test split (G3/G5 decision).
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
    p, y, d = cal["p0"].to_numpy(), cal["label"].cast(pl.Int8).to_numpy(), cal["day"].to_numpy()
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--config", default="configs/intake/conformal.yaml")
    ap.add_argument("--alpha-launder", type=float, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    a_l = a.alpha_launder if a.alpha_launder is not None else cfg["alpha_launder"]
    a_c = cfg["alpha_clean"]
    dcfg = io.load_config(a.data)
    pq = dcfg["parquet_dir"]
    p0 = pl.read_parquet(Path("data/scores") / a.data / "p0_one_bank.parquet").filter(pl.col("p0").is_not_null() & pl.col("p0").is_not_nan())
    tx = io.scan_transactions(pq, split=["train", "val"], columns=["tx_id", "timestamp", "day"]).collect()
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
    for al_ in (0.05, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99):
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
    starts = {s: io.scan_transactions(pq, split=s, columns=["timestamp"]).select(pl.col("timestamp").min()).collect()["timestamp"][0]
              for s in ("train", "val")}  # fmt: skip
    adir = Path(dcfg["alerts_dir"])
    adir.mkdir(parents=True, exist_ok=True)
    cols = ["alert_id", "tx_id", "bank", "timestamp", "p0", "label", "attempt_id", "typology", "split", "val_part",
            "drop_spans_split", "drop_window", "keep"]  # fmt: skip
    counts = {}
    for split in ("train", "val"):
        d = p0.filter((pl.col("split") == split) & (pl.col("p0") >= th["t_alert"])).join(pat, on="tx_id", how="left")
        d = d.sort(["timestamp", "tx_id", "bank"]).with_columns(
            (pl.lit(split[0].upper()) + pl.int_range(pl.len()).cast(pl.Utf8).str.zfill(6)).alias("alert_id"))  # fmt: skip
        d = al.add_drop_flags(d, att, {split: starts[split]}, window_hours=cfg["evidence_window_hours"])
        d.select(cols).write_parquet(adir / f"{split}.parquet")
        counts[split] = {"alerts": d.height, "positives": int(d["label"].sum()),
                         "keep_spans_rule_only": int((~d["drop_spans_split"]).sum()),
                         "keep_spans_rule_pos": int(d.filter(~pl.col("drop_spans_split"))["label"].sum()),
                         "keep_both_rules": int(d["keep"].sum()), "keep_both_rules_pos": int(d.filter(pl.col("keep"))["label"].sum()),
                         "unique_tx": d["tx_id"].n_unique()}  # fmt: skip
    out["alert_sets"] = counts
    Path("data/runs").mkdir(parents=True, exist_ok=True)
    Path(f"data/runs/alerts_{a.data}.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("thresholds", "achieved_vs_target", "eval_conformal", "eval_fixed_window", "alert_sets")}, indent=1, default=str))
    for s in sweep:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})


if __name__ == "__main__":
    main()
