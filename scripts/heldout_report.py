"""E12: what changes when the cycle attempts are not in training (held-out cycles).

Owner: T1 (Abhishek).  Spec: tracks.md 3.9 week 8.  Compares the full run with the `<ds>_heldout` run (configs/data/<ds>_heldout.yaml):

  python scripts/heldout_report.py --data hi_small [--backtest-full DIR --backtest-heldout DIR]

1. Detection: GFP one-bank p0 on the val `eval` part, cycle laundering vs the other typologies (mean p0, share above the alert threshold,
   ROC AUC against the clean rows), full vs held-out.
2. Intake: val alerts and the attempt-level and transaction-level recall by typology, full vs held-out.
3. Back-test transfer (T2's `ftm.backtest.heldout.transfer_report`): the questions accepted by the held-out back-test (train alerts without cycles) and by
   the full back-test, scored on the same val alerts split into cycle-vs-clean and other-vs-clean: separation = mean log-LR of laundering answers minus clean.
Writes data/runs/heldout_<ds>.json. Train and val only; the test split is never read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.data import io  # noqa: E402


def detection(ds: str) -> dict:
    """Per typology (val `eval` rows pooled over bank views): rows, mean p0, share >= the alert threshold, AUC vs the clean rows."""
    from sklearn.metrics import roc_auc_score

    out = {}
    thr = json.loads(Path(f"data/runs/alerts_{ds}.json").read_text())["thresholds"]["t_alert"]
    thr_h = json.loads(Path(f"data/runs/alerts_{ds}_heldout.json").read_text())["thresholds"]["t_alert"]
    pq = io.load_config(ds)["parquet_dir"]
    pat = io.scan_patterns(pq, split="val").select("tx_id", "typology").collect()
    for tag, d, t in (("full", ds, thr), ("heldout", f"{ds}_heldout", thr_h)):
        p0 = pl.read_parquet(Path("data/scores") / d / "p0_one_bank.parquet").filter(  # time-guard: not transactions
            (pl.col("split") == "val") & (pl.col("val_part") == "eval") & pl.col("p0").is_not_null())
        p0 = p0.join(pat, on="tx_id", how="left").with_columns(pl.col("typology").fill_null("none"))
        clean = p0.filter(~pl.col("label"))
        res = {"threshold": t, "clean_rows": clean.height, "clean_alert_share": float((clean["p0"] >= t).mean())}
        for typ in sorted(p0.filter(pl.col("label"))["typology"].unique().to_list()):
            pos = p0.filter(pl.col("label") & (pl.col("typology") == typ))
            y = np.r_[np.ones(pos.height), np.zeros(clean.height)]
            s = np.r_[pos["p0"].to_numpy(), clean["p0"].to_numpy()]
            res[typ] = {"rows": pos.height, "mean_p0": float(pos["p0"].mean()), "share_alerted": float((pos["p0"] >= t).mean()),
                        "auc_vs_clean": float(roc_auc_score(y, s)) if pos.height else None}  # fmt: skip
        out[tag] = res
    return out


def intake(ds: str) -> dict:
    """Val alerts and recall by typology (attempt-level: attempts with at least one alerted transaction)."""
    pq = io.load_config(ds)["parquet_dir"]
    pat = io.scan_patterns(pq, split="val").select("tx_id", "attempt_id", "typology").collect()
    out = {}
    for tag, d in (("full", ds), ("heldout", f"{ds}_heldout")):
        al = pl.read_parquet(Path(io.load_config(d)["alerts_dir"]) / "val.parquet")  # time-guard: not transactions
        att = pat.group_by("typology").agg(pl.col("attempt_id").n_unique().alias("attempts"), pl.col("tx_id").n_unique().alias("laundering_tx"))
        hit = al.filter(pl.col("attempt_id").is_not_null()).group_by("typology").agg(
            pl.col("attempt_id").n_unique().alias("attempts_alerted"), pl.col("tx_id").n_unique().alias("tx_alerted"))  # fmt: skip
        t = att.join(hit, on="typology", how="left").fill_null(0).with_columns(
            (pl.col("attempts_alerted") / pl.col("attempts")).alias("attempt_recall"), (pl.col("tx_alerted") / pl.col("laundering_tx")).alias("tx_recall"))  # fmt: skip
        out[tag] = {"alerts": al.height, "positives": int(al["label"].sum()), "by_typology": t.sort("typology").to_dicts()}
    return out


def transfer(ds: str, dir_full: str, dir_held: str, fdr: float = 0.10) -> dict:
    from ftm.backtest.engine import BacktestCache, assemble
    from ftm.backtest.heldout import transfer_report
    from ftm.backtest.world import load_world

    val = load_world(f"{ds}_heldout", "val", heldout_cycles=True)
    cyc = [a for a in val.alerts if a.typology == "cycle"]
    rest = [a for a in val.alerts if a.typology != "cycle"]
    clean = [a for a in val.alerts if not a.label]
    subsets = {"cycle_vs_clean": cyc + clean, "other_vs_clean": rest}
    out = {"val_alerts": len(val.alerts), "cycle_laundering": sum(a.label for a in cyc), "other_laundering": sum(a.label for a in rest),
           "clean": len(clean)}  # fmt: skip
    for tag, d in (("full_table", dir_full), ("heldout_table", dir_held)):
        entries = assemble(BacktestCache(Path(d).parent, Path(d).name), fdr)
        rep = transfer_report(entries, val.banks, subsets)
        out[tag] = {"dir": d, "n_accepted": rep["n_accepted"], "subsets": rep["subsets"], "per_question": rep["per_question"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--backtest-full", default=None, help="assembled back-test directory of the full run (realistic, train alerts)")
    ap.add_argument("--backtest-heldout", default=None, help="assembled back-test directory of the `--heldout-cycles` run")
    a = ap.parse_args()
    out = {"dataset": a.data, "detection": detection(a.data), "intake": intake(a.data)}
    if a.backtest_full and a.backtest_heldout:
        out["transfer"] = transfer(a.data, a.backtest_full, a.backtest_heldout)
    Path(f"data/runs/heldout_{a.data}.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "transfer"}, indent=1, default=float))
    if "transfer" in out:
        t = out["transfer"]
        for tag in ("full_table", "heldout_table"):
            print(tag, t[tag]["n_accepted"], "accepted;", {k: (v["n_questions_scored"], v["mean_separation"]) for k, v in t[tag]["subsets"].items()})


if __name__ == "__main__":
    main()
