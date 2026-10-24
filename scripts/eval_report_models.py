"""T4's gap check (`scripts/eval_report.py`) for the models and datasets it does not cover on its own.

Owner: T1 (Abhishek); the statistics are T4's (`ftm.eval`, `scripts/eval_report.py::report`), unchanged.

  python scripts/eval_report_models.py --data hi_small --model multi_gnn --on alerts   # Multi-GNN one-bank vs see-everything
  python scripts/eval_report_models.py --data hi_medium --on alerts                    # GFP one-bank only: metrics + cluster check

Multi-GNN: same rows and bootstrap as eval_report.py, with `data/scores/<ds>/p0_multi_gnn.parquet` in place of the GFP
p0 (`--on alerts` keeps the GFP-selected alerts, as the agent sees them, and swaps in Multi-GNN's score for them).
HI-Medium has no see-everything p0 yet: this prints one-bank Brier / PR-AUC / ECE (with a cluster-bootstrap CI against
the constant predictor) and the cluster check of decision 6 of `docs/t4_decisions_week2.md` (largest cluster share per
clustering and per val part). Writes data/runs/eval_<model>_<ds>_<on>.json. Val only; the test split is never read.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftm.data import io  # noqa: E402
from ftm.eval import bootstrap as bs  # noqa: E402
from ftm.eval import metrics as mt  # noqa: E402

spec = importlib.util.spec_from_file_location("eval_report", ROOT / "scripts" / "eval_report.py")
er = importlib.util.module_from_spec(spec)
spec.loader.exec_module(er)


def val_frame(ds: str, on: str, parts: tuple[str, ...], p0_file: str, model: str | None) -> pl.DataFrame:
    """One row per (tx, bank view) of the val `parts`: y, p_ob (+ p_se when the p0 file has see-everything), clusters' keys."""
    cfg = io.load_config(ds)
    pq = cfg["parquet_dir"]
    p0 = pl.read_parquet(ROOT / "data" / "scores" / ds / p0_file).filter(pl.col("split") == "val")  # time-guard: not transactions
    if model:
        p0 = p0.filter(pl.col("model") == model)
    p0 = p0.filter(pl.col("val_part").is_in(parts))
    ob = p0.filter(pl.col("mode") == "one_bank").select("tx_id", "bank", pl.col("p0").alias("p_ob"), "label", "val_part")
    if on == "alerts":  # the rows the agent sees: the GFP-selected alert set
        al = pl.read_parquet(ROOT / cfg["alerts_dir"] / "val.parquet").filter(pl.col("val_part").is_in(parts))  # time-guard: not transactions
        ob = al.select("tx_id", "bank").join(ob, on=["tx_id", "bank"], how="left")
    ob = ob.drop_nulls("p_ob")
    se = p0.filter(pl.col("mode") == "see_everything").select("tx_id", pl.col("p0").alias("p_se")).drop_nulls("p_se")  # a few seeds have no score
    f = ob.join(se, on="tx_id", how="inner") if se.height else ob.with_columns(pl.lit(None, dtype=pl.Float64).alias("p_se"))
    tx = io.scan_transactions(pq, split="val", columns=["tx_id", "src_id", "dst_id"]).collect()
    pat = io.scan_patterns(pq, split="val").select("tx_id", "attempt_id").collect()
    f = f.join(tx, on="tx_id", how="left").join(pat, on="tx_id", how="left")
    return f.with_columns(pl.col("label").cast(pl.Int8).alias("y")).sort("tx_id", "bank")


def clusters_of(f: pl.DataFrame, hub_cap: int | None) -> dict:
    att = f["attempt_id"].to_list()
    out = {"attempt+accounts": bs.cluster_ids(att, [f["src_id"].to_list(), f["dst_id"].to_list()]),
           "attempt+tx": bs.cluster_ids(att, [f["tx_id"].to_list()])}  # fmt: skip
    if hub_cap:
        out[f"attempt+accounts(hub_cap={hub_cap})"] = bs.cluster_ids(att, [f["src_id"].to_list(), f["dst_id"].to_list()], hub_cap=hub_cap)
    return out


def one_bank_only(f: pl.DataFrame, fit_rate: float, n_boot: int, seed: int, hub_cap: int | None) -> dict:
    y, p = f["y"].to_numpy(), f["p_ob"].to_numpy()
    cl = clusters_of(f, hub_cap)
    const = np.full(len(y), fit_rate)
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "n_boot": n_boot, "one_bank": mt.detection_summary(y, p),
           "constant": mt.detection_summary(y, const), "constant_rate": fit_rate, "brier_diff_ob_minus_constant": {}}  # fmt: skip
    for name, c in cl.items():
        d = bs.brier_diff(y, p, const, c, n_boot=n_boot, seed=seed)
        out["brier_diff_ob_minus_constant"][name] = {"diff": d["diff"], "ci95": d["ci"], "clusters": bs.cluster_report(c)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--model", default=None, choices=["gfp_lgbm", "multi_gnn"], help="default: gfp_lgbm (hi_medium) / multi_gnn here")
    ap.add_argument("--on", choices=["rows", "alerts"], default="alerts")
    ap.add_argument("--n-boot", type=int, default=bs.N_BOOT)
    ap.add_argument("--seed", type=int, default=bs.SEED)
    ap.add_argument("--hub-cap", type=int, default=None)
    a = ap.parse_args()
    model = a.model or ("multi_gnn" if a.data == "hi_small" else "gfp_lgbm")
    p0_file = "p0_multi_gnn.parquet" if model == "multi_gnn" else "p0.parquet"
    if model == "gfp_lgbm" and not (ROOT / "data/scores" / a.data / "p0_see_everything.parquet").exists():
        p0_file = "p0_one_bank.parquet"
    out = {"data": a.data, "model": model, "on": a.on}
    ev = val_frame(a.data, a.on, ("eval",), p0_file, model)
    if ev["p_se"].null_count() == 0:
        out["eval"] = er.report(ev, a.n_boot, a.seed, a.hub_cap)
    else:
        fit = val_frame(a.data, "rows", ("fit",), p0_file, model)
        out["eval"] = one_bank_only(ev, float(fit["y"].mean()), a.n_boot, a.seed, a.hub_cap)
        # cluster check (decision 6): all val alerts and the eval part, alerts and rows
        out["cluster_check"] = {}
        for label, parts, on in (("alerts_eval", ("eval",), "alerts"), ("alerts_all_val", ("fit", "cal", "eval"), "alerts"),
                                 ("rows_eval", ("eval",), "rows")):  # fmt: skip
            f = val_frame(a.data, on, parts, p0_file, model)
            out["cluster_check"][label] = {k: bs.cluster_report(v) for k, v in clusters_of(f, a.hub_cap or 20).items()}
    path = ROOT / "data" / "runs" / f"eval_{model}_{a.data}_{a.on}.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
