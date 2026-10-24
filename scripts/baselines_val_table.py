"""Week 9: every baseline on the validation `eval` part, one table per dataset (docs/t1_baselines_val.md).

Owner: T1 (Abhishek).  Rows are (transaction, bank view) of the val `eval` part. Each model's p0 comes from its own file; the models are compared
on the COMMON rows (every listed model has a p0 there) so that the paired CIs make sense; see-everything models are matched to the view rows by
tx_id. Brier, PR-AUC and F1 (threshold = best F1 on the model's own val `fit` rows of the common set); multi-seed models (DP-FedAvg, layer-wise
exchange) report the seed mean of every metric and the sd over seeds. CIs: cluster bootstrap by attempt + shared accounts with the hub cap 20
(the largest cluster of the plain definition is 16 % of the HI-Small and 58 % of the HI-Medium rows, above the 0.2 cut of decision 6), paired
for the Brier differences; 2,000 resamples for Brier, 500 for PR-AUC and F1.

  python scripts/baselines_val_table.py --data hi_small
Writes data/runs/baselines_val_<ds>.json. Val only, no test row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.baselines import gfp_gbt as g  # noqa: E402
from ftm.data import io  # noqa: E402
from ftm.eval import bootstrap as bs  # noqa: E402

KEY = ["tx_id", "bank"]
HUB_CAP = 20


def _val(path: Path, model: str | None = None) -> pl.DataFrame:
    d = pl.read_parquet(path).filter(pl.col("split") == "val")  # time-guard: not transactions
    return d.filter(pl.col("model") == model) if model else d


def registry(ds: str) -> dict[str, dict]:
    """name -> {"kind": "view" | "tx", "files": [parquet per seed], "model": filter}."""
    s = Path("data/scores") / ds
    reg: dict[str, dict] = {"GFP one-bank": {"kind": "view", "files": [s / "p0_one_bank.parquet"], "mode": "one_bank", "model": "gfp_lgbm"}}
    if (s / "p0_see_everything.parquet").exists():
        reg["GFP see-everything"] = {"kind": "tx", "files": [s / "p0_see_everything.parquet"], "mode": "see_everything", "model": "gfp_lgbm"}
    for mg in (s / "p0_multi_gnn.parquet", s / "p0_multi_gnn__one_bank.parquet"):  # HI-Medium only has the one-bank file
        if not mg.exists():
            continue
        d = pl.read_parquet(mg, columns=["mode"])["mode"].unique().to_list()
        if "one_bank" in d and "Multi-GNN one-bank" not in reg:
            reg["Multi-GNN one-bank"] = {"kind": "view", "files": [mg], "mode": "one_bank", "model": "multi_gnn"}
        if "see_everything" in d and "Multi-GNN see-everything" not in reg:
            reg["Multi-GNN see-everything"] = {"kind": "tx", "files": [mg], "mode": "see_everything", "model": "multi_gnn"}
    fd = s / "federated"
    for name in ("dpfedavg_client_eps16", "dpfedavg_client_eps8", "dpfedavg_client_epsinf", "dpfedavg_edge_eps8", "dpfedavg_edge_epsinf"):
        fs = sorted(fd.glob(f"{name}_seed*.parquet"))
        if fs:
            label = {"client": "client-level (unit: a bank)", "edge": "edge-level (unit: a transaction)"}[name.split("_")[1]]
            eps = name.split("eps")[1]
            reg[f"DP-FedAvg {label}, eps {'inf (no noise)' if eps == 'inf' else eps}"] = {"kind": "view", "files": fs, "mode": "one_bank", "model": None}
    cd = s / "ceydeli"
    for f in sorted({p.name.rsplit("_seed", 1)[0] for p in cd.glob("*_seed*.parquet")}) if cd.exists() else []:
        sync, ex = f.split("_", 1) if f.startswith(("fedavg", "fedprox", "local")) else ("sync_sgd", f[len("sync_sgd_"):])
        reg[f"Layer-wise exchange (Ceydeli), {sync}, exchange {ex}"] = {"kind": "view", "files": sorted(cd.glob(f"{f}_seed*.parquet")), "mode": "one_bank", "model": None}
    return reg


def build(ds: str, reg: dict) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """Frame of the val rows (fit and eval parts) with one p0 column per model and seed; returns it and {model: [columns]}."""
    base = None
    cols: dict[str, list[str]] = {}
    for name, r in reg.items():
        for i, f in enumerate(r["files"]):
            d = _val(f, r.get("model")).filter(pl.col("p0").is_not_null() & pl.col("p0").is_not_nan() & pl.col("val_part").is_in(["fit", "eval"]))
            d = d.filter(pl.col("mode") == r["mode"]) if "mode" in d.columns else d
            col = f"{name}|{i}"
            cols.setdefault(name, []).append(col)
            if r["kind"] == "view":
                d = d.select("tx_id", "bank", "val_part", "label", pl.col("p0").alias(col))
                base = d if base is None else base.join(d.drop("val_part", "label"), on=KEY, how="inner")
            else:
                d = d.select("tx_id", pl.col("p0").alias(col)).unique("tx_id")
                base = base.join(d, on="tx_id", how="inner") if base is not None else d
    return base, cols


def main() -> None:
    from sklearn.metrics import average_precision_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-boot-slow", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    reg = registry(a.data)
    # views first so that the base frame is a view frame
    reg = dict(sorted(reg.items(), key=lambda kv: kv[1]["kind"] != "view"))
    df, cols = build(a.data, reg)
    pq = io.load_config(a.data)["parquet_dir"]
    tx = io.scan_transactions(pq, split="val", columns=["tx_id", "src_id", "dst_id"]).collect()
    pat = io.scan_patterns(pq, split="val").select("tx_id", "attempt_id").collect()
    df = df.join(tx, on="tx_id", how="left").join(pat, on="tx_id", how="left").with_columns(pl.col("label").cast(pl.Int8).alias("y"))
    fit, ev = df.filter(pl.col("val_part") == "fit"), df.filter(pl.col("val_part") == "eval")
    y = ev["y"].to_numpy()
    cl = bs.cluster_ids(ev["attempt_id"].to_list(), [ev["src_id"].to_list(), ev["dst_id"].to_list()], hub_cap=HUB_CAP)
    const = float(fit["y"].mean())
    out = {"dataset": a.data, "common_rows_eval": ev.height, "positives": int(y.sum()), "views": sorted(set(ev["bank"].drop_nulls().to_list())),
           "cluster": bs.cluster_report(cl), "hub_cap": HUB_CAP, "models": {}}  # fmt: skip
    se_const = (const - y) ** 2
    out["models"]["Constant (fit base rate)"] = {"brier": [float(se_const.mean())], "seeds": 1, "pr_auc": [float(y.mean())], "f1": [None]}
    mean_se = {"Constant (fit base rate)": se_const}
    for name, cs in cols.items():
        b, pr, f1 = [], [], []
        for c in cs:
            thr = g.metrics(fit["y"].to_numpy(), fit[c].to_numpy())["thr"]
            m = g.metrics(y, ev[c].to_numpy(), thr)
            b.append(m["brier"]), pr.append(m["pr_auc"]), f1.append(m["f1"])
        mean_se[name] = np.mean([(ev[c].to_numpy() - y) ** 2 for c in cs], axis=0)
        out["models"][name] = {"seeds": len(cs), "brier": b, "pr_auc": pr, "f1": f1}
    ref = "GFP one-bank"
    for name, se in mean_se.items():
        r = out["models"][name]
        bm = bs.bootstrap_means(np.column_stack([se, se_const, mean_se[ref]]), cl, n_boot=a.n_boot, seed=a.seed)
        r["brier_mean"] = float(se.mean())
        r["brier_ci95"] = [float(np.quantile(bm[:, 0], 0.025)), float(np.quantile(bm[:, 0], 0.975))]
        r["skill_vs_constant"] = 1 - r["brier_mean"] / float(se_const.mean())
        d = bm[:, 0] - bm[:, 2]
        r["brier_minus_gfp_one_bank"] = {"diff": float(se.mean() - mean_se[ref].mean()), "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]}
        d = bm[:, 0] - bm[:, 1]
        r["brier_minus_constant"] = {"diff": float(se.mean() - se_const.mean()), "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]}
    # PR-AUC and F1 CIs (slow path) for the seed-0 predictions of every model
    for name, cs in cols.items():
        c = cs[0]
        thr = g.metrics(fit["y"].to_numpy(), fit[c].to_numpy())["thr"]
        p = ev[c].to_numpy()

        def stat(idx, p=p, thr=thr):
            yy = y[idx]
            if yy.sum() == 0:
                return np.nan
            return average_precision_score(yy, p[idx])

        def stat_f1(idx, p=p, thr=thr):
            yy, pr = y[idx], p[idx] >= thr
            tp = (pr & (yy == 1)).sum()
            return 2 * tp / max(pr.sum() + yy.sum(), 1)

        for key, fn in (("pr_auc_ci95", stat), ("f1_ci95", stat_f1)):
            s = bs.cluster_bootstrap(fn, cl, n_boot=a.n_boot_slow, seed=a.seed)
            out["models"][name][key] = [float(np.nanquantile(s, 0.025)), float(np.nanquantile(s, 0.975))]
    Path(f"data/runs/baselines_val_{a.data}.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"{a.data}: {ev.height} common eval rows, {int(y.sum())} positives, largest cluster share {out['cluster']['largest_share']:.4f}")
    for name, r in out["models"].items():
        sd = lambda v: float(np.std(v)) if len(v) > 1 else 0.0  # noqa: E731
        f1 = [x for x in r["f1"] if x is not None]
        print(f"{name:62s} Brier {np.mean(r['brier']):.6f} (sd {sd(r['brier']):.6f}) PR-AUC {np.mean(r['pr_auc']):.4f} F1 {np.mean(f1) if f1 else float('nan'):.3f} seeds {r['seeds']}")


if __name__ == "__main__":
    main()
