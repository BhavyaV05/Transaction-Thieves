"""Week 11 (C3, H10): Brier versus epsilon for DP-FedAvg next to the per-question epsilon curve, one figure per dataset.

Owner: T1 (Abhishek).  Val only (no test row).  Run from the repo root:
  python scripts/federated_eps_figure.py --data hi_small     # -> docs/figures/t1_federated_eps_hi_small.png, data/runs/federated_eps_hi_small.json

All curves are scored on the same rows: the val alerts that have a modelled partner bank (324 on HI-Small, 460 on HI-Medium; the set of
T2's eps sweep). DP-FedAvg: the calibrated per-row p0 of `train_federated.py --save-p0` (mean and sd over 3 seeds; the unit of epsilon is
a bank's dataset for client-level and a transaction for edge-level). Questions: `data/runs/eps_sweep_<ds>.json` (T2, `scripts/eps_sweep.py`,
EIG planner, naive fusion), x = the per-alert budget. Reference lines: the constant predictor (val `fit` base rate), GFP one-bank and, where
it exists, GFP see-everything. The layer-wise exchange (Sync-SGD, forward + backward exchange) is marked at the right edge as non-private.
The x axes are NOT comparable (three different units); only the y axis is shared.
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

INF_X = 32.0  # where "no noise" is drawn
UNITS = {
    "client": "DP-FedAvg, client-level: ε per bank dataset (training)",
    "edge": "DP-FedAvg, edge-level: ε per transaction (training)",
    "question": "Question answers (EIG planner): ε per question answer (per alert)",
}
COLORS = {"client": "#2a78d6", "edge": "#eb6834", "question": "#1baf7a"}  # categorical slots 1 to 3 of the validated reference palette


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def alert_rows(ds: str) -> pl.DataFrame:
    cfg = io.load_config(ds)
    a = pl.read_parquet(Path(cfg["alerts_dir"]) / "val.parquet")  # time-guard: not transactions
    banks = io.read_banks(cfg["views_dir"])["bank"].to_list()
    return a.filter(pl.col("counterparty_bank").is_in(banks)).select("tx_id", "bank", "p0", pl.col("label").cast(pl.Float64).alias("y"), "val_part")


def on_alerts(al: pl.DataFrame, f: Path, by: list[str]) -> np.ndarray:
    d = pl.read_parquet(f).filter(pl.col("split") == "val")  # time-guard: not transactions
    j = al.join(d.select(*by, pl.col("p0").alias("q")).unique(by), on=by, how="left")
    assert j["q"].null_count() == 0, f"{f.name}: {j['q'].null_count()} alert rows without a score"
    return j["q"].to_numpy()


def collect(ds: str) -> dict:
    al = alert_rows(ds)
    y = al["y"].to_numpy()
    s = Path("data/scores") / ds
    fit_rate = float(pl.read_parquet(s / "p0_one_bank.parquet").filter((pl.col("val_part") == "fit") & pl.col("p0").is_not_null())["label"].cast(pl.Float64).mean())
    out: dict = {"dataset": ds, "alerts": al.height, "positives": int(y.sum()), "const_rate": fit_rate,
                 "ref": {"constant": brier(y, np.full_like(y, fit_rate)), "gfp_one_bank": brier(y, al["p0"].to_numpy())}, "curves": {}}  # fmt: skip
    if (s / "p0_see_everything.parquet").exists():
        out["ref"]["gfp_see_everything"] = brier(y, on_alerts(al, s / "p0_see_everything.parquet", ["tx_id"]))
    for unit in ("client", "edge"):
        pts = []
        for f in sorted((s / "federated").glob(f"dpfedavg_{unit}_eps*_seed0.parquet")):
            tag = f.name.split("_eps")[1].split("_seed")[0]
            b = [brier(y, on_alerts(al, f.with_name(f.name.replace("seed0", f"seed{k}")), ["tx_id", "bank"])) for k in (0, 1, 2)]
            pts.append({"eps": float("inf") if tag == "inf" else float(tag), "brier": float(np.mean(b)), "sd": float(np.std(b))})
        out["curves"][unit] = sorted(pts, key=lambda r: r["eps"])
    sw = json.loads(Path(f"data/runs/eps_sweep_{ds}.json").read_text())
    out["curves"]["question"] = [{"eps": float(r["budget"]), "brier": float(r["brier_final"]), "brier_p0": float(r["brier_p0"]), "mean_questions": float(r["mean_questions"]),
                                  "mean_eps_spent": float(r["mean_eps"])} for r in sw]  # fmt: skip
    out["ref"]["questions_p0_check"] = float(sw[0]["brier_p0"])  # must equal gfp_one_bank on the same alerts
    lw = sorted((s / "ceydeli").glob("sync_sgd_bwd_seed*.parquet"))
    if lw:
        b = [brier(y, on_alerts(al, f, ["tx_id", "bank"])) for f in lw]
        out["layerwise_sync_sgd_bwd"] = {"brier": float(np.mean(b)), "sd": float(np.std(b))}
    return out


def draw(out: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=100)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    ref = out["ref"]
    lines = [("constant predictor", ref["constant"], (0, (1, 2))), ("GFP one-bank p0", ref["gfp_one_bank"], "-")]
    if "gfp_see_everything" in ref:
        lines.append(("GFP see-everything", ref["gfp_see_everything"], (0, (6, 3))))
    for name, v, ls in lines:
        ax.axhline(v, color=ink2, lw=1.0, ls=ls, zorder=1)
        ax.text(0.47, v, f"{name} {v:.4f}", fontsize=7.5, color=ink2, va="bottom", ha="left")
    for key, pts in out["curves"].items():
        x = np.array([INF_X if not np.isfinite(p["eps"]) else p["eps"] for p in pts])
        y = np.array([p["brier"] for p in pts])
        ax.plot(x, y, color=COLORS[key], lw=2, marker="o", ms=6, mec="#fcfcfb", mew=1.5, label=UNITS[key], zorder=3)
        if key in ("client", "edge"):
            sd = np.array([p["sd"] for p in pts])
            ax.fill_between(x, y - sd, y + sd, color=COLORS[key], alpha=0.15, lw=0)
    if "layerwise_sync_sgd_bwd" in out:
        v = out["layerwise_sync_sgd_bwd"]["brier"]
        ax.plot([INF_X * 1.5], [v], marker="D", ms=7, color=ink, mec="#fcfcfb", label="layer-wise exchange (Sync-SGD, fwd+bwd): non-private, no ε", zorder=4)
    ax.set_xscale("log")
    ticks = [0.5, 1, 2, 4, 8, 16, INF_X]
    ax.set_xticks(ticks + ([INF_X * 1.5] if "layerwise_sync_sgd_bwd" in out else []))
    ax.set_xticklabels(["0.5", "1", "2", "4", "8", "16", "no noise"] + (["no DP"] if "layerwise_sync_sgd_bwd" in out else []), fontsize=8, color=ink2)
    ax.minorticks_off()
    ax.set_xlim(0.45, INF_X * 2)
    ax.set_xlabel("ε  (three different units, see legend: the x axes are not comparable)", fontsize=8.5, color=ink2)
    ax.set_ylabel(f"Brier on {out['alerts']} val alerts ({out['positives']} positive)", fontsize=8.5, color=ink2)
    ax.tick_params(axis="y", labelsize=8, colors=ink2)
    ax.grid(axis="y", color=grid, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(grid)
    ax.set_title(f"{out['dataset']}: Brier versus ε, federated training and question answers", fontsize=10, color=ink, loc="left")
    ax.legend(fontsize=7.2, frameon=False, loc="lower left", bbox_to_anchor=(0.0, -0.46), ncol=1, labelcolor=ink2)
    if "gfp_see_everything" not in ref:
        ax.text(0.99, 0.02, "GFP see-everything and layer-wise exchange: not run on this dataset (BLOCKED-COMPUTE)", transform=ax.transAxes, fontsize=6.8, color=ink2, ha="right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hi_small")
    a = ap.parse_args()
    out = collect(a.data)
    assert abs(out["ref"]["gfp_one_bank"] - out["ref"]["questions_p0_check"]) < 1e-9, "the alert set differs from the eps sweep's"
    Path(f"data/runs/federated_eps_{a.data}.json").write_text(json.dumps(out, indent=1))
    draw(out, Path(f"docs/figures/t1_federated_eps_{a.data}.png"))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
