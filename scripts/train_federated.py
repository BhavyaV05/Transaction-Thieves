"""DP-FedAvg over the bank views: Brier-vs-epsilon curves for the two privacy units (ftm/baselines/federated.py).

Owner: T1 (Abhishek).  Run from the repo root:
  python scripts/train_federated.py --data hi_small --unit edge          # DP-SGD per client, unit = one transaction, "tx" features
  python scripts/train_federated.py --data hi_small --unit client        # clipped + noised client updates, unit = one bank's dataset, "ctx" features
Train split only for training; val `eval` for the numbers (calibrator on val `fit`). Never reads test (ftm.data.io). Writes
data/runs/federated_<ds>_<unit>.json with every setting, seed and the commit. CPU is enough (tiny MLP).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from ftm.baselines import federated as fed  # noqa: E402

torch.set_num_threads(2)  # leave the CPU to a concurrent GPU job

FEATURES = {"edge": "tx", "client": "ctx"}  # the edge-level guarantee needs per-example gradients that depend on one transaction only
DEFAULTS = {"edge": dict(rounds=10, local_steps=50, lr=0.5, clip=1.0), "client": dict(rounds=10, local_steps=600, lr=0.5, clip=1.0)}


def log(msg) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--unit", choices=["edge", "client"], required=True)
    ap.add_argument("--eps", type=float, nargs="*", default=[1, 2, 4, 8], help="privacy budgets; the non-private run (inf) is always added")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--local-steps", type=int, default=None)
    ap.add_argument("--q", type=float, default=0.02, help="edge arm: Poisson sampling rate per client (same for all clients)")
    ap.add_argument("--banks", type=int, nargs="*", default=None)
    a = ap.parse_args()
    hp = DEFAULTS[a.unit] | {k: v for k, v in dict(rounds=a.rounds, local_steps=a.local_steps).items() if v}
    cm = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    clients, info = fed.load_clients(a.data, FEATURES[a.unit], a.banks, log=log)
    log(f"{len(clients)} clients, {info['n_features']} features ({FEATURES[a.unit]}), skipped {info['skipped']}")
    rows = []
    for eps in [None, *a.eps]:
        res = []
        for seed in a.seeds:
            r = fed.run_federated(clients, a.unit, eps, seed=seed, q=a.q, log=log, **hp)
            m = fed.evaluate(clients, r.pop("raw"))
            res.append(m | {"sigma": r["sigma"], "epsilon": r["epsilon"], "seed": seed})
            log(f"{a.unit} eps={eps} seed={seed}: sigma={r['sigma']:.3f} brier={m['brier']:.6f} skill={m['brier_skill']:.3f} pr_auc={m['pr_auc']:.4f}")
        mean, sd = (lambda k, f=f, res=res: float(f([x[k] for x in res])) for f in (np.mean, np.std))
        rows.append({"epsilon": float("inf") if eps is None else eps, "sigma": res[0]["sigma"], "brier": mean("brier"), "brier_sd": sd("brier"),
                     "brier_skill": mean("brier_skill"), "pr_auc": mean("pr_auc"), "f1": mean("f1"), "brier_const": mean("brier_const"),
                     "runs": res})  # fmt: skip
    out = {"dataset": a.data, "unit": a.unit, "features": FEATURES[a.unit], "commit": cm, "delta": fed.DELTA, "q": a.q if a.unit == "edge" else 1.0,
           "hyper": hp, "clients": [c["bank"] for c in clients], "skipped": info["skipped"], "val_cut": info["cut"],
           "n_eval_rows": rows[0]["runs"][0]["n"], "n_eval_pos": rows[0]["runs"][0]["n_pos"], "rows": rows}  # fmt: skip
    Path("data/runs").mkdir(exist_ok=True)
    Path(f"data/runs/federated_{a.data}_{a.unit}.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"{'epsilon':>8} {'sigma':>7} {'Brier':>10} {'+-sd':>9} {'skill':>7} {'PR-AUC':>7} {'F1':>6}")
    for r in rows:
        print(f"{r['epsilon']:>8} {r['sigma']:>7.2f} {r['brier']:>10.6f} {r['brier_sd']:>9.6f} {r['brier_skill']:>7.3f} {r['pr_auc']:>7.4f} {r['f1']:>6.3f}")
    print(f"constant-predictor Brier {rows[0]['brier_const']:.6f}")


if __name__ == "__main__":
    main()
