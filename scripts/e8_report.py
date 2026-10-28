"""E8 / H6 Brier half: the same planner on the same val `eval` alerts with four evidence views (docs/t1_planner_view.md).

Owner: T3 (Akshat). Arms are run dirs of scripts/run_llm_agent.py that differ only in `--evidence-view`:

    python scripts/e8_report.py --runs outputs/t3_e8 --clusters outputs/e18/clusters.json

H6 statistic: Brier(scorer_pcst) - Brier(amount), paired per alert, cluster bootstrap by attempt+accounts (no hub cap on
alerts, as in E18). The other differences are context. End-to-end probability: no verdict = p0, as everywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftm.eval import bootstrap as bs  # noqa: E402
from ftm.eval import metrics as mt  # noqa: E402
from ftm.planners import e18  # noqa: E402

VIEWS = ("amount", "amount_1hop", "scorer_pcst", "long_prompt")
H6 = ("scorer_pcst", "amount")
CONTEXT = (("amount_1hop", "amount"), ("long_prompt", "amount"), ("scorer_pcst", "amount_1hop"))
PARITY = ("model", "mode", "seed", "alerts", "val_part", "data", "fusion", "limits", "alerts_hash")


def load_arm(run_dir: Path, view: str) -> dict:
    s = json.loads((run_dir / "summary.json").read_text())
    if s.get("evidence_view") != view:
        raise SystemExit(f"{run_dir}: summary says evidence_view {s.get('evidence_view')!r}, expected {view!r}")
    rows = e18.read_per_alert(run_dir)
    p, nvr = mt.end_to_end_probability([r["probability"] for r in rows], [r["p0"] for r in rows])
    tot = {k: sum(r[k] for r in rows) for k in ("outputs", "json_ok", "questions", "dsl_ok")}
    return {"summary": s, "ids": tuple(r["alert_id"] for r in rows), "y": np.array([r["label"] for r in rows], "f8"),
            "p": np.asarray(p, "f8"), "p0": np.array([r["p0"] for r in rows], "f8"), "no_verdict_rate": nvr,
            "valid_rate": e18.valid_dsl_output_rate(tot), "seconds": float(np.median([r["seconds"] for r in rows])),
            "questions_per_alert": tot["questions"] / len(rows)}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT / "outputs" / "t3_e8", help="<runs>/<view>/c000 per arm")
    ap.add_argument("--clusters", type=Path, default=ROOT / "outputs" / "e18" / "clusters.json")
    ap.add_argument("--n-boot", type=int, default=bs.N_BOOT)
    ap.add_argument("--out", type=Path, default=None, help="default: <runs>/e8_report.json")
    a = ap.parse_args(argv)

    arms = {v: load_arm(a.runs / v / "c000", v) for v in VIEWS}
    ref = arms[VIEWS[0]]
    for v, arm in arms.items():
        if arm["ids"] != ref["ids"]:
            raise SystemExit(f"{v} is not on the same alerts (in the same order) as {VIEWS[0]}")
        diff = {k: (arm["summary"].get(k), ref["summary"].get(k)) for k in PARITY if arm["summary"].get(k) != ref["summary"].get(k)}
        if diff or arm["summary"]["backtest"]["path"] != ref["summary"]["backtest"]["path"]:
            raise SystemExit(f"{v} differs from {VIEWS[0]} in more than the evidence view: {diff}")
    clusters, how = e18.clusters_for(ref["ids"], e18.load_clusters(a.clusters))

    def diff(x: str, z: str) -> dict:
        d = bs.brier_diff(ref["y"], arms[x]["p"], arms[z]["p"], clusters, n_boot=a.n_boot)
        return {"arm": x, "vs": z, "diff": d["diff"], "ci": list(d["ci"]),
                "alerts_with_different_p": int((np.abs(arms[x]["p"] - arms[z]["p"]) > 1e-12).sum())}

    def brier_ci(p) -> list[float]:
        se = (p - ref["y"]) ** 2
        return list(bs.percentile_ci(bs.bootstrap_means(se, clusters, a.n_boot)[:, 0]))

    rep = {"n_alerts": len(ref["ids"]), "positives": int(ref["y"].sum()), "clusters": how, **bs.cluster_report(clusters),
           "model": ref["summary"]["model"], "fusion": ref["summary"]["fusion"], "backtest": ref["summary"]["backtest"]["path"],
           "p0_brier": mt.brier(ref["y"], ref["p0"]),
           "arms": {v: {"brier": mt.brier(arm["y"], arm["p"]), "ci": brier_ci(arm["p"]), "valid_rate": arm["valid_rate"],
                        "no_verdict_rate": arm["no_verdict_rate"], "median_s": arm["seconds"],
                        "questions_per_alert": arm["questions_per_alert"], "run_id": arm["summary"]["run_id"]}
                    for v, arm in arms.items()},
           "h6": {**diff(*H6), "rejects_null": None},
           "context": [diff(x, z) for x, z in CONTEXT]}  # fmt: skip
    rep["h6"]["rejects_null"] = rep["h6"]["ci"][1] < 0  # Brier half only; the share half fails (docs/t1_planner_view.md)
    out = a.out or a.runs / "e8_report.json"
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
