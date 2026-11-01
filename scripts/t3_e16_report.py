"""E16 (planner distillation) numbers on the 160 val `eval` alerts (T3, docs/t3_e16.md). CPU only, no server, validation only.

    python scripts/t3_e16_report.py [--n-boot 2000] [--out outputs/t3_val/analysis/e16.json]

Reuses the loaders and the paired bootstrap of `scripts/t3_val_analysis.py`: Brier, paired dBrier vs p0 and AUC with attempt+accounts cluster
bootstrap 95% CIs (seed 0), all over the same alerts. Arms: e16_prompted (Qwen3-8B-AWQ, no adapter), e16_student_s0..2, the Gemma 4 12B
teacher-like reference ours_compose, and p0 alone. Also: verdict and valid-DSL rates, questions per alert, the accepted share of the questions,
the mean shift |p - p0| and the share of alerts moved.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftm.planners import e18  # noqa: E402

_spec = importlib.util.spec_from_file_location("t3_val_analysis", ROOT / "scripts" / "t3_val_analysis.py")
va = importlib.util.module_from_spec(_spec)
sys.modules["t3_val_analysis"] = va
_spec.loader.exec_module(va)

QWEN = ["e16_prompted", "e16_student_s0", "e16_student_s1", "e16_student_s2"]
REF = "ours_compose"
NUM = ["brier", "diff", "auc", "verdict_rate", "valid_dsl", "q_per_alert", "accepted_share", "mean_shift", "moved", "median_s"]


def seed_mean_sd(vals) -> tuple[float, float]:
    """Mean and sample sd (ddof 1) over seeds; sd 0 for a single value."""
    v = np.asarray([x for x in vals if x is not None], "f8")
    return (float("nan"), float("nan")) if not len(v) else (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def run_stats(alerts, rows, mapping, n_boot: int, p=None) -> dict:
    """One arm: `alerts` from va.load_run, `rows` the per_alert.jsonl rows (valid-output counts, seconds). p = probabilities to score (default: the run's)."""
    p = va.arr(alerts, "prob") if p is None else np.asarray(p, "f8")
    y, p0 = va.arr(alerts, "label"), va.arr(alerts, "p0")
    d = va.diff(alerts, p, mapping, n_boot)
    a = va.auc_pair_ci(y, p, p0, va.clusters_of(alerts, mapping), n_boot)
    steps = [s for al in alerts for s in al["steps"] if s["gate"] == "allowed"]
    tot = {k: sum(r[k] for r in rows) for k in ("outputs", "json_ok", "questions", "dsl_ok")} if rows else None
    return {"n": len(alerts), "brier": d["brier"], "diff": d["diff"], "diff_ci": (d["lo"], d["hi"]), "auc": a["auc"], "auc_ci": a["auc_ci"],
            "auc_diff": a["diff"], "auc_diff_ci": a["diff_ci"], "auc_p0": a["auc_p0"],
            "verdict_rate": float(np.mean([al["prob_raw"] is not None for al in alerts])),
            "valid_dsl": e18.valid_dsl_output_rate(tot) if tot and tot["outputs"] else None,
            "q_per_alert": len(steps) / len(alerts), "accepted_share": sum(s["accepted"] for s in steps) / len(steps) if steps else None,
            "mean_shift": float(np.abs(p - p0).mean()), "moved": float((np.abs(p - p0) > 1e-9).mean()),
            "median_s": float(np.median([r["seconds"] for r in rows])) if rows else None}


def load_arm(name: str, table) -> tuple[list[dict], list[dict]]:
    d = va.EVAL / name
    rows = [r for c in sorted(d.glob("c[0-9][0-9][0-9]")) for r in e18.read_per_alert(c)]
    return va.load_run([d], table), rows


def build(n_boot: int) -> dict:
    table, mapping = va.load_table(), e18.load_clusters(va.CLUSTERS)
    arms = {n: load_arm(n, table) for n in [*QWEN, REF]}
    ids = {n: sorted(a["alert_id"] for a in al) for n, (al, _) in arms.items()}
    assert all(len(v) == 160 and v == ids[REF] for v in ids.values()), "arms do not cover the same 160 alerts"
    res = {n: run_stats(al, rows, mapping, n_boot) for n, (al, rows) in arms.items()}
    base = arms[REF][0]
    res["p0"] = run_stats(base, [], mapping, n_boot, p=va.arr(base, "p0")) | {"q_per_alert": None, "accepted_share": None}  # p0 asks nothing
    # paired dBrier of every Qwen arm vs the prompted arm (same alerts, aligned by id)
    pr = {a["alert_id"]: a["prob"] for a in arms["e16_prompted"][0]}
    for n in QWEN[1:]:
        al = arms[n][0]
        d = va.diff(al, va.arr(al, "prob"), mapping, n_boot, p_ref=[pr[a["alert_id"]] for a in al])
        res[n]["diff_vs_prompted"], res[n]["diff_vs_prompted_ci"] = d["diff"], (d["lo"], d["hi"])
    stu = [res[f"e16_student_s{s}"] for s in range(3)]
    res["student_mean_sd"] = {k: seed_mean_sd([r[k] for r in stu]) for k in NUM}
    # the seed-averaged student probability, scored as one arm (an ensemble, not a single student)
    pm = np.mean([[a["prob"] for a in sorted(arms[f"e16_student_s{s}"][0], key=lambda a: a["alert_id"])] for s in range(3)], axis=0)
    al = sorted(arms["e16_student_s0"][0], key=lambda a: a["alert_id"])
    res["student_seed_avg"] = run_stats(al, [], mapping, n_boot, p=pm)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=va.N_BOOT)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "t3_val" / "analysis" / "e16.json")
    a = ap.parse_args(argv)
    res = build(a.n_boot)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
