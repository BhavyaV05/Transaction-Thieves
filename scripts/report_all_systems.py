"""Per-system ROC-AUC, PR-AUC, ECE and precision/recall at 10% on the final g5 test run.

Descriptive and read-only; it feeds Table 5 (appendix) of docs/report/final/report.tex and never changes a verdict.
Run after the final analysis:

    FTM_FINAL_RUN=1 python scripts/report_all_systems.py docs/results/test_analysis/all_systems_metrics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ftm import guards  # noqa: E402
from ftm.eval import final as fn  # noqa: E402
from ftm.eval import metrics as mt  # noqa: E402

RUN = "outputs/final/g5"
INPUTS = "configs/analysis/inputs_test.yaml"
FIXED = "outputs/analysis/val/fixed_on_val.json"


def summary(name: str, alert_set: str, y: np.ndarray, p: np.ndarray, m: np.ndarray) -> dict:
    yy, pp = y[m], p[m]
    k = max(1, int(round(0.1 * m.sum())))
    top = np.argsort(-pp, kind="stable")[:k]
    return {"system": name, "alerts": alert_set, "n": int(m.sum()), "pos": int(yy.sum()),
            "brier": mt.brier(yy, pp), "auc": float(roc_auc_score(yy, pp)), "prauc": mt.pr_auc(yy, pp),
            "ece": mt.ece(yy, pp, 10), "prec10": float(yy[top].mean()), "rec10": float(yy[top].sum() / yy.sum())}


def main(out_path: str) -> int:
    guards.require_final_run("descriptive per-system metrics on the test split")
    inputs, systems = fn.load_configs(INPUTS, None)
    run = fn.load_run("test", RUN, None, systems.get("never_results"))
    panel = fn.build_panel(run, inputs, systems, FIXED, 200, 0)
    y = panel.y.astype(float)
    out = []
    for key, s in panel.scores.items():
        m = s.present()
        r = summary(key.label(), key.alert_set, y, s.p_e2e, m)
        p0 = summary("p0", key.alert_set, y, panel.p0, m)
        r.update({f"p0_{k}": p0[k] for k in ("brier", "auc", "prauc", "ece", "prec10", "rec10")})
        out.append(r)
    for name, s in panel.sources.items():
        for alert_set in sorted({k.alert_set for k in panel.scores}):
            ref = next(v for k, v in panel.scores.items() if k.alert_set == alert_set)
            out.append(summary(name, alert_set, y, s.p_e2e, ref.present() & np.isfinite(s.p_e2e)))
    Path(out_path).write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
