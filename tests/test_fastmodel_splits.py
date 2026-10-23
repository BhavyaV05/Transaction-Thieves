"""T3: val fit/cal/eval roles in the fast-model cascade (T4 review of PR #10)."""

import numpy as np
import pytest

from ftm.fastmodel.decisions import DecisionExample
from ftm.fastmodel.splits import calibrate_cascade


class SplitModels:
    """Fast model: overconfident logits (temperature ~3 too sharp); 8B: sharp and right."""

    def __init__(self, rng):
        self.rng, self.large_calls = rng, []

    def raw_logps(self, ex):
        s = 3.0 * self.rng.normal(0.6, 1.0) * (1.0 if ex.label == "yes" else -1.0)
        z = np.array([s, -s]) if self.rng.random() < 0.8 else np.array([-s, s])  # right 80% of the time
        return z - np.logaddexp(*z)

    def predict_proba(self, ex):
        self.large_calls.append(ex)
        return {"yes": 0.97, "no": 0.03} if ex.label == "yes" else {"yes": 0.03, "no": 0.97}


def test_calibrate_cascade_uses_each_val_part_for_its_job_and_flags_thin_types():
    m = SplitModels(np.random.default_rng(3))
    ex, parts = [], []
    for part in ("fit", "cal", "eval"):
        for i in range(300):
            ex.append(DecisionExample("stop", "c", "yes" if i % 2 else "no", f"{part}{i}"))
            parts.append(part)
    for i in range(5):  # a thin type: 5 cal rows at alpha 0.1 cannot reach a quantile
        ex.append(DecisionExample("rule_applies", "c", "yes", f"r{i}"))
        parts.append("cal")
    temps, _, rep = calibrate_cascade(ex, parts, m, m, alpha=0.1)
    d, thin = rep["decisions"]["stop"], rep["decisions"]["rule_applies"]
    assert (d["n_fit"], d["n_cal"], d["n_eval"]) == (300, 300, 300) and d["n"] == 300 and not d["fast_full_set"]
    assert temps["stop"] > 1.5 and d["top_label_ece_eval"] < d["top_label_ece_eval_T1"]  # T fitted on `fit`
    assert thin["fast_full_set"] and thin["large_full_set"] and thin["always_defer"] and thin["no_fit_rows"]
    assert thin["n_cal"] == 5 and thin["n_eval"] == 0 and "n" not in thin
    assert rep["flagged"] == ["rule_applies"]
    # the 8B tier saw the cal rows (for its conformal sets), and only cal rows or eval rows it was asked about
    assert {e.alert_id[:3] for e in m.large_calls[:305]} <= {"cal", "r0", "r1", "r2", "r3", "r4"}
    with pytest.raises(ValueError, match="never enter"):
        calibrate_cascade(ex[:2], ["train", "fit"], m, m)
    with pytest.raises(ValueError):
        calibrate_cascade(ex[:2], ["fit"], m, m)
