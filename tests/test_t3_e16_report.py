"""E16 report helpers of scripts/t3_e16_report.py on synthetic alerts (no run logs needed)."""

import importlib.util
import math
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("t3_e16_report", Path(__file__).resolve().parents[1] / "scripts" / "t3_e16_report.py")
er = importlib.util.module_from_spec(spec)
spec.loader.exec_module(er)


def alert(i, label, p0, prob, steps=()):
    return {"alert_id": f"V{i:06d}", "label": label, "p0": p0, "prob_raw": prob, "prob": p0 if prob is None else prob,
            "steps": [{"gate": "allowed", "accepted": a} for a in steps]}


def test_seed_mean_sd():
    m, sd = er.seed_mean_sd([1.0, 2.0, 3.0])
    assert m == 2.0 and abs(sd - 1.0) < 1e-12
    assert er.seed_mean_sd([0.5]) == (0.5, 0.0)
    assert math.isnan(er.seed_mean_sd([None])[0])


def test_run_stats():
    al = [alert(1, 1, 0.6, 0.8, [True, False]), alert(2, 0, 0.4, None), alert(3, 0, 0.3, 0.3, [True])]
    rows = [{"seconds": 2.0, "outputs": 2, "json_ok": 2, "questions": 2, "dsl_ok": 2}, {"seconds": 4.0, "outputs": 1, "json_ok": 0, "questions": 0, "dsl_ok": 0},
            {"seconds": 6.0, "outputs": 1, "json_ok": 1, "questions": 1, "dsl_ok": 1}]
    mapping = {a["alert_id"]: f"c{i}" for i, a in enumerate(al)}
    r = er.run_stats(al, rows, mapping, 20)
    assert r["n"] == 3 and abs(r["verdict_rate"] - 2 / 3) < 1e-12
    assert abs(r["q_per_alert"] - 1.0) < 1e-12 and abs(r["accepted_share"] - 2 / 3) < 1e-12
    assert abs(r["mean_shift"] - 0.2 / 3) < 1e-12 and abs(r["moved"] - 1 / 3) < 1e-12
    assert r["median_s"] == 4.0 and abs(r["valid_dsl"] - 3 / 4) < 1e-12
    y, p0 = np.array([1, 0, 0]), np.array([0.6, 0.4, 0.3])
    assert abs(r["brier"] - float(((np.array([0.8, 0.4, 0.3]) - y) ** 2).mean())) < 1e-12 and r["diff"] < 0
    assert p0.shape == (3,)
