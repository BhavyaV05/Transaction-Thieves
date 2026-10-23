"""T4: run-log schema v1 and per-system scoring from run logs."""

import numpy as np
import polars as pl
import pytest

from ftm.eval import runlog, runs
from ftm.fusion import fuse

PY = {"str": pl.Utf8, "int": pl.Int64, "float": pl.Float64}


def _log(rows):
    return pl.DataFrame(rows, schema={c: PY[t] for c, t in runlog.RUN_LOG_COLUMNS.items()}, strict=False)


def test_v0_columns_unchanged_and_types_simple():
    assert runlog.V0_COLUMNS[:5] == ("run_id", "commit", "machine", "system", "variant")
    assert "gpu_seconds" in runlog.V0_COLUMNS and "log_version" not in runlog.V0_COLUMNS
    assert set(runlog.RUN_LOG_COLUMNS.values()) <= {"str", "int", "float"}


def test_v0_rows_still_load_with_v1_schema():
    df = _log([{"run_id": "r", "system": "react", "split": "val", "alert_id": "V1", "event": "verdict"}])
    assert df["log_version"].to_list() == [None]
    assert runlog.validate(df.columns, df["event"].to_list()) == []
    assert runlog.validate(["run_id"], ["oops"]) != []


def _synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    p0 = np.clip(0.3 + 0.2 * (y - 0.3) + 0.1 * rng.normal(size=n), 0.02, 0.98)
    rows, labels = [], {}
    for i in range(n):
        a = f"V{i:04d}"
        labels[a] = int(y[i])
        llr = 1.2 if y[i] else -1.2
        rows.append({"run_id": "r", "system": "ours", "variant": "compose", "seed": 0, "split": "val", "alert_id": a,
                     "event": "step", "step": 1, "family": "pass_through", "log_lr": llr + rng.normal(0, 0.5)})
        rows.append({"run_id": "r", "system": "ours", "variant": "compose", "seed": 0, "split": "val", "alert_id": a,
                     "event": "verdict", "p0": p0[i], "probability": None if i % 10 == 0 else float(y[i]) * 0.8 + 0.1,
                     "llm_calls": 4, "eps_spent": 1.0, "stop_reason": "cannot_cross"})
    return _log(rows), labels, p0, y


def test_score_end_to_end_questions_only_and_no_verdict():
    log, labels, p0, y = _synthetic()
    [r] = runs.score(log, labels, n_boot=300)
    assert r["n"] == 400 and r["no_verdict_rate"] == pytest.approx(0.1)
    assert r["brier_end_to_end"] < r["brier_p0"]
    assert r["brier_questions_only"] < r["brier_p0"]
    assert r["brier_e2e_minus_p0"]["ci95"][1] < 0
    assert r["mean_llm_calls"] == 4 and r["stop_reasons"] == {"cannot_cross": 400}


def test_gap_closed_against_upper_limit():
    log, labels, p0, y = _synthetic()
    upper = {"gfp": {a: (0.9 if labels[a] else 0.05) for a in labels}}
    [r] = runs.score(log, labels, upper=upper, n_boot=300)
    g = r["gap_closed_questions_only_vs_gfp"]
    assert 0 < g["point"] < 1.5 and g["ci95"][0] <= g["point"] <= g["ci95"][1]


def test_unused_answers_do_not_move_questions_only():
    log, labels, p0, _ = _synthetic()
    log = log.with_columns(pl.lit(None, dtype=pl.Float64).alias("log_lr"))  # no accepted back-test LR
    [r] = runs.score(log, labels, n_boot=100)
    assert r["brier_questions_only"] == pytest.approx(r["brier_p0"])


def test_unlabelled_alerts_dropped_and_duplicates_refused():
    log, labels, _, _ = _synthetic(n=50)
    half = dict(list(labels.items())[:25])
    [r] = runs.score(log, half, n_boot=50)
    assert r["n"] == 25 and r["dropped_unlabelled"] == 25
    dup = pl.concat([log, log.filter(pl.col("event") == "verdict").head(1)])
    with pytest.raises(ValueError):
        runs.score(dup, labels, n_boot=50)


def test_fitted_fusion_is_used_for_questions_only():
    log, labels, p0, y = _synthetic()
    X, _ = runs.answer_features(log, list(labels), families=["pass_through"])
    f = fuse.Fusion("per_family", families=["pass_through"]).fit(p0, X, y)
    [naive] = runs.score(log, labels, n_boot=50)
    [fitted] = runs.score(log, labels, fusion=f, n_boot=50)
    assert fitted["brier_questions_only"] <= naive["brier_questions_only"] + 1e-9
