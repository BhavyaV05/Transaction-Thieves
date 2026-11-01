"""Calibration maps (finding 12): fit on val fit, check on cal, freeze with a hash, apply without labels."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ftm.eval import calibrate as cal

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("calibrate_external", ROOT / "scripts" / "calibrate_external.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def _log(path: Path, system="react", n=400, split="val", label=True, levels=None, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    z = -1.4 + 2.2 * (2 * y - 1) * 0.5 + rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-(3.0 * z + 1.5)))  # over-confident and biased up: Platt should pull it back
    if levels:
        p = np.where(y == 1, levels[1], levels[0]) * np.ones(n)
        p = np.where(rng.random(n) < 0.3, p[::-1], p)
    parts = np.array(["fit", "cal", "eval"])[np.arange(n) % 3]
    p = p.astype(object)
    p[::17] = None  # no verdict
    df = pl.DataFrame({"system": system, "variant": "harness", "seed": 0, "split": split, "alert_id": [f"a{i}" for i in range(n)], "event": "verdict",
                       "probability": [None if v is None else float(v) for v in p], "p0": 0.2, "label": y if label else [None] * n,
                       "val_part": parts if split == "val" else [None] * n}, schema_overrides={"label": pl.Int64})
    df.write_parquet(path)
    return path


def test_platt_fit_improves_cal_and_freeze_roundtrip(tmp_path):
    f = _log(tmp_path / "a.parquet")
    df = cal.labelled_verdicts([f])
    m = cal.fit_map(df, "react")
    assert m["method"] == "platt" and m["check"]["brier_mapped"] < m["check"]["brier_raw"] and "rejected" not in m
    fz = cal.freeze_map(m, tmp_path / "m.json")
    assert cal.load_map(tmp_path / "m.json")["sha256"] == fz["sha256"]
    (tmp_path / "m.json").write_text(json.dumps({**fz, "params": {"a": 9.0, "b": 0.0}}))
    with pytest.raises(ValueError, match="own hash"):
        cal.load_map(tmp_path / "m.json")


def test_apply_needs_no_label_and_keeps_missing_verdicts_null(tmp_path):
    fit = _log(tmp_path / "a.parquet")
    m = cal.freeze_map(cal.fit_map(cal.labelled_verdicts([fit]), "react"), tmp_path / "m.json")
    blind = _log(tmp_path / "b.parquet", split="test", label=False, seed=1)
    out = tmp_path / "o.parquet"
    assert script.main(["apply", "--map", str(tmp_path / "m.json"), "--runlog", str(blind), "--out", str(out), "--expect-sha", m["sha256"]]) == 0
    o = pl.read_parquet(out)
    raw = o["probability_raw"]
    assert o["probability"].null_count() == raw.null_count() > 0
    nz = o.filter(pl.col("probability").is_not_null())
    assert (nz["probability"] != nz["probability_raw"]).all()
    with pytest.raises(ValueError, match="already calibrated"):
        cal.apply_map_to_frame(o, cal.load_map(tmp_path / "m.json"))


def test_fit_refuses_blind_and_test_logs(tmp_path):
    with pytest.raises(ValueError, match="no labels"):
        cal.labelled_verdicts([_log(tmp_path / "a.parquet", split="test", label=False)])


def test_useless_map_is_rejected_to_identity(tmp_path):
    rng = np.random.default_rng(3)
    n = 300
    df = pl.DataFrame({"system": "react", "variant": "h", "seed": 0, "alert_id": [str(i) for i in range(n)], "probability": rng.uniform(0.05, 0.95, n),
                       "p0": 0.2, "label": (rng.random(n) < 0.3).astype(int), "val_part": np.array(["fit", "cal"])[np.arange(n) % 2]})
    # label independent of p: whatever Platt does, it must not be trusted unless cal improves
    m = cal.fit_map(df, "react")
    if m["check"]["improves"]:
        assert m["method"] == "platt"
    else:
        assert m["method"] == "identity" and "rejected" in m
        assert np.allclose(cal.apply_params(m["method"], m["params"], [0.3, 0.8]), [0.3, 0.8])


def test_levels_and_sharma_verdict_maps(tmp_path):
    f = _log(tmp_path / "s.parquet", system="sharma_ported", levels=(0.1, 0.9))
    df = cal.labelled_verdicts([f])
    m = cal.fit_map(df, "sharma_ported")
    assert m["method"] in ("sharma_verdict", "identity")
    p = cal.apply_params("sharma_verdict", {"fraud_in": 0.9, "not_fraud_in": 0.1, "fraud": 0.5, "not_fraud": 0.05}, [0.9, 0.1, 0.42])
    assert p.tolist() == [0.5, 0.05, 0.42]  # escalate (p0) untouched
    lv = cal.fit_levels([0.05, 0.05, 0.95, 0.95], [0, 0, 1, 0])
    out = cal.apply_params("levels", lv, [0.05, 0.95, 0.33])
    assert out[2] == 0.33 and 0.0 < out[0] < out[1] < 1.0
