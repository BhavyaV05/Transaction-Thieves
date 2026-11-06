"""p0 v2 recalibration (finding 16-17): maps recover known shifts, are frozen and hashed, and keep p0 order; alert-set helpers (finding 20)."""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ftm.baselines import p0_recal as R
from ftm.data import io

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _toy(n=20000, shift=-0.8, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.beta(0.4, 4, n)
    y = (rng.uniform(size=n) < R.apply_map({"kind": "logit_affine", "a": shift, "b": 1.0}, p)).astype(float)
    return p, y


def test_intercept_recovers_a_prior_shift_and_keeps_order():
    p, y = _toy()
    m = R.fit_map("intercept", p, y, ridge=1e-6)
    assert m["b"] == 1.0 and m["a"] == pytest.approx(-0.8, abs=0.15)
    q = R.apply_map(m, p)
    assert np.all(np.diff(q[np.argsort(p)]) >= 0)  # monotone: the alert set cannot change
    assert q.mean() == pytest.approx(y.mean(), rel=0.1)


def test_platt_and_isotonic_are_monotone_and_bounded():
    p, y = _toy(seed=1)
    for kind in ("platt", "isotonic"):
        q = R.apply_map(R.fit_map(kind, p, y), p)
        assert q.min() >= 0 and q.max() <= 1
        assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)


def test_finite_when_scores_separate_labels():
    p = np.r_[np.full(50, 1e-4), np.full(50, 0.9999)]
    y = np.r_[np.zeros(50), np.ones(50)]
    a, b = R.fit_logit_recal(p, y, "platt")
    assert np.isfinite([a, b]).all()
    with pytest.raises(ValueError):
        R.fit_logit_recal(p, np.zeros(100))


def test_frozen_map_roundtrip_and_hash_guard(tmp_path):
    p, y = _toy()
    m = R.fit_map("isotonic", p, y)
    saved = R.save_map(m, tmp_path / "m.json", note="x")
    got = R.load_map(tmp_path / "m.json", expect_sha256=saved["sha256"])
    assert np.allclose(R.apply_map(got, p[:100]), R.apply_map(m, p[:100]))
    d = json.loads((tmp_path / "m.json").read_text())
    d["y"][0] += 0.01
    (tmp_path / "m.json").write_text(json.dumps(d))
    with pytest.raises(ValueError, match="hash"):
        R.load_map(tmp_path / "m.json")
    assert R.map_hash(m) == R.map_hash({**m, "extra": 1})  # provenance does not enter the hash


def test_committed_hi_medium_map_is_intact():
    f = Path(__file__).resolve().parents[1] / "configs" / "t1_final" / "p0_recal_hi_medium.json"
    if f.exists():
        assert R.load_map(f)["kind"] in ("logit_affine", "isotonic")


def test_cv_and_selection_rule():
    p, y = _toy()
    cv = {k: R.cv_brier(k, p, y) for k in ("identity", "intercept")}
    assert cv["intercept"] < cv["identity"]
    assert R.select_variant({"identity": 0.2, "intercept": 0.1, "platt": 0.1}) == "intercept"
    # an alert-only Platt that wrecks the all-rows Brier is excluded
    assert R.select_candidate({"identity_rows": 0.2, "platt_alerts": 0.1, "intercept_rows": 0.15}, {"identity_rows": 1e-3, "platt_alerts": 1e-2, "intercept_rows": 9e-4}) == "intercept_rows"


def test_cv_fit_mask_and_groups():
    p, y = _toy(4000)
    al = p > 0.3
    g = np.arange(len(p)) % 3
    v = R.cv_brier("intercept", p, y, groups=g, score_mask=al, fit_mask=al)
    assert np.isfinite(v) and 0 < v < 1


def test_alerts_dir_switch(monkeypatch):
    cfg = {"alerts_dir": "a", "alerts_dir_v2": "a_v2"}
    monkeypatch.delenv("FTM_P0_VARIANT", raising=False)
    assert io.alerts_dir(cfg) == Path("a") and io.alerts_dir(cfg, "v2") == Path("a_v2")
    monkeypatch.setenv("FTM_P0_VARIANT", "v2")
    assert io.alerts_dir(cfg) == Path("a_v2")
    with pytest.raises(ValueError):
        io.alerts_dir(cfg, "v3")
    assert "alerts_dir_v2" in io.load_config("hi_medium")


def test_repro_guard_requires_ack_and_logs(tmp_path):
    import train_gfp

    with pytest.raises(SystemExit, match="overlaps our test days"):
        train_gfp.guard_repro(False, "x.csv", "o.json", log=str(tmp_path / "l.log"))
    train_gfp.guard_repro(True, "x.csv", "o.json", log=str(tmp_path / "l.log"))
    assert "acknowledged" in (tmp_path / "l.log").read_text()


def test_make_alerts_day_frac_and_disjoint_parts():
    import make_alerts

    val = pl.DataFrame({"val_part": ["fit"] * 4 + ["cal"] * 2 + ["eval"] * 4})
    assert make_alerts.part_day_frac(val, "eval", "time") == 0.25
    assert make_alerts.part_day_frac(val, "eval", "volume") == pytest.approx(0.4)
    assert make_alerts.part_day_frac(val, "cal", "volume") == pytest.approx(0.2)
    make_alerts.check_disjoint("fit", ["cal"])
    make_alerts.check_disjoint("cal", None)
    with pytest.raises(SystemExit, match="disjoint"):
        make_alerts.check_disjoint("cal", ["cal"])
