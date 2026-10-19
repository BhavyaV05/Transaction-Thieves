"""T1: conformal intake, alert drop rules, locked subset, p0 schema. Synthetic data only."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ftm import guards
from ftm.intake import alerts, conformal


def _scores(rng, n, prev=0.2):
    y = (rng.random(n) < prev).astype(int)
    p = np.clip(rng.beta(2, 5, n) + 0.4 * y * rng.random(n), 0, 1)
    return p, y


def test_mondrian_coverage_per_class():
    rng = np.random.default_rng(0)
    miss = []
    for _ in range(100):
        p, y = _scores(rng, 3000)
        tl, tc = conformal.mondrian_thresholds(p[:1500], y[:1500], 0.10, 0.05)
        m = conformal.achieved_miss_rates(p[1500:], y[1500:], tl, tc)
        miss.append((m["launder_miss"], m["clean_miss"]))
    a1, a0 = np.mean(miss, axis=0)
    assert a1 <= 0.10 + 0.01 and a1 > 0.07  # valid, and not wildly conservative
    assert a0 <= 0.05 + 0.01 and a0 > 0.035


def test_weighted_quantile_unit_weights_is_order_statistic():
    s = np.arange(1.0, 11.0)
    # ceil((n+1)*level) = ceil(11 * 0.8) = 9th smallest
    assert conformal.weighted_quantile_with_test(s, np.ones(10), 0.8) == 9.0
    assert conformal.weighted_quantile_with_test(s, np.ones(10), 0.95) == float("inf")


def test_time_weights_decay_and_shift_threshold():
    w = conformal.time_weights([1, 2, 3, 4], half_life_days=1.0)
    np.testing.assert_allclose(w, [0.125, 0.25, 0.5, 1.0])
    assert np.all(conformal.time_weights([1, 5], None) == 1)
    # old laundering rows have high scores, recent ones low: weighting recent rows lowers the threshold
    rng = np.random.default_rng(1)
    days = np.repeat([1, 2, 3, 4], 200)
    p = np.where(days <= 2, rng.uniform(0.5, 1, 800), rng.uniform(0.0, 0.5, 800))
    y = np.ones(800, int)
    t_plain, _ = conformal.mondrian_thresholds(p, y, 0.2, 0.1, days=days, half_life_days=None)
    t_w, _ = conformal.mondrian_thresholds(p, y, 0.2, 0.1, days=days, half_life_days=0.5)
    assert t_w < t_plain


def test_selective_clear_threshold_bounds_risk():
    rng = np.random.default_rng(2)
    p = rng.random(20000) ** 3
    y = (rng.random(20000) < 0.05 * p).astype(int)
    lam = conformal.selective_clear_threshold(p[:10000], y[:10000], beta=0.01, delta=0.1)
    assert lam > 0
    cleared = p[10000:] < lam
    assert cleared.sum() > 100 and y[10000:][cleared].mean() <= 0.01
    # impossible target -> clears nothing
    assert conformal.selective_clear_threshold(p, np.ones_like(y), beta=0.01) == 0.0


def test_fixed_window():
    assert conformal.fixed_window([0.1, 0.15, 0.5, 0.85, 0.9]).tolist() == [False, True, True, True, False]


def test_drop_rules():
    t0 = datetime(2022, 9, 7)
    a = pl.DataFrame({
        "alert_id": [1, 2, 3, 4], "attempt_id": [10, 11, None, 10],
        "timestamp": [t0 + timedelta(hours=h) for h in (100, 100, 100, 10)], "split": ["val"] * 4,
    })  # fmt: skip
    sp = pl.DataFrame({"attempt_id": [10, 11], "spans_split": [True, False]})
    out = alerts.add_drop_flags(a, sp, {"val": t0}, window_hours=72).sort("alert_id")
    assert out["drop_spans_split"].to_list() == [True, False, False, True]
    assert out["drop_window"].to_list() == [False, False, False, True]  # 10 h after the split start < 72 h
    assert out["keep"].to_list() == [False, True, True, False]


def test_project_test_count():
    assert alerts.project_test_count(300.0, 4) == 1200.0


def _toy_alerts(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "alert_id": np.arange(n), "label": (rng.random(n) < 0.3).astype(int),
        "typology": pl.Series([None if t == "none" else t for t in rng.choice(["fan_in", "cycle", "stack", "none"], n)], dtype=pl.Utf8), "bank": rng.choice([1, 2, 3], n),
    })  # fmt: skip


def test_stratified_subset_is_locked_and_works_with_flag(monkeypatch):
    monkeypatch.delenv("FTM_FINAL_RUN", raising=False)
    with pytest.raises(guards.TestSplitLocked):
        alerts.stratified_subset(_toy_alerts())
    monkeypatch.setenv("FTM_FINAL_RUN", "1")
    a = _toy_alerts()
    s1 = alerts.stratified_subset(a, n=200, seed=3)
    s2 = alerts.stratified_subset(a, n=200, seed=3)
    assert s1.height == 200 and s1["alert_id"].n_unique() == 200
    assert s1["alert_id"].sort().to_list() == s2["alert_id"].sort().to_list()  # seeded
    # proportional: the label share is close to the population share
    assert abs(s1["label"].mean() - a["label"].mean()) < 0.05
    # every stratum is represented
    key = ["label", "typology", "bank"]
    strata = lambda d: set(d.with_columns(pl.col("typology").fill_null("none")).select(key).rows())  # noqa: E731
    assert strata(s1) == strata(a)
