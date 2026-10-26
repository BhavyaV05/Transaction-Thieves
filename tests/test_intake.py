"""T1: conformal intake, alert drop rules, locked subset, p0 schema. Synthetic data only."""


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


def test_span_flag_only_never_filters():
    a = pl.DataFrame({"alert_id": [1, 2, 3, 4], "attempt_id": [10, 11, None, 10]})
    sp = pl.DataFrame({"attempt_id": [10, 11], "spans_split": [True, False]})
    out = alerts.add_span_flag(a, sp).sort("alert_id")
    assert out.height == 4  # nothing dropped
    assert out["spans_split"].to_list() == [True, False, False, True]
    assert set(out.columns) == {"alert_id", "attempt_id", "spans_split"}  # no drop_window / keep: the 72 h rule is gone


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
        alerts.stratified_subset(_toy_alerts(), strata="label_typology_bank")
    monkeypatch.setenv("FTM_FINAL_RUN", "1")
    a = _toy_alerts()
    s1 = alerts.stratified_subset(a, n=200, seed=3, strata="label_typology_bank")
    s2 = alerts.stratified_subset(a, n=200, seed=3, strata="label_typology_bank")
    assert s1.height == 200 and s1["alert_id"].n_unique() == 200
    assert s1["alert_id"].sort().to_list() == s2["alert_id"].sort().to_list()  # seeded
    # proportional: the label share is close to the population share
    assert abs(s1["label"].mean() - a["label"].mean()) < 0.05
    # every stratum is represented
    key = ["label", "typology", "bank"]
    strata = lambda d: set(d.with_columns(pl.col("typology").fill_null("none")).select(key).rows())  # noqa: E731
    assert strata(s1) == strata(a)


def _toy_alerts_p0(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"alert_id": np.arange(n), "bank": rng.choice([1, 2, 3, 4], n, p=[0.5, 0.3, 0.15, 0.05]), "p0": rng.random(n) ** 2})


def test_strata_default_is_label_free_and_locked(monkeypatch):
    a = _toy_alerts_p0()  # no label / typology columns at all
    monkeypatch.delenv("FTM_FINAL_RUN", raising=False)
    with pytest.raises(guards.TestSplitLocked):
        alerts.stratified_subset(a)
    monkeypatch.setenv("FTM_FINAL_RUN", "1")
    s1 = alerts.stratified_subset(a, n=300)
    s2 = alerts.stratified_subset(a, n=300)
    assert s1.height == 300 and s1["alert_id"].n_unique() == 300
    assert s1["alert_id"].sort().to_list() == s2["alert_id"].sort().to_list()  # seed 0 by default
    assert s1["alert_id"].sort().to_list() != alerts.stratified_subset(a, n=300, seed=1)["alert_id"].sort().to_list()
    key = ["bank", "p0_decile"]
    assert set(s1.select(key).rows()) == set(alerts.stratified_subset(a, n=4000).select(key).rows())  # every stratum present
    assert sorted(s1["p0_decile"].unique().to_list()) == list(range(10))
    # proportional: the share of the largest bank is close to its share in the population
    assert abs((s1["bank"] == 1).mean() - (a["bank"] == 1).mean()) < 0.05
    # p0 does not depend on a label, so shuffling labels cannot change the subset
    assert alerts.stratified_subset(a.with_columns(pl.lit(0).alias("label")), n=300)["alert_id"].sort().to_list() == s1["alert_id"].sort().to_list()


def test_frozen_alerts_and_the_final_run_are_locked_and_use_the_val_threshold(monkeypatch, tmp_path):
    import json
    import sys
    from datetime import datetime
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import make_alerts

    from ftm import guards
    from ftm.baselines import gfp_gbt as g
    from ftm.data import io

    p0 = pl.DataFrame({"tx_id": [1, 2, 3, 4, 5], "bank": [7, 7, 8, 8, 7], "split": ["test"] * 5, "val_part": [None] * 5, "label": [False, True, False, True, False],
                       "p0": [0.01, 0.5, None, 0.2, float("nan")]}, schema_overrides={"val_part": pl.Utf8})  # fmt: skip
    t = datetime(2022, 9, 20)
    tx = pl.DataFrame({"tx_id": [1, 2, 3, 4, 5], "timestamp": [t, t, t, t, t], "day": [13, 13, 13, 18, 14], "from_bank": [7, 7, 9, 9, 7], "to_bank": [8, 9, 8, 8, 7]})
    f = alerts.frozen_alerts(p0.join(tx, on="tx_id"), 0.2)
    assert f["tx_id"].to_list() == [2, 4]  # >= the threshold; null and NaN p0 are never alerts
    assert alerts.test_alert_ids(f)["alert_id"].to_list() == ["F000000", "F000001"]

    monkeypatch.delenv("FTM_FINAL_RUN", raising=False)
    with pytest.raises(guards.TestSplitLocked):
        make_alerts.final_run("toy", {"bulk_days": {"toy": 16}, "targets": {}})
    monkeypatch.setenv("FTM_FINAL_RUN", "1")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "alerts").mkdir()
    (tmp_path / "data" / "runs").mkdir(parents=True)
    (tmp_path / "data" / "runs" / "alerts_toy.json").write_text(json.dumps({"thresholds": {"t_alert": 0.2}, "alpha_launder_used": 0.6, "sweep": [{"alpha_launder": 0.6, "proj_test_alerts": 3.0}]}))
    monkeypatch.setattr(io, "load_config", lambda ds: {"parquet_dir": "pq", "alerts_dir": str(tmp_path / "alerts")})
    monkeypatch.setattr(io, "scan_transactions", lambda *a, **k: tx.lazy())
    monkeypatch.setattr(io, "scan_patterns", lambda *a, **k: pl.DataFrame({"tx_id": [2], "attempt_id": [5], "typology": ["cycle"]}).lazy())
    monkeypatch.setattr(io, "scan_attempts", lambda *a, **k: pl.DataFrame({"attempt_id": [5], "spans_split": [True]}).lazy())
    monkeypatch.setattr(g, "read_p0", lambda *a, **k: p0)
    make_alerts.final_run("toy", {"bulk_days": {"toy": 16}, "targets": {"test": [1, 5]}})
    out = pl.read_parquet(tmp_path / "alerts" / "test.parquet")
    assert out["alert_id"].to_list() == ["F000000", "F000001"] and out["test_tail"].to_list() == [False, True]  # day 18 is the sparse tail
    assert out["spans_split"].to_list() == [True, False] and out["typology"].to_list() == ["cycle", None]
    assert out["direction"].to_list() == ["out", "in"] and out["counterparty_bank"].to_list() == [9, 9]
    rec = json.loads((tmp_path / "data" / "runs" / "alerts_toy_test.json").read_text())
    assert rec["alerts"] == 2 and rec["bulk_alerts"] == 1 and rec["t_alert_from_val"] == 0.2
