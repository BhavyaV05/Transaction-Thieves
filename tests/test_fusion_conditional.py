"""The p0-conditional fusion (ftm.fusion.conditional) and its Fusion-protocol adapter, on the synthetic world and by hand."""

import json
import math

import numpy as np
import polars as pl
import pytest

from ftm.backtest.engine import (
    BacktestCache,
    BacktestConfig,
    Backtester,
    accept,
    lr_table_rows,
    write_alert_rows,
)
from ftm.backtest.synth import make_world
from ftm.fusion.conditional import ConditionalFusion, alert_matrix, choose_shrink, fit_ridge_cv
from ftm.fusion.fuse import logit, sigmoid
from ftm.llm.fusion_adapter import ConditionalLR
from ftm.llm.investigation import LREntry, LRTable, NaiveLRFusion
from ftm.types import QuestionLogEntry, Reply

CFG = BacktestConfig(min_answers=60, min_per_class=10, min_per_bin=15)
INFORMATIVE = "n_counterparties(ALERT.receiver, dir=in, window=before(72))"
DUP = "n_counterparties(ALERT.receiver, dir=in, window=before(48))"  # a near-duplicate of INFORMATIVE: same family
PROGS = [INFORMATIVE, DUP, "returns_home(ALERT.receiver, window=after(72))", "dormant(ALERT.receiver, days=7)"]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    w = make_world(900, seed=5, signal=1.0)
    cache = BacktestCache(tmp_path_factory.mktemp("bt"), "ns")
    bt = Backtester(w.banks, w.alerts, CFG, cache=cache)
    entries = accept(bt.run_batch(PROGS, [1.0, 2.0]), 0.1)
    table = LRTable.from_rows(lr_table_rows(entries))
    rows = pl.read_parquet(write_alert_rows(cache)).to_dicts()
    rng = np.random.default_rng(1)  # a detector score that knows part of the label
    p0_of = {a.alert_id: float(sigmoid(logit(0.2) + 0.9 * (a.label - 0.2) + rng.normal(0, 0.8))) for a in w.alerts}
    return w, table, rows, p0_of


def test_the_design_has_one_row_per_askable_alert_and_zero_for_unanswered(built):
    w, table, rows, p0_of = built
    p0, X, y, fams, ids = alert_matrix(rows, table, p0_of, draws=2)
    assert len(ids) == 2 * len({r["alert_id"] for r in rows}) and X.shape == (len(ids), len(fams))
    assert ids[0] == ids[1] and set(fams) == {t.family for t in table.entries() if t.accepted}
    labels = {a.alert_id: a.label for a in w.alerts}
    assert all(labels[i] == bool(v) for i, v in zip(ids, y, strict=True))
    assert (X != 0).any()


def test_fitted_fusion_beats_p0_out_of_time_and_one_weight_covers_duplicates(built):
    _, table, rows, p0_of = built
    p0, X, y, fams, _ = alert_matrix(rows, table, p0_of, draws=1)
    n = len(y) * 2 // 3
    m = fit_ridge_cv(p0[:n], X[:n], y[:n], fams)
    brier = lambda p: float(np.mean((p - y[n:]) ** 2))  # noqa: E731
    assert brier(m.predict(p0[n:], X[n:])) < brier(p0[n:])
    naive = ConditionalFusion(fitted=False, families=fams, base="p0")
    assert brier(m.predict(p0[n:], X[n:])) <= brier(naive.predict(p0[n:], X[n:])) + 1e-9
    # the shared family (n_counterparties 72h and 48h are near-duplicates): their summed evidence is down-weighted below naive
    i = fams.index(table.get(INFORMATIVE, 2.0).family)
    assert m.w[i] < 1.0


def test_ridge_pulls_toward_naive_and_the_intercept_is_free():
    rng = np.random.default_rng(0)
    p0 = rng.uniform(0.05, 0.6, 300)
    X = rng.normal(size=(300, 2))
    y = (rng.random(300) < sigmoid(logit(p0) + 0.5 * X[:, 0] - 1.0)).astype(float)
    loose, tight = (ConditionalFusion(ridge=r).fit(p0, X, y, ["f", "g"]) for r in (0.01, 1e6))
    assert abs(tight.b - 1) < 1e-3 and np.allclose(tight.w, 1, atol=1e-3) and tight.a == pytest.approx(loose.a, abs=1.0)
    assert tight.a < -0.5  # the intercept still absorbs the miscalibration under a huge ridge
    assert abs(loose.w[0] - 0.5) < abs(tight.w[0] - 0.5)


def test_choose_shrink_zeroes_the_evidence_on_pure_noise_and_keeps_it_on_signal():
    rng = np.random.default_rng(3)
    p0 = rng.uniform(0.1, 0.5, 2000)
    X = rng.normal(size=(2000, 1))
    m = ConditionalFusion(ridge=1.0).fit(p0, X, (rng.random(2000) < p0).astype(float), ["f"])
    m.w = np.array([2.0])  # a deliberately over-confident weight
    choose_shrink(m, p0, X, (rng.random(2000) < p0).astype(float))  # labels independent of X
    assert m.shrink == 0.0
    y = (rng.random(2000) < sigmoid(logit(p0) + X[:, 0])).astype(float)
    m.w = np.array([1.0])
    choose_shrink(m, p0, X, y)
    assert m.shrink >= 0.75 and "shrink_grid_brier" in m.meta


def _log(prog, value, eps=2.0):
    e = QuestionLogEntry(1, "2", prog, eps, gate_decision="allowed")
    e.response = Reply("2", "bool", value, eps, 1.0)
    return e


def test_adapter_posterior_matches_the_model_and_uses_is_the_naive_selection():
    table = LRTable([LREntry("a", 2.0, True, {"true": 4.0, "false": 0.5}, "fa"), LREntry("b", 2.0, True, {"true": 3.0, "false": 0.5}, "fa"),
                     LREntry("c", 2.0, False, {"true": 9.0, "false": 0.1}, "fc")])  # fmt: skip
    m = ConditionalFusion(["fa"], a=-0.3, b=0.8, w=np.array([0.6]), shrink=0.5, fitted=True)
    f = ConditionalLR(m, table)
    log = [_log("a", True), _log("b", True), _log("c", True)]
    x = 0.6 * (math.log(4) + math.log(3))
    assert f.posterior(0.2, log) == pytest.approx(float(sigmoid(-0.3 + 0.8 * logit(0.2) + 0.5 * x)))
    assert [f.uses(e) for e in log] == [NaiveLRFusion(table).uses(e) for e in log] == [True, True, False]
    assert f.posterior(0.3, []) == pytest.approx(float(sigmoid(-0.3 + 0.8 * logit(0.3))))
    unseen = ConditionalLR(ConditionalFusion(["fz"], w=np.array([0.1]), fitted=True), table)  # families the fit did not see: weight 1
    assert unseen.posterior(0.2, log) == pytest.approx(float(sigmoid(logit(0.2) + x / 0.6)))


def test_params_roundtrip_with_a_content_hash_and_a_tampered_file_is_refused(tmp_path):
    m = ConditionalFusion(["f", "g"], a=0.1, b=0.9, w=np.array([0.4, 1.2]), ridge=10.0, shrink=0.75, base="p0", fitted=True,
                          meta={"p0_file": "x.parquet"})  # fmt: skip
    p = tmp_path / "f.json"
    h = m.to_json(p)
    back = ConditionalFusion.from_json(p)
    assert back.content_hash() == h and back.params() == m.params()
    d = json.loads(p.read_text())
    d["w"][0] = 0.5
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="hash"):
        ConditionalFusion.from_json(p)


def test_max_weight_is_the_largest_weight_times_the_shrink():
    m = ConditionalFusion(["f", "g"], w=np.array([0.4, -1.6]), shrink=0.5, fitted=True)
    assert m.max_weight() == pytest.approx(0.8)
    assert ConditionalFusion(["f"], shrink=0.5).max_weight() == pytest.approx(0.5)  # unfitted: naive weights
