"""Replay math of scripts/t3_val_analysis.py on synthetic rows (no run logs needed)."""

import importlib.util
import math
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("t3_val_analysis", Path(__file__).resolve().parents[1] / "scripts" / "t3_val_analysis.py")
va = importlib.util.module_from_spec(spec)
spec.loader.exec_module(va)


def step(log_lr, table_log_lr=None, family="f", kind="reply"):
    return {"log_lr": log_lr, "table_log_lr": table_log_lr if table_log_lr is not None else log_lr, "family": family, "response_kind": kind}


def test_identity_and_no_contribs():
    assert va.replay_prob(0.3, []) == 0.3
    p = va.replay_prob(0.5, [math.log(3.0)])
    assert abs(p - 0.75) < 1e-9


def test_shrinkage():
    z = va.logit(0.2)
    assert abs(va.replay_prob(0.2, [1.0, 0.5], lam=0.5) - va.sigmoid(z + 0.75)) < 1e-9
    assert abs(va.replay_prob(0.2, [1.0, 0.5], lam=0.0) - 0.2) < 1e-9


def test_caps():
    assert abs(va.replay_prob(0.5, [2.0, -0.5], cap_answer=1.0) - va.sigmoid(0.5)) < 1e-9
    assert abs(va.replay_prob(0.5, [2.0, 2.0], cap_total=1.5) - va.sigmoid(1.5)) < 1e-9
    assert va.replay_prob(0.5, [-3.0], cap_total=1.0) < 0.5


def test_abstain():
    assert va.replay_prob(0.4, [0.3, 0.2], tau=0.6) == 0.4
    assert va.replay_prob(0.4, [0.3, 0.4], tau=0.6) != 0.4
    assert va.replay_prob(0.4, [-0.7], tau=0.6) < 0.4


def test_accepted_only_vs_all_entries():
    steps = [step(0.5), step(None, table_log_lr=0.9), step(0.2, family="g")]
    assert va.select_contribs(steps) == [0.5, 0.2]  # live rule: accepted entries only
    assert va.select_contribs(steps, accepted_only=False) == [0.5, 0.9, 0.2]
    assert va.select_contribs(steps, drop_families={"g"}) == [0.5]
    assert va.select_contribs([step(0.5, kind="timeout")]) == []  # timeouts / refusals never update


def test_prob_vector_and_fit_grid():
    alerts = [{"p0": 0.5, "label": 1, "steps": [step(1.0)]}, {"p0": 0.5, "label": 0, "steps": [step(-1.0)]}]
    p1 = va.prob_vector(alerts)
    assert p1[0] > 0.5 > p1[1]
    y = np.array([1, 0])
    lam, b = va.fit_grid(lambda g: va.brier(y, va.prob_vector(alerts, lam=g)), [0.0, 0.5, 1.0, 1.5])
    assert lam == 1.5 and b < va.brier(y, [0.5, 0.5])  # informative updates: more weight is better
    flipped = [{**alerts[0], "label": 0}, {**alerts[1], "label": 1}]
    lam, _ = va.fit_grid(lambda g: va.brier(np.array([0, 1]), va.prob_vector(flipped, lam=g)), [0.0, 0.5, 1.0])
    assert lam == 0.0  # updates point the wrong way: keep p0


def test_logistic_slope():
    rng = np.random.default_rng(0)
    off = rng.normal(size=4000)
    x = rng.normal(size=4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-(off + 0.5 * x)))).astype(float)
    assert abs(va.logistic_offset_slope(y, off, x) - 0.5) < 0.12


def test_smoothed_log_lrs():
    rows = [("g", "a", 1)] * 6 + [("g", "a", 0)] * 2 + [("g", "b", 1)] * 2 + [("g", "b", 0)] * 6
    t = va.smoothed_log_lrs(rows, alpha=1.0)
    # n1 = 8, n0 = 8, K = 2: P(a|1) = 7/10, P(a|0) = 3/10
    assert abs(t[("g", "a")] - math.log(0.7 / 0.3)) < 1e-9
    assert abs(t[("g", "a")] + t[("g", "b")]) < 1e-9  # symmetric counts
    big = va.smoothed_log_lrs(rows, alpha=1e9)
    assert max(abs(v) for v in big.values()) < 1e-6  # alpha -> infinity: no update
    assert abs(va.smoothed_log_lrs(rows, alpha=0.0)[("g", "a")] - math.log(3.0)) < 1e-9  # raw ratio (6/8) / (2/8)


def test_logistic_ridge_shrinks_answer_coefficients():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(500), rng.normal(size=500), rng.normal(size=500)])
    y = (rng.random(500) < 1 / (1 + np.exp(-(0.3 + X[:, 1] + 1.0 * X[:, 2])))).astype(float)
    free = va.logistic_ridge(X, y, [0, 0, 0])
    pen = va.logistic_ridge(X, y, [0, 0, 1e6])
    assert abs(free[2] - 1.0) < 0.3 and abs(pen[2]) < 0.01 and abs(pen[1] - free[1]) < 0.4


def test_auc_midrank_and_degenerate():
    assert va.auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 1.0
    assert va.auc([0, 0, 1, 1], [0.4, 0.3, 0.2, 0.1]) == 0.0
    assert va.auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5  # all tied
    assert abs(va.auc([0, 1, 1, 0], [0.1, 0.9, 0.1, 0.5]) - 0.625) < 1e-12  # 2 pos x 2 neg: (1 + 0.5 + 1 + 0)/4
    assert va.auc([1, 1], [0.1, 0.2]) is None


def test_value_and_extreme_shares():
    p = [0.85, 0.85, 0.5, 0.01, 0.99]
    assert abs(va.value_share(p, 0.85) - 0.4) < 1e-12
    assert abs(va.extreme_share(p) - 0.4) < 1e-12
    assert va.value_share([], 0.5) == 0.0


def test_auc_pair_ci_and_cv_brier():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(40), np.ones(40)].astype(int)
    p0 = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.15, 80), 0.01, 0.99)
    p = rng.uniform(0, 1, 80)
    cl = np.arange(80) // 2
    r = va.auc_pair_ci(y, p, p0, cl, 200)
    assert r["auc_p0"] > 0.9 and abs(r["auc"] - 0.5) < 0.2 and r["diff"] < 0
    assert r["diff_ci"][1] < 0  # CI of the AUC difference excludes zero here
    folds = np.arange(80) % 5
    z = np.log(p0 / (1 - p0))
    assert va.cv_brier(y, z[:, None], folds) < 0.25
