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
