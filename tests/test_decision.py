"""T4: C14 decision threshold and stop rule."""

import numpy as np
import pytest

from ftm.decision import threshold as th


def test_bayes_threshold():
    assert th.bayes_threshold(cost_fn=9, cost_fp=1) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        th.bayes_threshold(0, 1)


def test_cost_fit_recovers_bayes_threshold_on_calibrated_scores():
    rng = np.random.default_rng(0)
    p = rng.beta(0.5, 3, 200_000)
    y = (rng.random(len(p)) < p).astype(int)  # calibrated by construction
    t = th.fit_threshold(p, y, "cost", cost_fn=4, cost_fp=1)
    assert t == pytest.approx(th.bayes_threshold(4, 1), abs=0.02)


def test_capacity_threshold_respects_every_day():
    rng = np.random.default_rng(1)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    day = np.repeat(np.arange(10), 100)
    t = th.fit_threshold(p, y, "capacity", day=day, capacity_per_day=5)
    esc = p >= t
    assert all(esc[day == d].sum() <= 5 for d in range(10))
    assert any(esc[day == d].sum() == 5 for d in range(10))  # the lowest such threshold, so capacity is used


def test_f1_and_errors():
    p = np.array([0.9, 0.8, 0.3, 0.1])
    y = np.array([1, 1, 0, 0])
    assert th.fit_threshold(p, y, "f1") == pytest.approx(0.8)
    with pytest.raises(ValueError):
        th.fit_threshold(p, y, "capacity")
    with pytest.raises(ValueError):
        th.fit_threshold(p, y, "accuracy")


def test_decide():
    assert th.decide([0.2, 0.5, 0.7], 0.5).tolist() == ["clear", "escalate", "escalate"]


def test_stop_budget_first():
    s = th.should_stop(0.5, 0.5, eps_left=0.1, min_eps=0.25, calls_left=5, sim_days_left=2, fast_says_clear=True)
    assert s == th.StopDecision(True, "budget")
    assert th.should_stop(0.5, 0.5, eps_left=2, min_eps=0.25, calls_left=0, sim_days_left=2).reason == "budget"
    assert th.should_stop(0.5, 0.5, eps_left=2, min_eps=0.25, calls_left=5, sim_days_left=0).reason == "budget"


def test_stop_fast_model_clear():
    s = th.should_stop(0.4, 0.5, eps_left=2, min_eps=0.25, calls_left=5, sim_days_left=2, fast_says_clear=True)
    assert s.reason == "fast_clear"


def test_stop_when_ceiling_cannot_cross_threshold():
    # posterior 0.01 vs threshold 0.5: log-odds gap 4.6; eps 2 with yes/no answers moves it at most 2
    assert th.should_stop(0.01, 0.5, eps_left=2, min_eps=0.25, calls_left=5, sim_days_left=2).reason == "cannot_cross"
    # counts with a gap of 3 per unit eps could move it 6: keep asking
    s = th.should_stop(0.01, 0.5, eps_left=2, min_eps=0.25, calls_left=5, sim_days_left=2, max_delta=3)
    assert s == th.StopDecision(False, "continue")
    # close to the threshold: keep asking
    assert not th.should_stop(0.4, 0.5, eps_left=1, min_eps=0.25, calls_left=5, sim_days_left=2).stop


def test_ceiling_matches_proposal_example():
    # proposal step 3: at eps = 1, yes/no answers cannot move 0.46 above ~0.70
    z = np.log(0.46 / 0.54) + th.max_log_odds_shift(1.0)
    assert 1 / (1 + np.exp(-z)) == pytest.approx(0.70, abs=0.005)


def _brute(p, y, objective, cost_fn=1.0, cost_fp=1.0):
    best_t, best = None, -np.inf
    for t in sorted(set(p) | {np.nextafter(1.0, 2.0)}, reverse=True):
        esc = p >= t
        if objective == "cost":
            s = -(cost_fn * ((~esc) & (y == 1)) + cost_fp * (esc & (y == 0))).mean()
        else:
            tp = (esc & (y == 1)).sum()
            d = esc.sum() + (y == 1).sum()
            s = 2 * tp / d if d else 0.0
        if s > best + 1e-15:
            best, best_t = s, t
    return best_t


@pytest.mark.parametrize("seed", range(5))
def test_vectorised_fit_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    p = np.round(rng.random(300), 2)  # rounding makes ties
    y = (rng.random(300) < p).astype(int)
    assert th.fit_threshold(p, y, "cost", cost_fn=3, cost_fp=1) == pytest.approx(_brute(p, y, "cost", 3, 1))
    assert th.fit_threshold(p, y, "f1") == pytest.approx(_brute(p, y, "f1"))
