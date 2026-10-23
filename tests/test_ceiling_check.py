"""T4: ceiling checks on the privacy code (G2)."""

import math

import pytest

from ftm.eval import ceiling
from ftm.privacy import mechanisms as mech


@pytest.mark.parametrize("g", ceiling.default_grid(), ids=lambda g: f"{g['kind']}-{g['eps']}-{g.get('sensitivity', 1)}")
def test_exact_likelihoods_never_exceed_ceiling(g):
    kw = {k: v for k, v in g.items() if k != "kind"}
    assert ceiling.exact_check(g["kind"], **kw)["ok"]


@pytest.mark.parametrize("kind,kw", [("bool", {"eps": 1.0}), ("kary", {"eps": 0.5, "arity": 4, "true_b": 2}),
                                     ("count", {"eps": 2.0, "sensitivity": 1, "true_b": 3}),
                                     ("count", {"eps": 1.0, "sensitivity": 3, "true_a": 1, "true_b": 4})])  # fmt: skip
def test_empirical_draws_stay_under_ceiling(kind, kw):
    r = ceiling.empirical_check(kind, n=8000, **kw)
    assert r["cells_judged"] >= 2 and r["ok"] is True, r["worst_z_excess"]


def test_empirical_check_catches_missing_noise(monkeypatch):
    # a partner bank that forgot the noise for yes/no answers: every answer is exact, LR is unbounded
    monkeypatch.setattr(mech, "randomize", lambda kind, v, eps, s, rng, arity=None: bool(v) if rng.random() < 0.97 else not v)
    r = ceiling.empirical_check("bool", eps=0.5, n=8000)
    assert r["ok"] is False and r["worst_z_excess"] > ceiling.Z_CRIT


def test_no_overlap_is_inconclusive_not_a_pass():
    r = ceiling.empirical_check("count", eps=4.0, sensitivity=1, true_a=0, true_b=3, n=5000)
    assert r["ok"] is None


def test_backtest_check_flags_unclipped_excess_only():
    eps = 1.0
    c = math.exp(eps)
    fine = {"canonical": "q1", "eps": eps, "ceiling": c, "cells": {"false": [730, 270], "true": [270, 730]}}
    broken = {"canonical": "q2", "eps": eps, "ceiling": c, "cells": {"false": [950, 50], "true": [50, 950]}}
    exact_arm = {"canonical": "q3", "eps": None, "ceiling": None, "cells": {"false": [1000, 0]}}
    assert ceiling.backtest_check([fine])["ok"]
    r = ceiling.backtest_check([fine, broken, exact_arm])
    assert not r["ok"] and {v["canonical"] for v in r["violations"]} == {"q2"} and r["entries_skipped"] == 1
