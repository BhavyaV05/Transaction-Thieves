"""T4: the G2 check (expert questions at eps 2, under the ceiling) on the synthetic world."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from ftm.eval import g2

EPS = (1.0, 2.0)
FIELDS = {"world", "fusion", "ask_eps", "budget", "n_alerts", "n_pos", "pool", "brier_p0", "brier_post", "gain",
          "gain_ci95", "gain_naive", "gain_min", "gain_ok", "clusters", "mean_questions", "mean_eps", "max_eps",
          "budget_ok", "families_used", "fusion_matches_planner", "ceiling", "ceiling_ok", "passed"}  # fmt: skip


@pytest.fixture(scope="module")
def good():
    return g2.synthetic_inputs(360, 3, eps_levels=EPS)


@pytest.fixture(scope="module")
def result(good):
    banks, entries, val = good
    return g2.run_g2(banks, entries, val, ask_eps=1.0, eps_levels=EPS, n_boot=300)


def tampered(entries):
    """The entries as a partner bank that forgot its noise would have produced them: every cell pure (cells above the minimum count), so the
    unclipped LR is far above the ceiling."""
    out = []
    for e in entries:
        e = dict(e)
        if e.get("cells") and e.get("ceiling") not in (None, math.inf):
            e["cells"] = {k: [1000, 50] if i == 0 else [50, 1000] for i, k in enumerate(e["cells"])}
        out.append(e)
    return out


def test_runs_end_to_end_and_reports_the_documented_fields(result):
    r = result
    assert FIELDS <= set(r) and r["world"] == "synth" and r["fusion"] == "naive"
    assert r["n_alerts"] > 100 and r["pool"] >= 1 and r["mean_questions"] > 0.5 and r["families_used"]
    lo, hi = r["gain_ci95"]
    assert lo <= hi  # a CI is reported
    assert r["brier_p0"] - r["brier_post"] == pytest.approx(r["gain"])
    assert r["clusters"]["n_rows"] == r["n_alerts"] and r["ceiling"]["ok"] is r["ceiling_ok"]
    assert r["budget_ok"] and r["max_eps"] <= r["budget"] + 1e-9  # the harness held the budget
    assert r["fusion_matches_planner"]  # fusion by hand equals the planner's running posterior
    assert g2.verdict_line(r).startswith(f"G2 {'PASS' if r['passed'] else 'FAIL'} [synth")


def test_useful_questions_pass_both_criteria(result):
    assert result["gain"] >= g2.GAIN_MIN and result["gain_ok"]
    assert result["ceiling_ok"] and not result["ceiling"]["violations"]
    assert result["passed"]


def test_useless_questions_fail_the_gain_criterion():
    banks, entries, val = g2.synthetic_inputs(360, 3, signal=0.0, eps_levels=EPS)  # no label difference at any bank
    r = g2.run_g2(banks, entries, val, ask_eps=1.0, eps_levels=EPS, n_boot=300)
    assert r["gain"] < g2.GAIN_MIN and not r["gain_ok"] and not r["passed"]
    assert r["ceiling_ok"]  # the privacy code is fine; only the questions are useless


def test_a_noise_free_lr_cache_fails_the_ceiling_criterion(good, result):
    banks, entries, val = good
    r = g2.run_g2(banks, tampered(entries), val, ask_eps=1.0, eps_levels=EPS, n_boot=300)
    assert not r["ceiling_ok"] and r["ceiling"]["violations"] and not r["passed"]
    assert {v["canonical"] for v in r["ceiling"]["violations"]} <= {e["program_canonical"] for e in entries}
    assert r["gain_ok"]  # the gain alone would have passed: it is the ceiling that fails


def test_per_family_fusion_is_cross_fitted_over_whole_clusters(good):
    cl = np.repeat(np.arange(40), 3)  # 40 clusters of 3 alerts
    folds = g2.fold_labels(cl, 5, 0)
    assert len(np.unique(folds)) == 5 and all(len(set(folds[cl == c])) == 1 for c in np.unique(cl))
    rng = np.random.default_rng(0)
    y = (rng.random(120) < 0.3).astype(float)
    p0 = np.full(120, 0.3)
    X = (y * 1.5 + rng.normal(0, 1, 120))[:, None]
    post = g2.crossfit_posterior(p0, X, y, cl, "per_family")
    assert post.shape == (120,) and np.all((post > 0) & (post < 1))
    assert np.mean((post - y) ** 2) < np.mean((p0 - y) ** 2)  # real signal survives the cross-fit
    pure_noise = rng.normal(0, 1, (120, 1))
    post_n = g2.crossfit_posterior(p0, pure_noise, y, cl, "per_family")
    assert np.mean((post_n - y) ** 2) > np.mean((post - y) ** 2)
    banks, entries, val = good
    r = g2.run_g2(banks, entries, val, fusion="per_family", ask_eps=1.0, eps_levels=EPS, n_boot=200)
    assert r["fusion"] == "per_family" and r["gain_naive"] == pytest.approx(
        g2.run_g2(banks, entries, val, ask_eps=1.0, eps_levels=EPS, n_boot=50)["gain"])  # naive gain is always reported


def test_the_script_exits_2_when_the_real_data_is_not_there(tmp_path):
    spec = importlib.util.spec_from_file_location("g2_check", Path(__file__).resolve().parents[1] / "scripts" / "g2_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "g2.json"
    assert mod.main(["--world", "real", "--data", "no_such_dataset", "--out", str(out)]) == mod.NOT_RUN
    assert not out.exists()
