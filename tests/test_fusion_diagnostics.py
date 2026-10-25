"""T4: fusion diagnostics on synthetic worlds with known structure (never a result)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from ftm.eval import g2
from ftm.fusion import diagnostics as dg
from ftm.fusion.fuse import logit, sigmoid

N = 4000


def make_world(kind: str, n: int = N, seed: int = 0):
    """(y, p0, clusters, Answers). p0 is calibrated: it is the posterior given a latent score s, log LR(s) = 1.5 s.

    independent  the answer depends on the label only: given the label it is independent of p0
    copy         the answer is a noisy copy of s: it has a marginal LR, but p0 already knows what it says
    flipped      independent answers, but the claimed (train) LRs have the opposite sign of the validation ones
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(float)
    s = rng.normal(np.where(y == 1, 0.75, -0.75), 1.0)
    p0 = sigmoid(logit(0.3) + 1.5 * s)
    yes = rng.random(n) < np.where(y == 1, 0.75, 0.25) if kind != "copy" else (s + 0.2 * rng.normal(size=n)) > 0
    q1, q0 = yes[y == 1].mean(), yes[y == 0].mean()  # P(yes | launder), P(yes | clean): what a back-test would claim
    claimed = {True: np.log(q1 / q0), False: np.log((1 - q1) / (1 - q0))}
    sign = -1.0 if kind == "flipped" else 1.0
    ans = dg.Answers(alert=np.arange(n), question=np.full(n, "q @ eps 1"), family=np.full(n, "f"),
                     cell=np.where(yes, "true", "false"), claimed=np.array([sign * claimed[bool(v)] for v in yes]))  # fmt: skip
    return y, p0, np.arange(n), ans


def run(kind: str, strata: int = 6, **kw):
    y, p0, cl, ans = make_world(kind, **kw)
    return dg.diagnose(y, p0, cl, ans, kind, strata=strata, n_boot=100, n_boot_share=40)


@pytest.fixture(scope="module")
def independent():
    return run("independent")


@pytest.fixture(scope="module")
def copy():
    return run("copy")


@pytest.fixture(scope="module")
def flipped():
    return run("flipped")


def test_independent_answers_conditional_lr_equals_marginal_and_naive_helps(independent):
    r = independent
    assert len(r["cells"]) == 2
    for c in r["cells"]:
        assert c["cond_log_lr"] == pytest.approx(c["val_log_lr"], abs=0.15)
        assert c["val_log_lr"] == pytest.approx(c["claimed_log_lr"], abs=0.01)  # the claim is this very sample's LR
        assert c["direction_agrees"] is True and not c["contradicted"]
    assert r["conditional"]["explained_by_p0"] < 0.25
    assert r["direction"]["share_answers_disagree"] == 0
    assert r["calibration"]["naive"]["gain"] > 0.005
    assert "not what explains the loss" in r["interpretation"][0]


def test_answers_that_copy_p0_have_conditional_lr_near_one_and_naive_hurts(copy):
    r = copy
    for c in r["cells"]:
        assert abs(c["val_log_lr"]) > 0.8  # a large marginal LR ...
        assert abs(c["cond_log_lr"]) < 0.5 * abs(c["val_log_lr"])  # ... that p0 already explains
    assert r["conditional"]["explained_by_p0"] > 0.5
    assert r["calibration"]["naive"]["gain"] < -0.005
    assert "already holds most" in r["interpretation"][0]
    # the fitted fusion with a zero prior and cv must not lose to p0 by more than a small tolerance
    assert r["calibration"]["per_family_cv"]["gain"] > -0.003
    assert r["calibration"]["per_family_cv"]["gain"] > r["calibration"]["naive"]["gain"]
    assert len(r["calibration"]["per_family_cv"]["chosen_per_fold"]) == 5


def test_flipped_validation_lrs_are_flagged_as_a_direction_disagreement(flipped):
    r = flipped
    d = r["direction"]
    assert d["share_answers_disagree"] > 0.9 and d["cells_disagree"] == 2 and d["cells_contradicted"] == 2
    assert all(c["direction_agrees"] is False for c in r["cells"])
    assert "wrong way" in r["interpretation"][1]
    assert r["calibration"]["naive"]["gain"] < -0.01  # trusting the flipped table hurts a lot


def test_report_is_json_and_markdown_ready(independent):
    r = independent
    txt = json.dumps(load_script("fusion_diagnostics")._clean(r), allow_nan=False)
    assert "explained_by_p0" in txt
    md = dg.to_markdown(r)
    assert md.startswith("# Fusion diagnostics: independent") and "| p0 |" in md and "per_family_cv" in md
    assert dg.summary_line(r) == r["summary"] and "explained by p0" in r["summary"]


def test_helpers():
    p0 = np.r_[np.full(50, 0.1), np.full(50, 0.5)]
    lab, edges = dg.p0_strata(p0, 3)
    assert lab.max() == 1 and edges == [0.5]  # ties: fewer strata than asked, never an empty one
    assert dg.p0_strata(np.full(10, 0.2), 4)[0].max() == 0
    lab, _ = dg.p0_strata(np.linspace(0, 1, 300), 3)
    assert np.bincount(lab).tolist() == [100, 100, 100]
    lo, hi = dg.wilson(5, 20)
    assert lo < 0.25 < hi and 0 <= lo and hi <= 1
    # one stratum: the Mantel-Haenszel estimate is the back-test's plain LR (with the half-count correction)
    a, b, n1, n0 = np.array([30.0]), np.array([10.0]), np.array([50.0]), np.array([100.0])
    assert np.exp(dg._log_mh(a, b, n1, n0, n1 + n0)) == pytest.approx((30.5 / 50) / (10.5 / 100))
    # a stratum with no launderer is uninformative and does not move the estimate
    two = dg._log_mh(np.r_[a, 0], np.r_[b, 5], np.r_[n1, 0], np.r_[n0, 20], np.r_[n1 + n0, 20])
    assert two == pytest.approx(dg._log_mh(a, b, n1, n0, n1 + n0), abs=0.02)


def test_no_answers_is_reported_not_crashed():
    y = np.array([0.0, 1, 0, 1, 0, 0])
    ans = dg.Answers(*(np.array([], dtype=t) for t in (int, str, str, str, "f8")))
    r = dg.diagnose(y, np.full(6, 0.3), np.arange(6), ans, "empty", n_boot=20)
    assert r["n_answers"] == 0 and "no answers" in r["summary"] and r["interpretation"][0].startswith("No answered")
    assert "# Fusion diagnostics: empty" in dg.to_markdown(r)


# --- the planner run on the synthetic world and the script --------------------------------------------------------
@pytest.fixture(scope="module")
def planned():
    banks, entries, val = g2.synthetic_inputs(360, 3, eps_levels=(1.0, 2.0))
    return dg.run_diagnostics(banks, entries, val, "synth", ask_eps=1.0, eps_levels=(1.0, 2.0), n_boot=100, n_boot_share=20)


def test_runs_on_the_planner_answers_of_the_synthetic_world(planned):
    r = planned
    assert r["n_alerts"] > 100 and r["n_answers"] > 50 and r["cells"] and r["pool"] >= 1
    assert r["strata"]["k"] == 1  # the synthetic p0 is one constant, so one stratum: conditional = marginal
    assert all(c["cond_log_lr"] == pytest.approx(c["val_log_lr"]) for c in r["cells"])
    assert {"p0", "naive", "shared", "per_family", "per_family_cv"} <= set(r["calibration"])
    # the naive posterior here is g2's: same gain as run_g2 reports
    banks, entries, val = g2.synthetic_inputs(360, 3, eps_levels=(1.0, 2.0))
    base = g2.run_g2(banks, entries, val, ask_eps=1.0, eps_levels=(1.0, 2.0), n_boot=100)
    assert r["calibration"]["naive"]["gain"] == pytest.approx(base["gain"])


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_writes_json_and_markdown_and_exits_2_without_data(tmp_path, capsys):
    mod = load_script("fusion_diagnostics")
    out = tmp_path / "fd.json"
    assert mod.main(["--world", "real", "--data", "no_such_dataset", "--out", str(out)]) == mod.NOT_RUN
    assert not out.exists() and not out.with_suffix(".md").exists()
    assert mod.main(["--world", "synth", "--n", "240", "--eps-levels", "1", "2", "--ask-eps", "1", "--n-boot", "50",
                     "--n-boot-share", "10", "--out", str(out)]) == mod.DONE  # fmt: skip
    r = json.loads(out.read_text())
    assert r["world"] == "synth" and r["cells"] and "calibration" in r
    assert out.with_suffix(".md").read_text().startswith("# Fusion diagnostics: synth")
    assert "fusion diagnostics [synth]" in capsys.readouterr().out
