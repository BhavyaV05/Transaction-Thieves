"""Week 9: the learned policy, the ablation runner and its config, the G4 table."""

import importlib.util
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import yaml

from ftm.banks.partner import Question
from ftm.orchestrator.ablations import (
    CONFIG,
    format_table,
    load_spec,
    run_ablations,
    synthetic_env,
)
from ftm.orchestrator.policy_data import collect_policy_data
from ftm.orchestrator.runner import ALL_SYSTEMS, build_planner
from ftm.planners.learned import (
    EIG_COL,
    FEATURES,
    Featurizer,
    LearnedPolicy,
    LearnedPolicyPlanner,
    split_val,
)
from ftm.planners.pool import Candidate, QuestionPool
from ftm.planners.session import PlanSession
from ftm.types import AlertContext, Limits

ROOT = Path(__file__).resolve().parents[1]


def cand(lr, p_clean, p_launder, eps=1.0, fam="f", prog="p(ALERT.receiver)"):
    cells = [f"c{i}" for i in range(len(lr))]
    return Candidate(prog, eps, fam, "kary", "receiver", dict(zip(cells, lr, strict=True)), (),
                     {c: (a, b) for c, a, b in zip(cells, p_clean, p_launder, strict=True)}, 1)  # fmt: skip


def session(p0=0.3):
    ctx = AlertContext("A", "1", datetime(2022, 9, 5).isoformat(), "val", p0,
                       {"direction": "out", "counterparty_bank": 2, "amount": 4321.0})  # fmt: skip
    return PlanSession(ctx, None, Limits())


@pytest.fixture(scope="module")
def env():
    return synthetic_env(300, 4)


class StubPolicy:
    """Predicts a fixed gain per program, so the planner's ranking and stop rule can be tested without LightGBM."""

    def __init__(self, gains):
        self.featurizer = Featurizer(["f", "g"])
        self.gains = gains

    def predict(self, X):
        raise AssertionError("the stub is keyed by program, see choose()")


def test_a_feature_row_has_one_number_per_feature_and_the_amount_is_coarse():
    fz = Featurizer(["f"])
    s = session()
    row = fz.row(s, cand([0.5, 2.0], [0.7, 0.3], [0.3, 0.7]))
    assert len(row) == len(FEATURES) == 15 and all(isinstance(x, float) for x in row)
    assert row[FEATURES.index("direction_out")] == 1.0
    assert row[FEATURES.index("log_amount")] == 3.5  # log10(4322) = 3.64, in half-decades
    other = replace(s.alert, summary={**s.alert.summary, "amount": 4400.0})
    s2 = PlanSession(other, None, Limits())
    assert fz.row(s2, cand([0.5, 2.0], [0.7, 0.3], [0.3, 0.7]))[FEATURES.index("log_amount")] == row[FEATURES.index("log_amount")]
    assert fz.row(s, cand([0.5, 2.0], [0.7, 0.3], [0.3, 0.7], fam="unseen"))[FEATURES.index("family")] == -1


def test_split_val_cuts_in_time_without_overlap(env):
    fit, ev = split_val(env.val, 0.5)
    assert len(fit) + len(ev) == len(env.val) and not {a.alert_id for a in fit} & {a.alert_id for a in ev}
    assert max(a.binding.t0 for a in fit) <= min(a.binding.t0 for a in ev)
    assert split_val(env.val, 0.5) == (fit, ev)


def test_policy_data_are_realised_gains_from_validation_alerts_only(env):
    fit, _ = split_val(env.val)
    data = collect_policy_data(env.banks, fit[:40], env.pools["realistic"], seed=1)
    assert data.X.shape == (len(data.y), len(FEATURES)) and len(data.alert_ids) == len(data.y) > 40
    assert np.isfinite(data.X).all() and np.isfinite(data.y).all()
    assert data.n_queries == len(data.y) and data.eps_cal > 0 and 0 < data.n_declined < data.n_queries
    assert np.all(data.y[data.X[:, EIG_COL] == 0] <= 1)  # a gain is a difference of two Brier scores
    again = collect_policy_data(env.banks, fit[:40], env.pools["realistic"], seed=1)
    assert np.array_equal(again.X, data.X) and np.array_equal(again.y, data.y)  # reproducible


def test_a_declined_question_is_a_zero_gain_row(env):
    from ftm.banks.partner import BankConfig

    fit, _ = split_val(env.val)
    declining = {b: bank._with_config(replace(bank.cfg, p_decline=1.0)) for b, bank in env.banks.items()}
    data = collect_policy_data(declining, fit[:10], env.pools["realistic"], seed=1)
    assert len(data.y) > 0 and (data.y == 0).all() and data.n_declined == data.n_queries
    assert isinstance(next(iter(declining.values())).cfg, BankConfig)


def test_with_config_keeps_the_ledger_and_changes_the_behaviour(env):
    fit, _ = split_val(env.val)
    a = fit[0]
    bank = env.banks[a.binding.partner_bank]
    prog = env.pools["realistic"].candidates[0]
    q = Question(prog.program, prog.eps, a.binding)
    other = bank._with_config(replace(bank.cfg, p_decline=1.0))
    assert other.bank == bank.bank and other._ledger is bank._ledger and other.seed == bank.seed
    assert type(other.handle(q, a.binding.t0)).__name__ == "Decline"


def test_planner_asks_the_largest_predicted_gain_per_eps_and_stops_when_none_helps():
    cheap = cand([0.5, 2.0], [0.7, 0.3], [0.3, 0.7], 0.5, "f", "a(ALERT.receiver)")
    dear = cand([0.5, 2.0], [0.7, 0.3], [0.3, 0.7], 2.0, "g", "b(ALERT.receiver)")
    pool = QuestionPool([cheap, dear])

    class Pol(StubPolicy):
        def predict(self, X):
            return np.array([0.04 if x[FEATURES.index("eps")] == 0.5 else 0.10 for x in X])  # per eps: 0.08 vs 0.05

    pl = LearnedPolicyPlanner(pool, policy=Pol({}))
    s = session()
    assert pl.choose(s, [dear, cheap]) is cheap
    assert pl.choose(s, []) is None

    class Nothing(StubPolicy):
        def predict(self, X):
            return np.zeros(len(X))

    assert LearnedPolicyPlanner(pool, policy=Nothing({})).choose(s, [cheap, dear]) is None
    with pytest.raises(ValueError, match="fitted policy"):
        LearnedPolicyPlanner(pool).choose(s, [cheap])


def test_build_planner_wires_the_ablation_switches():
    pool = QuestionPool([cand([0.5, 2.0], [0.7, 0.3], [0.3, 0.7])])
    assert "learned" in ALL_SYSTEMS
    with pytest.raises(ValueError, match="policy"):
        build_planner("learned", pool)
    p = build_planner("eig", pool, one_per_family=False, waves=(2, 1), threshold=0.3)
    assert p.one_per_family is False and p.waves == (2, 1) and p.threshold == 0.3
    assert build_planner("random", pool, seed=3).seed == 3


# --- LightGBM ----------------------------------------------------------------------------------------------------
def test_the_fitted_policy_starts_from_eig_and_survives_a_save_and_load(env, tmp_path):
    pytest.importorskip("lightgbm")
    fit, held = split_val(env.val)
    data = collect_policy_data(env.banks, fit, env.pools["realistic"], seed=2)
    pol = LearnedPolicy.fit(data, rounds=40, seed=2)
    assert pol.meta["rounds"] >= 1 and pol.meta["eps_cal"] == data.eps_cal
    pred = pol.predict(data.X)
    assert pred.shape == data.y.shape and np.isfinite(pred).all()
    # boosted residual over EIG: with this little signal the correction stays small
    assert np.mean(np.abs(pred - data.X[:, EIG_COL])) < 0.02
    path = pol.save(tmp_path / "p" / "policy.json")
    back = LearnedPolicy.load(path)
    assert np.allclose(back.predict(data.X), pred) and back.featurizer.families == pol.featurizer.families
    assert set(pol.importance()) == set(FEATURES)
    with pytest.raises(ValueError, match="no policy rows"):
        LearnedPolicy.fit(collect_policy_data(env.banks, [], env.pools["realistic"]))


# --- the ablation config and runner ---------------------------------------------------------------------------------
def test_the_config_parses_and_covers_table_5b():
    spec = load_spec()
    names = [a.name for a in spec.arms]
    assert spec.reference in names and len(set(names)) == len(names)
    assert {"random", "fixed_order", "eig", "learned_policy", "enumerator"} <= set(names)
    by = {a.name: a for a in spec.arms}
    assert by["enumerator"].pool_name == "enumerator" and by["eig"].pool_name == "realistic"
    assert by["eig_cap_8"].settings["cap"] == 8.0 and by["eig_batched_2_1"].settings["waves"] == [2, 1]
    assert by["eig"].settings["budget"] == 2.0 and by["eig_eps_4"].settings["budget"] == 4.0
    assert {a.group for a in spec.arms} >= {"planners", "budget", "cap", "async", "robustness", "fusion", "backtest"}


def test_the_config_rejects_unknown_settings_and_a_missing_reference(tmp_path):
    good = yaml.safe_load(CONFIG.read_text())
    bad = {**good, "arms": [*good["arms"], {"name": "x", "group": "g", "system": "eig", "bugdet": 1}]}
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="unknown settings"):
        load_spec(p)
    p.write_text(yaml.safe_dump({**good, "reference": "nope"}))
    with pytest.raises(ValueError, match="reference"):
        load_spec(p)
    p.write_text(yaml.safe_dump({**good, "arms": [good["arms"][0], good["arms"][0]]}))
    with pytest.raises(ValueError, match="unique"):
        load_spec(p)


@pytest.fixture(scope="module")
def rows(env):
    only = ["random", "eig", "learned_policy", "eig_eps_4", "eig_decline_50", "eig_timeout_20", "eig_fusion_shared",
            "eig_exact_label_lrs", "enumerator"]  # fmt: skip
    return {r["arm"]: r for r in run_ablations(env.banks, env.val, env.pools, load_spec(), seed=4, n_boot=100, only=only)}


def test_every_arm_is_scored_on_the_same_alerts_and_the_reference_has_no_difference(rows, env):
    ran = [r for r in rows.values() if not r["skipped"]]
    assert len({r["alerts"] for r in ran}) == 1 and ran[0]["alerts"] == len(split_val(env.val)[1])
    assert len({r["brier_p0"] for r in ran}) == 1  # same alerts, same starting score
    assert rows["eig"]["brier_diff_vs_ref"] == 0.0
    for r in ran:
        lo, hi = r["brier_gain_ci"]
        assert lo <= r["brier_gain"] <= hi and r["brier_gain"] == pytest.approx(r["brier_p0"] - r["brier_final"])
        assert r["mean_eps"] <= r["settings"]["budget"] + 1e-9


def test_the_switches_change_what_they_say_they_change(rows):
    assert rows["eig_decline_50"]["refused_share"] > rows["eig"]["refused_share"] + 0.2
    assert rows["eig_timeout_20"]["refused_share"] > rows["eig"]["refused_share"] + 0.1
    assert rows["eig_timeout_20"]["mean_sim_days"] > rows["eig"]["mean_sim_days"]  # a timeout costs five days
    assert rows["eig_eps_4"]["mean_eps"] > rows["eig"]["mean_eps"] and rows["eig_eps_4"]["settings"]["budget"] == 4.0
    assert rows["eig_fusion_shared"]["settings"]["fusion"] == "shared" and not rows["eig_fusion_shared"]["skipped"]


def test_missing_inputs_are_reported_as_skipped_never_dropped(rows):
    assert "exact_labels" in rows["eig_exact_label_lrs"]["skipped"]
    assert rows["enumerator"]["skipped"] is None  # the synthetic env has an enumerated table
    learned = rows["learned_policy"]
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        assert "lightgbm" in learned["skipped"]
    else:
        assert learned["skipped"] is None and learned["brier_final"] == pytest.approx(rows["eig"]["brier_final"], abs=0.02)


def test_a_missing_pool_skips_instead_of_crashing(env):
    out = run_ablations(env.banks, env.val, {"realistic": env.pools["realistic"]}, load_spec(), n_boot=50, only=["enumerator"])
    by = {r["arm"]: r for r in out}
    assert "enumerator" in by["enumerator"]["skipped"] and by["eig"]["skipped"] is None


def test_the_table_has_a_row_per_arm(rows):
    table = format_table(list(rows.values()), "eig").splitlines()
    assert len(table) == 2 + len(rows) and table[0].startswith("| Group |") and any("skipped" in ln for ln in table)


def test_ablation_runs_are_reproducible(env):
    a = run_ablations(env.banks, env.val, env.pools, load_spec(), seed=1, n_boot=50, only=["random"])
    b = run_ablations(env.banks, env.val, env.pools, load_spec(), seed=1, n_boot=50, only=["random"])
    keep = lambda r: {k: v for k, v in r.items() if k != "mean_seconds_per_alert"}  # noqa: E731
    assert [keep(x) for x in a] == [keep(x) for x in b]


# --- the scripts ---------------------------------------------------------------------------------------------------------
def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_ablations_cli_writes_the_table(tmp_path):
    cli = load_script("run_ablations")
    rows = cli.main(["--synthetic", "240", "--only", "random", "--n-boot", "50", "--out", str(tmp_path), "--seed", "3"])
    assert (tmp_path / "ablations.json").exists() and "| Group |" in (tmp_path / "ablations.md").read_text()
    assert {r["arm"] for r in rows} == {"random", "eig"}  # the reference always runs


def test_train_policy_cli_and_run_experiment_with_a_policy(tmp_path):
    pytest.importorskip("lightgbm")
    cli = load_script("train_policy")
    out = cli.main(["--synthetic", "300", "--out", str(tmp_path / "pol.json"), "--seed", "2", "--rounds", "30"])
    assert Path(out["policy"]).exists() and out["heldout"]["rows"] > 0 and out["world"] == "synthetic"
    exp = load_script("run_experiment")
    with pytest.raises(SystemExit):
        exp.main(["--system", "learned", "--synthetic", "100", "--out", str(tmp_path)])  # points at run_ablations
    with pytest.raises(SystemExit):
        exp.main(["--system", "learned", "--table", "x.jsonl", "--out", str(tmp_path)])  # needs --policy
