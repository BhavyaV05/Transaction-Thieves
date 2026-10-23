"""C5: DP mechanisms (distribution, ceilings) and the privacy accountant (per-alert, per-account, eps_cal)."""

import math

import numpy as np
import pytest
from scipy import stats

from ftm.compliance.gate import Gate
from ftm.dsl.parser import gate_validator, parse
from ftm.privacy.accountant import BudgetExceeded, NullAccountant, PrivacyAccountant, charge_keys
from ftm.privacy.mechanisms import (
    geometric_pmf,
    kary_randomized_response,
    likelihood,
    log_lr,
    lr_ceiling,
    noisy_count,
    randomize,
    randomized_response,
    two_sided_geometric,
)

N = 10_000  # draws per distribution test (plan.md 6.4: "KS test over 10k draws")


def rng(seed=0):
    return np.random.default_rng(seed)


# --- mechanisms -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("eps", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_randomized_response_keep_rate(eps):
    r = rng(1)
    kept = sum(randomized_response(True, eps, r) for _ in range(N))
    p = math.exp(eps) / (1 + math.exp(eps))
    assert stats.binomtest(kept, N, p).pvalue > 1e-3


@pytest.mark.parametrize("eps,k", [(0.5, 3), (1.0, 4), (2.0, 5)])
def test_kary_rr_matches_its_likelihood(eps, k):
    r = rng(2)
    true = 1
    counts = np.bincount([kary_randomized_response(true, k, eps, r) for _ in range(N)], minlength=k)
    expected = np.array([likelihood("kary", eps, 1, true, o, k) for o in range(k)]) * N
    assert stats.chisquare(counts, expected).pvalue > 1e-3


@pytest.mark.parametrize("eps,sens", [(0.5, 1), (1.0, 1), (2.0, 1), (1.0, 2), (2.0, 3)])
def test_two_sided_geometric_matches_pmf(eps, sens):
    r = rng(3)
    draws = np.array([two_sided_geometric(eps, sens, r) for _ in range(N)])
    lim = 8
    obs = [int((draws == z).sum()) for z in range(-lim, lim + 1)]
    obs = [int((draws < -lim).sum())] + obs + [int((draws > lim).sum())]
    p = [geometric_pmf(z, eps, sens) for z in range(-lim, lim + 1)]
    tail = (1 - sum(p)) / 2
    exp = np.array([tail] + p + [tail]) * N
    keep = exp >= 5  # chi-square needs expected counts of at least 5; merge the rest into the nearest kept bin
    obs, exp = np.array(obs, float), np.array(exp)
    o2, e2 = list(obs[keep]), list(exp[keep])
    o2[-1] += obs[~keep].sum() * 0.5
    e2[-1] += exp[~keep].sum() * 0.5
    o2[0] += obs[~keep].sum() * 0.5
    e2[0] += exp[~keep].sum() * 0.5
    e2 = np.array(e2) * (sum(o2) / sum(e2))
    assert stats.chisquare(o2, e2).pvalue > 1e-3
    # and the Kolmogorov-Smirnov distance to the exact CDF is within the DKW bound
    zs = np.arange(-60, 61)
    cdf = np.cumsum([geometric_pmf(int(z), eps, sens) for z in zs])
    emp = np.array([(draws <= z).mean() for z in zs])
    assert np.abs(emp - cdf).max() < math.sqrt(math.log(2 / 1e-3) / (2 * N))


def test_pmf_sums_to_one_and_noise_is_unbiased():
    assert sum(geometric_pmf(z, 1.0, 2) for z in range(-400, 401)) == pytest.approx(1.0)
    r = rng(4)
    assert np.mean([noisy_count(10, 1.0, 1, r) for _ in range(N)]) == pytest.approx(10, abs=0.2)


@pytest.mark.parametrize("kind,arity,sens", [("bool", None, 1), ("kary", 4, 1), ("count", None, 1), ("count", None, 3)])
@pytest.mark.parametrize("eps", [0.25, 1.0, 2.0])
def test_neighbouring_answers_never_exceed_the_dp_ceiling(kind, arity, sens, eps):
    # true answers one sensitivity apart: every output's likelihood ratio is at most e^eps
    a, b = (True, False) if kind == "bool" else (1, 0) if kind == "kary" else (10, 10 + sens)
    outputs = [True, False] if kind == "bool" else range(arity) if kind == "kary" else range(-60, 80)
    worst = max(abs(log_lr(kind, eps, sens, a, b, o, arity)) for o in outputs)
    assert worst <= eps + 1e-9
    assert lr_ceiling(kind, eps, sens, diff=sens) == pytest.approx(math.exp(eps))


def test_count_ceiling_grows_with_the_gap_between_true_answers():
    assert lr_ceiling("count", 1.0, 1, diff=3) == pytest.approx(math.exp(3.0))
    assert lr_ceiling("count", 2.0, 2, diff=1) == pytest.approx(math.exp(1.0))
    worst = max(log_lr("count", 1.0, 1, 13, 10, o) for o in range(-40, 60))
    assert worst == pytest.approx(3.0)


@pytest.mark.parametrize("eps", [0.5, 2.0])
def test_empirical_likelihood_ratio_stays_under_the_ceiling(eps):
    r = rng(5)
    n = 40_000
    yes_if_true = np.mean([randomized_response(True, eps, r) for _ in range(n)])
    yes_if_false = np.mean([randomized_response(False, eps, r) for _ in range(n)])
    se = math.sqrt(yes_if_false * (1 - yes_if_false) / n) / yes_if_false
    assert yes_if_true / yes_if_false <= math.exp(eps) * (1 + 3 * se)


def test_no_noise_arm_is_exact():
    r = rng(6)
    assert randomized_response(True, math.inf, r) is True
    assert kary_randomized_response(2, 4, None, r) == 2
    assert noisy_count(7, math.inf, 1, r) == 7
    assert randomize("count", 7, math.inf, 3, r) == 7
    assert lr_ceiling("bool", math.inf) == math.inf


def test_bad_inputs():
    r = rng(7)
    with pytest.raises(ValueError):
        randomized_response(True, -1.0, r)
    with pytest.raises(ValueError):
        kary_randomized_response(5, 4, 1.0, r)
    with pytest.raises(ValueError):
        randomize("kary", 1, 1.0, 1, r)  # no arity


# --- accountant -------------------------------------------------------------------------------------------------
def test_per_alert_budget_is_enforced_across_banks():
    acc = PrivacyAccountant(eps_per_alert=2.0)
    assert acc.can_spend("A1", 5, 1.0)
    acc.spend("A1", 5, 1.0)
    acc.spend("A1", 6, 0.5)  # another bank, same alert: same pot
    assert acc.remaining("A1") == pytest.approx(0.5)
    assert not acc.can_spend("A1", 7, 1.0)
    with pytest.raises(BudgetExceeded):
        acc.spend("A1", 7, 1.0)
    assert acc.can_spend("A1", 7, 0.5)
    assert acc.can_spend("A2", 5, 2.0)  # other alerts are untouched
    assert acc.report()["refusals"]["alert"] == 2


def test_per_account_cap_holds_across_alerts_and_asking_banks():
    acc = PrivacyAccountant(eps_per_alert=2.0, per_account_cap=3.0)
    acct = [(5, 77)]
    acc.spend("A1", 5, 2.0, acct)  # asked by home bank X
    assert acc.can_spend("A2", 5, 1.0, acct)
    acc.spend("A2", 5, 1.0, acct)  # a different alert, maybe a different asking bank: same account
    assert not acc.can_spend("A3", 5, 0.25, acct)
    with pytest.raises(BudgetExceeded):
        acc.spend("A3", 5, 0.25, acct)
    assert acc.can_spend("A3", 5, 2.0, [(5, 78)])  # another account at the same bank is fine
    assert acc.can_spend("A3", 6, 2.0, [(6, 77)])  # the same node number at another bank is another account
    assert acc.report()["refusals"]["account"] == 2
    assert acc.report()["max_account_eps"] == pytest.approx(3.0)


def test_a_refused_spend_changes_nothing():
    acc = PrivacyAccountant(eps_per_alert=2.0, per_account_cap=1.0)
    with pytest.raises(BudgetExceeded):
        acc.spend("A1", 5, 1.5, [(5, 1), (5, 2)])
    assert acc.alert_spent["A1"] == 0 and not acc.account_spent


def test_unlimited_cap_still_tracks():
    acc = PrivacyAccountant(eps_per_alert=2.0, per_account_cap=None)
    for i in range(10):
        acc.spend(f"A{i}", 5, 2.0, [(5, 1)])
    assert acc.report()["max_account_eps"] == pytest.approx(20.0)


def test_calibration_is_a_separate_reported_ledger():
    acc = PrivacyAccountant(eps_per_alert=2.0, per_account_cap=2.0, eps_cal=5.0)
    acc.spend_cal(5, 2.0, [(5, 1)])
    acc.spend_cal(5, 2.0, [(5, 1)])
    assert acc.can_spend("A1", 5, 2.0, [(5, 1)])  # live budgets untouched by back-testing
    assert acc.report()["eps_cal_spent"] == pytest.approx(4.0)
    with pytest.raises(BudgetExceeded):
        acc.spend_cal(5, 2.0)
    assert acc.report()["refusals"]["cal"] == 1


def test_state_round_trip_and_config():
    acc = PrivacyAccountant(2.0, 8.0, None)
    acc.spend("A1", 5, 1.0, [(5, 1), (5, 2)])
    acc.spend_cal(5, 0.5, [(5, 1)])
    back = PrivacyAccountant.from_dict(acc.to_dict())
    assert back.report() == acc.report()
    assert back.account_spent[(5, 2)] == 1.0
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load(Path("configs/privacy/default.yaml").read_text())
    assert PrivacyAccountant.from_config(cfg).per_account_cap == cfg["per_account_cap"] == 8.0


def test_only_finite_positive_eps_is_priced():
    acc = PrivacyAccountant()
    for bad in (0.0, -1.0, math.inf):
        with pytest.raises(ValueError):
            acc.can_spend("A1", 5, bad)
    assert NullAccountant().can_spend("A1", 5, math.inf)


class _B:
    sender, receiver = 10, 11
    payers, payees = frozenset({1, 2, 3}), frozenset({4})

    def slot(self, name):
        return getattr(self, name)


def test_charge_keys_follow_what_the_question_can_see():
    b = _B()
    assert charge_keys(parse("n_tx(ALERT.receiver, dir=in, window=before(72))"), b, 5) == [(5, 11)]
    assert charge_keys(parse("overlap(ALERT.sender, dir=out, window=after(24), among=ALERT.payers)"), b, 5) == [(5, 10)]
    assert charge_keys(parse("at_least(n_tx(ALERT.sender, dir=in, window=before(72)), k=2)"), b, 7) == [(7, 10)]
    assert charge_keys(parse("set_contacts(ALERT.payers, dir=out, window=before(72))"), b, 5) == [(5, 1), (5, 2), (5, 3)]


def test_accountant_plugs_into_the_compliance_gate():
    acc = PrivacyAccountant(eps_per_alert=2.0)
    gate = Gate.from_config(validator=gate_validator, accountant=acc)
    prog = "n_tx(ALERT.receiver, dir=in, window=before(72))"
    kw = dict(alert_id="A1", home_bank=1, bank=5, program=prog, partners=[5, 6])
    assert gate.check("ask_bank", eps=1.0, **kw).decision == "allowed"
    acc.spend("A1", 5, 1.0)
    acc.spend("A1", 5, 1.0)
    assert gate.check("ask_bank", eps=0.25, **kw).decision == "over_budget"
    assert gate.check("ask_bank", eps=0.25, **{**kw, "alert_id": "A2"}).decision == "allowed"
    assert gate.check("ask_bank", eps=0.7, **{**kw, "alert_id": "A2"}).decision == "blocked"  # not an eps level
    assert gate.check("ask_bank", eps=1.0, **{**kw, "program": "nope(ALERT.receiver)"}).decision == "blocked"
