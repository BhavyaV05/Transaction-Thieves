"""Per-account cap: invariants under random streams, the sweep tool, repeat statistics."""

import importlib.util
import random
from pathlib import Path

import pytest

from ftm.backtest.synth import make_world
from ftm.privacy.accountant import BudgetExceeded, PrivacyAccountant
from ftm.privacy.cap_sweep import repeat_stats, simulate_cap, subject_stream

STREAM = [("a1", [(2, 7)]), ("a2", [(2, 7)]), ("a3", [(2, 8)]), ("a4", [(2, 7)]), ("a5", [(3, 7)])]


def test_repeat_stats_count_alerts_per_partner_account():
    s = repeat_stats(STREAM)
    assert s == {"alerts": 5, "accounts": 3, "max_alerts_per_account": 3, "accounts_in_2plus_alerts": 1,
                 "accounts_in_3plus_alerts": 1, "accounts_in_5plus_alerts": 0, "alerts_touching_a_repeat_account": 3}  # fmt: skip
    assert repeat_stats([])["accounts"] == 0


def test_the_cap_bites_only_on_repeat_accounts_and_only_when_low():
    rows = {r["cap"]: r for r in simulate_cap(STREAM, [2.0, 4.0, None], questions_per_alert=2, eps=1.0)}
    assert rows[None]["refused"] == 0 and rows[None]["max_account_eps"] == pytest.approx(6.0)  # off: tracked, never refused
    assert rows[4.0]["refused"] == 2 and rows[4.0]["max_account_eps"] == pytest.approx(4.0)  # account (2, 7): 3 alerts x 2 eps
    assert rows[2.0]["refused"] > rows[4.0]["refused"] and rows[2.0]["max_account_eps"] <= 2.0
    assert rows[4.0]["refusal_reasons"] == {"account": 2}


@pytest.mark.parametrize("seed", range(5))
def test_no_account_and_no_alert_ever_exceeds_its_budget(seed):
    rng = random.Random(seed)
    cap = rng.choice([1.0, 3.0, 8.0])
    acc = PrivacyAccountant(eps_per_alert=2.0, per_account_cap=cap)
    for _ in range(400):
        keys = [(rng.randint(1, 3), rng.randint(1, 12)) for _ in range(rng.choice([1, 1, 1, 4]))]  # set questions charge several
        try:
            acc.spend(f"A{rng.randint(0, 60)}", keys[0][0], rng.choice([0.25, 0.5, 1.0, 2.0]), keys)
        except BudgetExceeded:
            pass
    assert max(acc.account_spent.values()) <= cap + 1e-9
    assert max(acc.alert_spent.values()) <= 2.0 + 1e-9
    assert acc.report()["n_spends"] > 0 and sum(acc.refusals.values()) > 0


def test_refusals_fall_as_the_cap_rises():
    w = make_world(200, seed=1)
    rng = random.Random(0)
    stream = [(aid, [(2, rng.randint(1, 15))]) for aid, _ in subject_stream(w.alerts)]  # accounts now repeat a lot
    shares = [r["refused_share"] for r in simulate_cap(stream, [1.0, 2.0, 4.0, 8.0, 16.0, None], 2, 1.0)]
    assert shares == sorted(shares, reverse=True) and shares[0] > 0.5 and shares[-1] == 0.0


def test_subject_stream_uses_the_partner_side_account_in_time_order():
    w = make_world(30, seed=2)
    s = subject_stream(w.alerts)
    by = {a.alert_id: a.binding for a in w.alerts}
    assert len(s) == 30
    assert all(keys[0] == (by[aid].receiver_bank, by[aid].receiver) for aid, keys in s)
    assert [by[aid].t0 for aid, _ in s] == sorted(by[aid].t0 for aid, _ in s)


def test_cli_synthetic():
    spec = importlib.util.spec_from_file_location("cs", Path(__file__).resolve().parents[1] / "scripts" / "cap_sweep.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    out = cli.main(["--synthetic", "120", "--caps", "2", "8"])
    assert out["repeats"]["alerts"] == 120 and [r["cap"] for r in out["caps"]] == [2.0, 8.0, None]
