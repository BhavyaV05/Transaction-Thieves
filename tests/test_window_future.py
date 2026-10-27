"""T1's flag (docs/t1_handoffs.md, A4): `cross_currency_out(ALERT.sender, window=after(72))` looks at the 72 h after the
alert time. A live answer must not see the future: the ledger is cut at the answer time, `as_of + latency`.

Three checks, none of which needs the real data:
  1. a reply never depends on a row after its cutoff (full ledger vs the ledger cut at the cutoff give the same reply);
  2. a cross-currency transfer beyond the slowest possible answer is never seen, one right after the alert is seen exactly
     when the reply is slow enough;
  3. the same holds through the harness for a question asked late in the session (the clock moves the ask time).
"""

from dataclasses import replace
from datetime import datetime, timedelta

import polars as pl
import pytest

from ftm.banks.binding import AlertBinding
from ftm.banks.partner import BankConfig, PartnerBank, Question
from ftm.compliance.gate import Gate
from ftm.dsl.expert import expert_programs
from ftm.dsl.parser import gate_validator
from ftm.harness import Harness
from ftm.privacy.accountant import PrivacyAccountant
from ftm.types import AlertContext, Limits, Reply

T0 = datetime(2022, 9, 5, 12, 0)
BANK_OF = {0: 1, 1: 2, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1, 7: 1}
QUIET = BankConfig(p_decline=0.0, p_timeout=0.0, notes_enabled=False)
CROSS = "cross_currency_out(ALERT.sender, window=after(72))"
ALERT = AlertBinding("A1", home_bank=1, t0=T0, sender=0, receiver=1, amount=1000.0, currency="USD",
                     payers=frozenset({0, 6}), payees=frozenset({3, 4}))  # fmt: skip
MAX_LATENCY_DAYS = QUIET.tail_max_days


def H(h):
    return T0 + timedelta(hours=h)


def ledger(rows):
    """rows: (time, src, dst, amount, currency of the payment)."""
    return pl.DataFrame(
        {
            "timestamp": [r[0] for r in rows], "src_id": [r[1] for r in rows], "dst_id": [r[2] for r in rows],
            "from_bank": [BANK_OF[r[1]] for r in rows], "to_bank": [BANK_OF[r[2]] for r in rows],
            "amount_paid": [float(r[3]) for r in rows], "amount_received": [float(r[3]) for r in rows],
            "payment_currency": [r[4] for r in rows], "receiving_currency": [r[4] for r in rows],
            "payment_format": ["ACH"] * len(rows),
        },
        schema_overrides={"timestamp": pl.Datetime("us")},
    )  # fmt: skip


# bank 1 holds the alerted sender 0. Its history before the alert, the alerted transfer, then (all in euros) later transfers
# spread over the 72 h after the alert and beyond.
PAST = [(H(-30), 0, 1, 500, "USD"), (H(-10), 6, 1, 400, "USD"), (H(0), 0, 1, 1000, "USD")]
LATER = [(H(h), 0, 2, 100 + h, "Euro") for h in (1, 6, 12, 24, 36, 48, 60, 71)]


def cut(led: pl.DataFrame, cutoff: datetime) -> pl.DataFrame:
    return led.filter(pl.col("timestamp") <= cutoff)


def alert_with(i: int) -> AlertBinding:
    return replace(ALERT, alert_id=f"A{i}")


def test_a_reply_never_depends_on_rows_after_its_cutoff():
    full = ledger(PAST + LATER)
    b_full = PartnerBank(1, full, QUIET, 3)
    programs = [p["program"] for p in expert_programs("sender") if "after(" in p["program"]]
    assert CROSS in programs or any("cross_currency_out" in p for p in programs)
    n = 0
    for i in range(40):
        a = alert_with(i)
        for prog in programs:
            for eps in (0.5, 2.0):
                q = Question(prog, eps, a)
                reply, latency = b_full.respond(q, T0)
                assert isinstance(reply, Reply)
                seen = PartnerBank(1, cut(full, T0 + timedelta(days=latency)), QUIET, 3)
                reply_cut, latency_cut = seen.respond(q, T0)
                assert latency_cut == latency
                assert reply_cut.value == reply.value, (prog, eps, latency)
                n += 1
    assert n >= 40 * 2  # a few after-window programs, both budgets


def test_a_transfer_beyond_the_slowest_answer_is_never_seen():
    # one hour past the slowest possible reply, inside a window long enough to cover it
    far = ledger(PAST + [(T0 + timedelta(days=MAX_LATENCY_DAYS, hours=1), 0, 2, 50, "Euro")])
    prog = "cross_currency_out(ALERT.sender, window=after(168))"
    b = PartnerBank(1, far, QUIET, 0)
    for i in range(200):
        r = b.handle(Question(prog, float("inf"), alert_with(i)), T0)
        assert isinstance(r, Reply) and r.value is False


def test_a_transfer_right_after_the_alert_is_seen_exactly_when_the_reply_is_slow_enough():
    b = PartnerBank(1, ledger(PAST + [(H(2), 0, 2, 70, "Euro")]), QUIET, 0)
    seen = unseen = 0
    for i in range(300):
        reply, latency = b.respond(Question(CROSS, float("inf"), alert_with(i)), T0)
        assert reply.value is (latency >= 2 / 24)  # the transfer is 2 h after the alert; the cutoff is T0 + latency
        seen += reply.value
        unseen += not reply.value
    assert seen and unseen  # both cases occur, so the check above bites


def _harness(led: pl.DataFrame):
    bank = PartnerBank(1, led, QUIET, 0)
    acc = PrivacyAccountant(eps_per_alert=10.0)
    gate = Gate.from_config(validator=gate_validator, accountant=acc)
    gate.eps_levels = (0.5, 1.0, 2.0)
    # the alert belongs to bank 3, which asks bank 1 (a bank cannot ask itself); bank 1 holds the alerted sender
    return Harness({1: bank}, gate, acc, {a.alert_id: replace(a, home_bank=3) for a in map(alert_with, range(30))})


def alert_ctx(i: int) -> AlertContext:
    return AlertContext(alert_id=f"A{i}", home_bank="3", as_of=T0.isoformat(), split="val", p0=0.1)


def test_through_the_harness_a_late_question_sees_only_what_exists_at_its_own_answer_time():
    full = ledger(PAST + [(H(30), 0, 2, 130, "Euro")])  # visible only to a late answer: first + second reply over 30 h
    values = set()
    for i in range(30):
        tools = _harness(full).toolbox()
        tools.begin(alert_ctx(i), Limits(eps_per_alert=10.0, max_sim_time_days=7.0))
        # a first question moves the clock; the second is asked at T0 + elapsed and answered at T0 + elapsed + latency
        tools.ask_bank(1, "n_tx(ALERT.sender, dir=out, window=before(24))", 0.5)
        asked_at = tools.elapsed_days
        r2 = tools.ask_bank(1, CROSS, 0.5)
        log = tools.end()
        if not isinstance(r2, Reply):
            continue
        cutoff = T0 + timedelta(days=asked_at + r2.latency_days)
        # the same session on the ledger cut at that instant gives the same reply
        again = _harness(cut(full, cutoff)).toolbox()
        again.begin(alert_ctx(i), Limits(eps_per_alert=10.0, max_sim_time_days=7.0))
        again.ask_bank(1, "n_tx(ALERT.sender, dir=out, window=before(24))", 0.5)
        r2b = again.ask_bank(1, CROSS, 0.5)
        again.end()
        assert isinstance(r2b, Reply) and r2b.value == r2.value and r2b.latency_days == r2.latency_days, (i, log)
        values.add(r2.value)
    assert values == {True, False}  # both the early (blind) and the late (seeing) case occurred, so the comparison bites


@pytest.mark.parametrize("hours", [1, 12, 71])
def test_the_alerted_transfer_itself_is_not_in_the_after_window(hours):
    # after(h) = (t0, t0+h]: a cross-currency transfer AT t0 does not count, one just after does (once the reply is slow enough)
    at_t0 = ledger(PAST + [(H(0), 0, 2, 70, "Euro")])
    b = PartnerBank(1, at_t0, QUIET, 0)
    for i in range(50):
        assert b.handle(Question(f"cross_currency_out(ALERT.sender, window=after({hours}))", float("inf"), alert_with(i)),
                        T0).value is False  # fmt: skip
