"""C4 executor: hand-computed cases plus a brute-force cross-check on 50 random toy graphs."""

import random
from datetime import datetime, timedelta

import polars as pl
import pytest

from ftm.dsl.executor import ExecContext, ExecError, execute
from ftm.dsl.expert import expert_programs
from ftm.dsl.parser import Node, parse
from ftm.dsl.spec import Window

T0 = datetime(2022, 9, 5, 12, 0)
BANK, HOME = 2, 1
W72 = Window("before", 72)


def H(hours):
    return T0 + timedelta(hours=hours)


def ledger(rows):
    """rows: (timestamp, src, dst, amount[, currency[, format]]). Account -> bank map is fixed below."""
    banks = {0: 1, 1: 2, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1, 7: 1}
    df = pl.DataFrame(
        {
            "timestamp": [r[0] for r in rows],
            "src_id": [r[1] for r in rows],
            "dst_id": [r[2] for r in rows],
            "from_bank": [banks[r[1]] for r in rows],
            "to_bank": [banks[r[2]] for r in rows],
            "amount_paid": [float(r[3]) for r in rows],
            "amount_received": [float(r[3]) for r in rows],
            "payment_currency": [r[4] if len(r) > 4 else "USD" for r in rows],
            "receiving_currency": [r[4] if len(r) > 4 else "USD" for r in rows],
            "payment_format": [r[5] if len(r) > 5 else "ACH" for r in rows],
        },
        schema_overrides={"timestamp": pl.Datetime("us")},
    )
    return df


def ctx(**kw):
    base = dict(bank=BANK, t0=T0, cutoff=H(120), alert_amount=1000.0, alert_currency="USD", home_bank=HOME,
                sender=0, receiver=1, payers=frozenset({0, 6}), payees=frozenset({3, 4}))
    base.update(kw)
    return ExecContext(**base)


def run(text, df, **kw):
    return execute(parse(text), df, ctx(**kw))


def test_hand_computed_counts_and_windows():
    df = ledger([
        (H(-80), 0, 1, 500),  # outside before(72)
        (H(-10), 0, 1, 400),
        (H(-1), 6, 1, 2500),
        (H(0), 0, 1, 1000),   # the alerted transfer: inside before(72) and around
        (H(2), 1, 3, 900),    # SUBJECT (account 1) pays account 3 at bank 3
        (H(5), 1, 4, 100),
        (H(30), 1, 3, 50),    # outside after(24), inside after(48)
    ])
    assert run("n_counterparties(ALERT.receiver, dir=in, window=before(72))", df) == 2  # accounts 0 and 6
    assert run("n_tx(ALERT.receiver, dir=in, window=before(72))", df) == 3
    assert run("n_tx(ALERT.receiver, dir=in, window=before(72), max_amount=1000)", df) == 1  # 400 only
    assert run("n_counterparties(ALERT.receiver, dir=out, window=after(24))", df) == 2
    assert run("n_counterparties(ALERT.receiver, dir=out, window=after(48))", df) == 2
    assert run("n_banks(ALERT.receiver, dir=in, window=before(72))", df) == 1  # both senders are at bank 1
    assert run("at_least(n_counterparties(ALERT.receiver, dir=in, window=before(72)), k=2)", df) is True
    assert run("at_least(n_counterparties(ALERT.receiver, dir=in, window=before(72)), k=3)", df) is False
    # outflow 1000 in 24 h vs inflow 3900 in 72 h: not 80%, but 25% is
    assert run("passed_on(ALERT.receiver, window=after(24), min_frac=0.8)", df) is False
    assert run("passed_on(ALERT.receiver, window=after(24), min_frac=0.25)", df) is True
    assert run("overlap(ALERT.receiver, dir=out, window=after(24), among=ALERT.payees)", df) == 2
    assert run("overlap(ALERT.receiver, dir=in, window=before(72), among=ALERT.payers)", df) == 2
    assert run("returns_home(ALERT.receiver, window=after(72))", df) is False
    assert run("time_to_forward(ALERT.receiver)", df) == 1  # 900 >= half of 1000 at 2 h: the 1-6 h bucket
    assert run("inflow_bucket(ALERT.receiver)", df) == 2  # 3900 / 1000 = 3.9x: the 3-10x bucket
    assert run("n_round(ALERT.receiver, window=before(72), unit=1000)", df) == 1  # 1000 only
    assert run("cross_currency_out(ALERT.receiver, window=after(24))", df) is False


def test_dormant_and_new_account_both_ways():
    quiet = ledger([(H(-1), 0, 1, 500), (H(2), 1, 3, 400)])  # nothing before the 72 h window
    busy = ledger([(H(-24 * 5), 0, 1, 80), (H(-1), 0, 1, 500), (H(2), 1, 3, 400)])
    assert run("dormant(ALERT.receiver, days=7)", quiet) is True
    assert run("dormant(ALERT.receiver, days=7)", busy) is False
    assert run("new_account(ALERT.receiver, days=2)", quiet) is True
    assert run("new_account(ALERT.receiver, days=2)", busy) is False
    assert run("new_account(ALERT.receiver, days=7)", busy) is True  # first transfer 5 days before the alert


def test_top_outflow_bank_index_and_none():
    df = ledger([(H(1), 1, 3, 400), (H(2), 1, 0, 900), (H(3), 1, 4, 100)])  # bank 3 gets 500, bank 1 gets 900
    prog = "top_outflow_bank(ALERT.receiver, banks=[3, 1, 9], window=after(24))"
    assert run(prog, df) == 0  # the list is sorted to [1, 3, 9]; bank 1 (900) beats bank 3 (500): position 0
    assert run("top_outflow_bank(ALERT.receiver, banks=[9, 8], window=after(24))", df) == 2  # none of them


def test_cutoff_hides_the_future():
    df = ledger([(H(2), 1, 3, 900), (H(30), 1, 4, 50)])
    prog = "n_counterparties(ALERT.receiver, dir=out, window=after(72))"
    assert run(prog, df, cutoff=H(120)) == 2
    assert run(prog, df, cutoff=H(12)) == 1  # a reply after 12 h sees 12 h of the 72 h window


def test_a_bank_that_does_not_hold_subject_answers_zero():
    df = ledger([(H(-1), 0, 1, 500), (H(1), 1, 3, 500)])
    prog = "n_counterparties(ALERT.receiver, dir=any, window=around(72))"
    assert run(prog, df) == 2  # bank 2 holds account 1: sees accounts 0 and 3
    assert run(prog, df, bank=3) == 0  # bank 3 sees the transfer 1 -> 3 but does not hold account 1
    assert run(prog, df, receiver=7) == 0  # account 7 is at bank 1


def test_chain_length_is_time_respecting_and_capped():
    df = ledger([(H(1), 1, 2, 10), (H(2), 2, 5, 10), (H(3), 5, 1, 10), (H(0.5), 5, 2, 10)])
    assert run("chain_length(ALERT.receiver, window=after(24), max_hops=6)", df) == 3  # 1 -> 2 -> 5 -> 1
    assert run("chain_length(ALERT.receiver, window=after(24), max_hops=2)", df) == 2


def test_missing_entity_map_is_an_error():
    df = ledger([(H(1), 1, 2, 10)])
    with pytest.raises(ExecError):
        run("n_same_entity(ALERT.receiver, window=around(24))", df)


def test_executor_returns_scalars_only():
    df = ledger([(H(1), 1, 2, 10)])
    for p in expert_programs()[:10]:
        assert isinstance(execute(parse(p["program"]), df, ctx()), (bool, int))


# ---------------------------------------------------------------------------------------------------------------
# Brute force: a second, deliberately naive implementation over plain Python rows.
def _bf_events(rows, subj, c):
    ev = []
    for r in rows:
        if r["src"] == subj and r["fb"] == c.bank:
            ev.append(dict(ts=r["ts"], dir="out", cp=r["dst"], cpb=r["tb"], amt=r["paid"], cur=r["pcur"], fmt=r["fmt"]))
        if r["dst"] == subj and r["tb"] == c.bank:
            ev.append(dict(ts=r["ts"], dir="in", cp=r["src"], cpb=r["fb"], amt=r["recv"], cur=r["rcur"], fmt=r["fmt"]))
    return ev


def _bf_in(ts, w, c):
    h = timedelta(hours=w.hours)
    if w.mode == "before":
        return c.t0 - h <= ts <= c.t0
    if w.mode == "after":
        return c.t0 < ts <= c.t0 + h
    return c.t0 - h / 2 <= ts <= c.t0 + h / 2


def _bf_win(ev, w, c, d="any"):
    return [e for e in ev if _bf_in(e["ts"], w, c) and (d == "any" or e["dir"] == d)]


def _bf_chain(rows, node, t, left, first, w, c):
    if left == 0:
        return 0
    best = 0
    for r in rows:
        if r["src"] == node and r["fb"] == c.bank and r["tb"] == c.bank and _bf_in(r["ts"], w, c) and r["ts"] >= t:
            if first and not r["ts"] > c.t0:
                continue
            best = max(best, 1 + _bf_chain(rows, r["dst"], r["ts"], left - 1, False, w, c))
    return best


def brute(n: Node, rows, c: ExecContext):
    rows = [r for r in rows if r["ts"] <= c.cutoff]
    a = n.arg
    if n.name == "at_least":
        return brute(n.subject, rows, c) >= a("k")
    if n.name == "set_contacts":
        got = set()
        for r in rows:
            if not _bf_in(r["ts"], a("window"), c):
                continue
            if a("dir") == "in" and r["fb"] == c.bank and r["dst"] in c.slot(n.subject):
                got.add(r["dst"])
            if a("dir") == "out" and r["tb"] == c.bank and r["src"] in c.slot(n.subject):
                got.add(r["src"])
        return len(got)
    subj = c.slot(n.subject)
    ev = _bf_events(rows, subj, c)
    w = a("window") if any(k == "window" for k, _ in n.args) else None
    name = n.name
    if name == "n_counterparties":
        return len({e["cp"] for e in _bf_win(ev, w, c, a("dir"))})
    if name == "n_banks":
        return len({e["cpb"] for e in _bf_win(ev, w, c, a("dir"))})
    if name == "n_tx":
        return sum(1 for e in _bf_win(ev, w, c, a("dir")) if a("min_amount") <= e["amt"] < a("max_amount"))
    if name == "n_round":
        u = a("unit")
        return sum(1 for e in _bf_win(ev, w, c) if e["amt"] > 0 and abs(e["amt"] / u - round(e["amt"] / u)) * u < 0.005)
    if name == "n_format":
        return sum(1 for e in _bf_win(ev, w, c) if e["fmt"].lower().replace(" ", "_") == a("format"))
    if name == "n_repeated":
        win = _bf_win(ev, w, c)
        return sum(
            1 for i, x in enumerate(win)
            if any(j != i and abs(x["amt"] - y["amt"]) <= 0.01 * max(x["amt"], y["amt"]) for j, y in enumerate(win))
        )
    if name == "n_same_entity":
        me = c.entity_of.get(subj)
        return sum(1 for e in _bf_win(ev, w, c) if e["cp"] != subj and c.entity_of.get(e["cp"]) == me)
    if name == "n_new_counterparties":
        before = {e["cp"] for e in ev if e["dir"] == "out" and e["ts"] <= c.t0}
        return len({e["cp"] for e in _bf_win(ev, w, c, "out")} - before)
    if name == "chain_length":
        return _bf_chain(rows, subj, c.t0, a("max_hops"), True, w, c)
    if name == "passed_on":
        rec = sum(e["amt"] for e in _bf_win(ev, W72, c, "in"))
        sent = sum(e["amt"] for e in _bf_win(ev, w, c, "out"))
        return rec > 0 and sent >= a("min_frac") * rec - 1e-9
    if name == "returns_home":
        return any(e["cpb"] == c.home_bank for e in _bf_win(ev, w, c, "out"))
    if name == "matching_amount_out":
        out = [e for e in ev if e["dir"] == "out" and c.t0 < e["ts"] <= c.t0 + timedelta(hours=24)]
        return any(abs(e["amt"] - c.alert_amount) <= a("tol") * c.alert_amount + 1e-9 for e in out)
    if name == "velocity_balanced":
        win = _bf_win(ev, w, c)
        i = sum(e["amt"] for e in win if e["dir"] == "in")
        o = sum(e["amt"] for e in win if e["dir"] == "out")
        return i >= c.alert_amount and 0.9 * i - 1e-9 <= o <= 1.1 * i + 1e-9
    if name == "cross_currency_out":
        return any(e["cur"] != c.alert_currency for e in _bf_win(ev, w, c, "out"))
    if name == "dormant":
        hi = c.t0 - timedelta(hours=72)
        return not any(hi - timedelta(days=a("days")) <= e["ts"] < hi for e in ev)
    if name == "new_account":
        return bool(ev) and min(e["ts"] for e in ev) >= c.t0 - timedelta(days=a("days"))
    if name == "volume_spike":
        day = timedelta(hours=24)
        last = sum(e["amt"] for e in ev if e["dir"] == "in" and c.t0 - day <= e["ts"] <= c.t0)
        prev = sum(e["amt"] for e in ev if e["dir"] == "in" and c.t0 - day - timedelta(days=7) <= e["ts"] < c.t0 - day)
        return last > 0 and last >= a("mult") * prev / 7 - 1e-9
    if name == "payees_pay_each_other":
        payees = {e["cp"] for e in _bf_win(ev, w, c, "out")}
        return any(
            r["src"] != r["dst"] and r["src"] in payees and r["dst"] in payees and _bf_in(r["ts"], w, c) for r in rows
        )
    if name == "top_outflow_bank":
        tot = [sum(e["amt"] for e in _bf_win(ev, w, c, "out") if e["cpb"] == b) for b in a("banks")]
        return tot.index(max(tot)) if max(tot) > 0 else len(tot)
    if name == "time_to_forward":
        acc = 0.0
        for e in sorted((e for e in ev if e["dir"] == "out" and e["ts"] > c.t0), key=lambda e: e["ts"]):
            acc += e["amt"]
            if acc >= 0.5 * c.alert_amount - 1e-9:
                h = (e["ts"] - c.t0).total_seconds() / 3600
                return 0 if h < 1 else 1 if h < 6 else 2 if h < 24 else 3
        return 3
    if name == "inflow_bucket":
        r = sum(e["amt"] for e in _bf_win(ev, W72, c, "in")) / c.alert_amount
        return 0 if r < 1 else 1 if r < 3 else 2 if r < 10 else 3
    if name == "overlap":
        return len({e["cp"] for e in _bf_win(ev, w, c, a("dir"))} & set(c.slot(a("among"))))
    raise AssertionError(name)


EXTRA = [
    "n_tx(ALERT.{s}, dir=out, window=around(96), min_amount=200, max_amount=5000)",
    "n_counterparties(ALERT.{s}, dir=any, window=around(48))",
    "n_banks(ALERT.{s}, dir=any, window=before(200))",
    "chain_length(ALERT.{s}, window=after(120), max_hops=5)",
    "top_outflow_bank(ALERT.{s}, banks=[1, 3], window=around(100))",
    "set_contacts(ALERT.payers, dir=in, window=before(72))",
    "set_contacts(ALERT.payees, dir=out, window=around(120))",
    "set_contacts(ALERT.payers, dir=out, window=after(48))",
    "at_least(overlap(ALERT.{s}, dir=in, window=before(72), among=ALERT.payers), k=2)",
    "at_least(set_contacts(ALERT.payers, dir=in, window=around(200)), k=1)",
]


def random_world(seed):
    rng = random.Random(seed)
    nodes = list(range(10))
    bank_of = {n: rng.choice([1, 2, 2, 3]) for n in nodes}
    subj = rng.choice([n for n in nodes if bank_of[n] == 2])
    rows = []
    for _ in range(rng.randint(25, 70)):
        a, b = rng.sample(nodes, 2)
        if rng.random() < 0.5:
            a, b = (subj, b) if rng.random() < 0.5 else (a, subj)
        if a == b:
            continue
        ts = T0 + timedelta(minutes=rng.randint(-12 * 24 * 60, 5 * 24 * 60))
        amt = rng.choice([100.0, 500.0, 1000.0, 1000.0, 3000.0, 9500.0, 2500.0, 50.0]) * rng.choice([1, 1, 1.005, 0.5, 2])
        rows.append(dict(ts=ts, src=a, dst=b, fb=bank_of[a], tb=bank_of[b], paid=amt, recv=amt,
                         pcur=rng.choice(["USD", "USD", "EUR"]), rcur=rng.choice(["USD", "EUR"]),
                         fmt=rng.choice(["ACH", "Cash", "Wire", "Credit Card"])))
    rows = [r for r in rows if r["fb"] == 2 or r["tb"] == 2]
    df = pl.DataFrame(
        {
            "timestamp": [r["ts"] for r in rows], "from_bank": [r["fb"] for r in rows],
            "to_bank": [r["tb"] for r in rows], "src_id": [r["src"] for r in rows], "dst_id": [r["dst"] for r in rows],
            "amount_received": [r["recv"] for r in rows], "amount_paid": [r["paid"] for r in rows],
            "receiving_currency": [r["rcur"] for r in rows], "payment_currency": [r["pcur"] for r in rows],
            "payment_format": [r["fmt"] for r in rows],
        },
        schema={"timestamp": pl.Datetime("us"), "from_bank": pl.Int64, "to_bank": pl.Int64, "src_id": pl.Int64,
                "dst_id": pl.Int64, "amount_received": pl.Float64, "amount_paid": pl.Float64,
                "receiving_currency": pl.String, "payment_currency": pl.String, "payment_format": pl.String},
    )
    others = [n for n in nodes if n != subj]
    c = ExecContext(
        bank=2, t0=T0, cutoff=T0 + timedelta(hours=rng.choice([6, 12, 24, 72, 200])),
        alert_amount=rng.choice([500.0, 1000.0, 3000.0]), alert_currency="USD", home_bank=1,
        sender=rng.choice(others), receiver=subj,
        payers=frozenset(rng.sample(others, 4)), payees=frozenset(rng.sample(others, 4)),
        entity_of={n: n % 4 for n in nodes},
    )
    return df, rows, c


@pytest.mark.parametrize("seed", range(50))
def test_executor_matches_brute_force_on_random_graphs(seed):
    df, rows, c = random_world(seed)
    programs = [p["program"] for p in expert_programs("receiver")] + [p.format(s="receiver") for p in EXTRA]
    programs += [p["program"] for p in expert_programs("sender")][:12]
    for text in programs:
        node = parse(text)
        want = brute(node, rows, c)
        got = execute(node, df, c)
        assert got == want, (seed, text, got, want)
    assert len(programs) > 70
