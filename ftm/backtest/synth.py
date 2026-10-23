"""A small synthetic world for tests and smoke runs: alerts whose partner-side behaviour depends on the label.

Owner: T2 (Bhavya).  Not data and never a result: it exists so the back-testing engine, the harness and the planners
can be exercised on a laptop without the AMLworld files. `signal` scales how differently laundering and clean
alerts behave at the partner bank (0 = no difference at all, so every question is a null question).

Each alert: a home-bank customer pays an account at a partner bank. Before the alert the partner account received
money from a few accounts (more of them for laundering); after it, a laundering account forwards most of the money
within hours far more often than a clean one. Cash use and amounts are independent of the label (null signals).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl

from ftm.backtest.engine import BtAlert
from ftm.banks.binding import bind_alert
from ftm.banks.partner import BankConfig, PartnerBank

HOME = 1
PARTNERS = (2, 3, 4)
T_START = datetime(2022, 9, 1, 0, 0)
SCHEMA = {
    "timestamp": pl.Datetime("us"), "from_bank": pl.Int64, "to_bank": pl.Int64, "src_id": pl.Int64,
    "dst_id": pl.Int64, "amount_received": pl.Float64, "amount_paid": pl.Float64,
    "receiving_currency": pl.String, "payment_currency": pl.String, "payment_format": pl.String,
}  # fmt: skip


@dataclass
class World:
    banks: dict[int, PartnerBank]
    alerts: list[BtAlert]
    ledgers: dict[int, pl.DataFrame]
    home_view: pl.DataFrame
    full: pl.DataFrame  # every transaction, time-sorted


def _poisson(rng: random.Random, lam: float) -> int:
    import math

    l_, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= l_:
            return k
        k += 1


def make_world(n_alerts: int = 400, seed: int = 0, signal: float = 1.0, prevalence: float = 0.2,
               bank_config: BankConfig | None = None, days: float = 9.0) -> World:  # fmt: skip
    rng = random.Random(seed)
    rows: list[dict] = []
    bank_of: dict[int, int] = {}
    nxt = [0]

    def new_node(bank: int) -> int:
        nxt[0] += 1
        bank_of[nxt[0]] = bank
        return nxt[0]

    def tx(ts, a, b, amt, fmt="ACH", cur="USD"):
        rows.append(dict(timestamp=ts, from_bank=bank_of[a], to_bank=bank_of[b], src_id=a, dst_id=b,
                         amount_received=amt, amount_paid=amt, receiving_currency=cur, payment_currency=cur,
                         payment_format=fmt))  # fmt: skip

    alerts_meta = []
    for i in range(n_alerts):
        y = rng.random() < prevalence
        t0 = T_START + timedelta(days=4 + rng.random() * (days - 4))
        bank = rng.choice(PARTNERS)
        s, r = new_node(HOME), new_node(bank)
        amt = round(rng.choice([800, 1500, 3000, 6000, 9500]) * rng.uniform(0.9, 1.1), 2)
        # the alerted transfer, and what the partner account received just before it
        for _ in range(_poisson(rng, 1.5 + 3.0 * signal * y)):
            tx(t0 - timedelta(hours=rng.uniform(1, 70)), new_node(rng.choice((HOME, *PARTNERS))), r,
               round(rng.uniform(100, 4000), 2), fmt=rng.choice(["ACH", "Wire", "Cash", "ACH"]))
        tx(t0, s, r, amt)
        # what it did afterwards
        if rng.random() < 0.15 + 0.65 * signal * y:
            for _ in range(rng.randint(1, 3)):
                tx(t0 + timedelta(hours=rng.uniform(0.5, 20)), r, new_node(rng.choice((HOME, *PARTNERS, 5))),
                   round(0.85 * amt / 2, 2), fmt=rng.choice(["ACH", "Wire", "Cash", "ACH"]))
        for _ in range(_poisson(rng, 1.0)):  # unrelated activity
            tx(t0 + timedelta(hours=rng.uniform(-60, 40)), r, new_node(rng.choice(PARTNERS)),
               round(rng.uniform(20, 300), 2), fmt=rng.choice(["ACH", "Wire", "Cash"]))
        alerts_meta.append((f"S{i:05d}", y, t0, s, r, amt))
    allrows = sorted(rows, key=lambda r: r["timestamp"])
    full = pl.DataFrame(allrows, schema=SCHEMA)
    cfg = bank_config or BankConfig(p_decline=0.05, p_timeout=0.0, notes_enabled=False)
    ledgers = {b: full.filter((pl.col("from_bank") == b) | (pl.col("to_bank") == b)) for b in (HOME, *PARTNERS, 5)}
    banks = {b: PartnerBank(b, ledgers[b], cfg, seed) for b in PARTNERS}
    alerts = []
    for aid, y, t0, s, r, amt in alerts_meta:
        txrow = dict(timestamp=t0, src_id=s, dst_id=r, from_bank=HOME, to_bank=bank_of[r], amount_received=amt,
                     receiving_currency="USD")  # fmt: skip
        home_upto = ledgers[HOME].filter(pl.col("timestamp") <= t0)  # the home bank has only seen the past
        alerts.append(BtAlert(aid, bool(y), bind_alert(aid, txrow, HOME, home_upto)))
    return World(banks, alerts, ledgers, ledgers[HOME], full)
