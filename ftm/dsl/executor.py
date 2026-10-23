"""Exact execution of a DSL question on one partner bank's ledger (C4, the vectorised executor).

Owner: T2 (Bhavya).  Used by ftm/banks (C3) before noise is added, and by back-testing (C7).

Inputs: the bank's ledger (a Polars frame: every transaction with an account of that bank on at least one side;
`ftm.data.views`), and an `ExecContext` that binds the ALERT slots to real node ids and fixes the clock. Output: one
scalar (bool or int). No row ever leaves this module, and the ledger is cut at `ctx.cutoff` before anything else, so
a question cannot see past the partner's answer time (time guard, plan.md 5.2 step 6).

Semantics worth knowing (full list in ftm/dsl/README.md):
  * SUBJECT is an account the asked bank holds: a ledger row counts for SUBJECT only on the side whose bank is
    `ctx.bank`. A bank that does not hold SUBJECT sees no events and answers 0 / false.
  * Windows: before(h) = [t0-h, t0]; after(h) = (t0, t0+h]; around(h) = [t0-h/2, t0+h/2]; all cut at `cutoff`.
    The alerted transaction itself is part of the data and falls in before/around windows.
  * "amount" is the amount on SUBJECT's side: amount_paid for money SUBJECT sent, amount_received for money it got.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import polars as pl

from ftm.dsl.parser import Node
from ftm.dsl.spec import Window

LEDGER_COLUMNS = (
    "timestamp", "from_bank", "to_bank", "src_id", "dst_id", "amount_received", "amount_paid",
    "receiving_currency", "payment_currency", "payment_format",
)  # fmt: skip
REPEAT_TOL = 0.01
SAME_AMOUNT_TOL = 1e-9


class ExecError(RuntimeError):
    """The question cannot be run on this ledger/context (e.g. it needs the entity map and none was given)."""


@dataclass(frozen=True)
class ExecContext:
    bank: int  # the asked bank
    t0: datetime  # alert time
    cutoff: datetime  # partner's answer time: the ledger is cut here
    alert_amount: float  # amount the alerted receiver got (amount_received of the alerted transaction)
    alert_currency: str  # receiving currency of the alerted transaction
    home_bank: int
    sender: int  # node ids of the alerted transaction's parties
    receiver: int
    payers: frozenset = frozenset()
    payees: frozenset = frozenset()
    entity_of: dict | None = None  # node id -> entity id (needed by n_same_entity only)
    extra: dict = field(default_factory=dict)

    def slot(self, name: str):
        return getattr(self, name)


# ---------------------------------------------------------------------------------------------------------------
def execute(node: Node, ledger: pl.DataFrame, ctx: ExecContext) -> bool | int:
    """Exact answer of `node`: bool for bool questions, int for count / set_count / k-ary."""
    missing = [c for c in LEDGER_COLUMNS if c not in ledger.columns]
    if missing:
        raise ExecError(f"ledger is missing columns {missing}")
    if ctx.cutoff < ctx.t0:
        raise ExecError("cutoff is before the alert time")
    led = ledger.filter(pl.col("timestamp") <= ctx.cutoff)
    return _run(node, led, ctx)


def _run(node: Node, led: pl.DataFrame, ctx: ExecContext):
    if node.name == "at_least":
        return _run(node.subject, led, ctx) >= node.arg("k")
    if node.name == "set_contacts":
        return _set_contacts(node, led, ctx)
    ev = _events(led, ctx.slot(node.subject), ctx.bank)
    fn = _IMPL.get(node.name)
    if fn is None:
        raise ExecError(f"{node.name} has no executor")
    return fn(node, ev, led, ctx)


def _events(led: pl.DataFrame, subj: int, bank: int) -> pl.DataFrame:
    """SUBJECT's transfers as seen by `bank`: one row per (transfer, side)."""
    out = led.filter((pl.col("src_id") == subj) & (pl.col("from_bank") == bank)).select(
        "timestamp",
        pl.lit("out").alias("dir"),
        pl.col("dst_id").alias("cp"),
        pl.col("to_bank").alias("cp_bank"),
        pl.col("amount_paid").alias("amt"),
        pl.col("payment_currency").alias("cur"),
        pl.col("payment_format").alias("fmt"),
    )
    inn = led.filter((pl.col("dst_id") == subj) & (pl.col("to_bank") == bank)).select(
        "timestamp",
        pl.lit("in").alias("dir"),
        pl.col("src_id").alias("cp"),
        pl.col("from_bank").alias("cp_bank"),
        pl.col("amount_received").alias("amt"),
        pl.col("receiving_currency").alias("cur"),
        pl.col("payment_format").alias("fmt"),
    )
    return pl.concat([out, inn])


def _window_mask(w: Window, ctx: ExecContext) -> pl.Expr:
    ts = pl.col("timestamp")
    if w.mode == "before":
        return (ts >= ctx.t0 - timedelta(hours=w.hours)) & (ts <= ctx.t0)
    if w.mode == "after":
        return (ts > ctx.t0) & (ts <= ctx.t0 + timedelta(hours=w.hours))
    half = timedelta(hours=w.hours / 2)
    return (ts >= ctx.t0 - half) & (ts <= ctx.t0 + half)


def _win(ev: pl.DataFrame, w: Window, ctx: ExecContext, direction: str = "any") -> pl.DataFrame:
    ev = ev.filter(_window_mask(w, ctx))
    return ev if direction == "any" else ev.filter(pl.col("dir") == direction)


def _total(ev: pl.DataFrame) -> float:
    return float(ev["amt"].sum() or 0.0)


# ---------------------------------------------------------------------------------------------------------------
def _n_counterparties(n, ev, led, ctx):
    return int(_win(ev, n.arg("window"), ctx, n.arg("dir"))["cp"].n_unique())


def _n_banks(n, ev, led, ctx):
    return int(_win(ev, n.arg("window"), ctx, n.arg("dir"))["cp_bank"].n_unique())


def _n_tx(n, ev, led, ctx):
    w = _win(ev, n.arg("window"), ctx, n.arg("dir"))
    return w.filter((pl.col("amt") >= n.arg("min_amount")) & (pl.col("amt") < n.arg("max_amount"))).height


def _n_round(n, ev, led, ctx):
    unit = n.arg("unit")
    w = _win(ev, n.arg("window"), ctx).filter(pl.col("amt") > 0)
    k = (pl.col("amt") / unit).round(0)
    return w.filter(((pl.col("amt") / unit) - k).abs() * unit < 0.005).height


def _n_format(n, ev, led, ctx):
    fmt = pl.col("fmt").str.to_lowercase().str.replace_all(" ", "_")
    return _win(ev, n.arg("window"), ctx).filter(fmt == n.arg("format")).height


def _n_repeated(n, ev, led, ctx):
    amts = _win(ev, n.arg("window"), ctx)["amt"].to_list()
    return sum(
        1
        for i, a in enumerate(amts)
        if any(j != i and abs(a - b) <= REPEAT_TOL * max(a, b) for j, b in enumerate(amts))
    )


def _n_same_entity(n, ev, led, ctx):
    if ctx.entity_of is None:
        raise ExecError("n_same_entity needs ExecContext.entity_of")
    me = ctx.entity_of.get(ctx.slot(n.subject))
    if me is None:
        return 0
    w = _win(ev, n.arg("window"), ctx)
    return sum(1 for c in w["cp"].to_list() if c != ctx.slot(n.subject) and ctx.entity_of.get(c) == me)


def _n_new_counterparties(n, ev, led, ctx):
    paid_before = set(ev.filter((pl.col("dir") == "out") & (pl.col("timestamp") <= ctx.t0))["cp"].to_list())
    now = set(_win(ev, n.arg("window"), ctx, "out")["cp"].to_list())
    return len(now - paid_before)


def _chain_length(n, ev, led, ctx):
    w = n.arg("window")
    mask = _window_mask(w, ctx)
    edges = (
        led.filter((pl.col("from_bank") == ctx.bank) & (pl.col("to_bank") == ctx.bank) & mask)
        .select("src_id", "dst_id", "timestamp")
        .rows()
    )
    reach = {ctx.slot(n.subject): ctx.t0}  # node -> earliest time it can be left, after k hops
    best = 0
    for hop in range(1, n.arg("max_hops") + 1):
        nxt: dict = {}
        for s, d, t in edges:
            if s in reach and t >= reach[s] and (hop > 1 or t > ctx.t0):
                if d not in nxt or t < nxt[d]:
                    nxt[d] = t
        if not nxt:
            break
        best, reach = hop, nxt
    return best


def _passed_on(n, ev, led, ctx):
    received = _total(_win(ev, Window("before", 72), ctx, "in"))
    sent = _total(_win(ev, n.arg("window"), ctx, "out"))
    return received > 0 and sent >= n.arg("min_frac") * received - SAME_AMOUNT_TOL


def _returns_home(n, ev, led, ctx):
    return _win(ev, n.arg("window"), ctx, "out").filter(pl.col("cp_bank") == ctx.home_bank).height > 0


def _matching_amount_out(n, ev, led, ctx):
    out = _win(ev, Window("after", 24), ctx, "out")
    tol = n.arg("tol")
    return any(abs(a - ctx.alert_amount) <= tol * ctx.alert_amount + SAME_AMOUNT_TOL for a in out["amt"].to_list())


def _velocity_balanced(n, ev, led, ctx):
    w = _win(ev, n.arg("window"), ctx)
    inflow, outflow = _total(w.filter(pl.col("dir") == "in")), _total(w.filter(pl.col("dir") == "out"))
    return inflow >= ctx.alert_amount and 0.9 * inflow - SAME_AMOUNT_TOL <= outflow <= 1.1 * inflow + SAME_AMOUNT_TOL


def _cross_currency_out(n, ev, led, ctx):
    return _win(ev, n.arg("window"), ctx, "out").filter(pl.col("cur") != ctx.alert_currency).height > 0


def _dormant(n, ev, led, ctx):
    hi = ctx.t0 - timedelta(hours=72)
    lo = hi - timedelta(days=n.arg("days"))
    return ev.filter((pl.col("timestamp") >= lo) & (pl.col("timestamp") < hi)).height == 0


def _new_account(n, ev, led, ctx):
    if ev.height == 0:
        return False
    return ev["timestamp"].min() >= ctx.t0 - timedelta(days=n.arg("days"))


def _volume_spike(n, ev, led, ctx):
    ins = ev.filter(pl.col("dir") == "in")
    day = timedelta(hours=24)
    last = _total(ins.filter((pl.col("timestamp") >= ctx.t0 - day) & (pl.col("timestamp") <= ctx.t0)))
    prev = _total(ins.filter((pl.col("timestamp") >= ctx.t0 - day - timedelta(days=7)) & (pl.col("timestamp") < ctx.t0 - day)))
    return last > 0 and last >= n.arg("mult") * (prev / 7) - SAME_AMOUNT_TOL


def _payees_pay_each_other(n, ev, led, ctx):
    payees = set(_win(ev, n.arg("window"), ctx, "out")["cp"].to_list())
    if len(payees) < 2:
        return False
    pairs = led.filter(_window_mask(n.arg("window"), ctx) & (pl.col("src_id") != pl.col("dst_id")))
    return any(s in payees and d in payees for s, d in pairs.select("src_id", "dst_id").rows())


def _top_outflow_bank(n, ev, led, ctx):
    banks = n.arg("banks")
    out = _win(ev, n.arg("window"), ctx, "out")
    totals = [_total(out.filter(pl.col("cp_bank") == b)) for b in banks]
    best = max(totals)
    return totals.index(best) if best > 0 else len(banks)


def _time_to_forward(n, ev, led, ctx):
    out = ev.filter((pl.col("dir") == "out") & (pl.col("timestamp") > ctx.t0)).sort("timestamp")
    need, acc = 0.5 * ctx.alert_amount, 0.0
    for t, a in zip(out["timestamp"].to_list(), out["amt"].to_list(), strict=True):
        acc += a
        if acc >= need - SAME_AMOUNT_TOL:
            h = (t - ctx.t0).total_seconds() / 3600
            return 0 if h < 1 else 1 if h < 6 else 2 if h < 24 else 3
    return 3


def _inflow_bucket(n, ev, led, ctx):
    ratio = _total(_win(ev, Window("before", 72), ctx, "in")) / ctx.alert_amount
    return 0 if ratio < 1 else 1 if ratio < 3 else 2 if ratio < 10 else 3


def _overlap(n, ev, led, ctx):
    members = ctx.slot(n.arg("among"))
    return len(set(_win(ev, n.arg("window"), ctx, n.arg("dir"))["cp"].to_list()) & set(members))


def _set_contacts(n, led, ctx):
    members = set(ctx.slot(n.subject))
    w = led.filter(_window_mask(n.arg("window"), ctx))
    if n.arg("dir") == "in":  # a member received from an account at the asked bank
        hit = w.filter((pl.col("from_bank") == ctx.bank) & pl.col("dst_id").is_in(list(members)))["dst_id"]
    else:  # a member sent to an account at the asked bank
        hit = w.filter((pl.col("to_bank") == ctx.bank) & pl.col("src_id").is_in(list(members)))["src_id"]
    return int(hit.n_unique())


_IMPL = {
    "n_counterparties": _n_counterparties,
    "n_banks": _n_banks,
    "n_tx": _n_tx,
    "n_round": _n_round,
    "n_format": _n_format,
    "n_repeated": _n_repeated,
    "n_same_entity": _n_same_entity,
    "n_new_counterparties": _n_new_counterparties,
    "chain_length": _chain_length,
    "passed_on": _passed_on,
    "returns_home": _returns_home,
    "matching_amount_out": _matching_amount_out,
    "velocity_balanced": _velocity_balanced,
    "cross_currency_out": _cross_currency_out,
    "dormant": _dormant,
    "new_account": _new_account,
    "volume_spike": _volume_spike,
    "payees_pay_each_other": _payees_pay_each_other,
    "top_outflow_bank": _top_outflow_bank,
    "time_to_forward": _time_to_forward,
    "inflow_bucket": _inflow_bucket,
    "overlap": _overlap,
}
