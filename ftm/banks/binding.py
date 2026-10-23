"""AlertBinding: an alert with real node ids, as a partner bank resolves the ALERT slots (C3/C4).

Owner: T2 (Bhavya).  The LLM and the planner only ever see aliases (plan.md invariant 4); the binding holds real
node ids and exists only inside the simulator (banks, back-testing), never in a prompt or a log a model reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl

from ftm.dsl.executor import ExecContext

EVIDENCE_H = 72  # ALERT.payers / ALERT.payees look back this far in the home bank's view


@dataclass(frozen=True)
class AlertBinding:
    alert_id: str
    home_bank: int
    t0: datetime
    sender: int
    receiver: int
    amount: float  # amount_received of the alerted transaction
    currency: str  # receiving currency
    payers: frozenset = frozenset()  # accounts that paid `receiver` in the 72 h up to t0 (home bank view)
    payees: frozenset = frozenset()  # accounts `sender` paid in the 72 h up to t0 (home bank view)
    sender_bank: int | None = None  # banks of the two parties (who holds the subject of a question)
    receiver_bank: int | None = None

    @property
    def partner_bank(self) -> int | None:
        """The bank on the other side of the alerted transfer, or None if both parties are at the home bank."""
        if self.sender_bank is None or self.receiver_bank is None:
            return None
        if self.sender_bank != self.home_bank:
            return self.sender_bank
        return self.receiver_bank if self.receiver_bank != self.home_bank else None

    def slot(self, name: str):
        if name not in ("sender", "receiver", "payers", "payees"):
            raise KeyError(name)
        return getattr(self, name)

    def exec_context(self, bank: int, cutoff: datetime, entity_of: dict | None = None) -> ExecContext:
        return ExecContext(
            bank=bank, t0=self.t0, cutoff=cutoff, alert_amount=self.amount, alert_currency=self.currency,
            home_bank=self.home_bank, sender=self.sender, receiver=self.receiver, payers=self.payers,
            payees=self.payees, entity_of=entity_of,
        )  # fmt: skip


def bind_alert(alert_id: str, tx: dict, home_bank: int, home_view: pl.DataFrame) -> AlertBinding:
    """Bind the alerted transaction `tx` (a row of transactions.parquet, as a dict) to real node ids.

    `home_view` is the home bank's own ledger, already cut at the alert time (ftm.data.io guards that).
    """
    t0 = tx["timestamp"]
    lo = t0 - timedelta(hours=EVIDENCE_H)
    window = home_view.filter((pl.col("timestamp") >= lo) & (pl.col("timestamp") <= t0))
    payers = window.filter(pl.col("dst_id") == tx["dst_id"])["src_id"].unique().to_list()
    payees = window.filter(pl.col("src_id") == tx["src_id"])["dst_id"].unique().to_list()
    return AlertBinding(
        alert_id=alert_id, home_bank=int(home_bank), t0=t0, sender=int(tx["src_id"]), receiver=int(tx["dst_id"]),
        amount=float(tx["amount_received"]), currency=str(tx["receiving_currency"]),
        payers=frozenset(int(x) for x in payers), payees=frozenset(int(x) for x in payees),
        sender_bank=int(tx["from_bank"]) if "from_bank" in tx else None,
        receiver_bank=int(tx["to_bank"]) if "to_bank" in tx else None,
    )  # fmt: skip
