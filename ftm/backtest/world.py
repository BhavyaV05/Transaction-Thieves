"""Load the real world for back-testing: partner banks with their private ledgers and the labelled past alerts.

Owner: T2 (Bhavya).  Reads only through `ftm.data.io` (time guard and test lock), the alert sets of T1
(`data/alerts/<ds>/{train,val}.parquet`, hand-off #6) and the per-bank views (hand-off #1). Only the train and val
splits can be loaded here; the test split stays locked (ftm.guards).

A partner bank's ledger is its view restricted to the split being back-tested, so an answer can never see a later split.
The alert's label comes from the alert set and goes into `BtAlert.label`, which only the back-test engine reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from ftm.backtest.engine import BtAlert
from ftm.banks.binding import bind_alert
from ftm.banks.partner import BankConfig, PartnerBank
from ftm.data import io

LEDGER_COLUMNS = [
    "timestamp", "from_bank", "to_bank", "src_id", "dst_id", "amount_received", "amount_paid",
    "receiving_currency", "payment_currency", "payment_format",
]  # fmt: skip
ALERT_TX_COLUMNS = ["tx_id", "timestamp", "from_bank", "to_bank", "src_id", "dst_id", "amount_received", "receiving_currency"]


@dataclass
class RealWorld:
    banks: dict[int, PartnerBank]
    alerts: list[BtAlert]
    alert_set_id: str
    n_skipped: int  # alerts with no partner bank (both parties at the home bank) or no ledger
    ledgers: dict = field(default_factory=dict)  # bank -> its view of the split (the home banks' views too)


def load_world(data: str, split: str = "train", seed: int = 0, bank_config: BankConfig | None = None,
               max_alerts: int | None = None, only_keep: bool = False, root: str | Path = ".") -> RealWorld:  # fmt: skip
    if split not in ("train", "val"):
        raise ValueError("back-testing runs on train or val alerts only; the test split is locked")
    root = Path(root)
    cfg = io.load_config(data)
    pq, views = root / cfg["parquet_dir"], root / cfg["views_dir"]
    alerts = pl.read_parquet(root / cfg["alerts_dir"] / f"{split}.parquet")  # time-guard: not transactions
    if only_keep and "keep" in alerts.columns:
        alerts = alerts.filter(pl.col("keep"))
    alerts = alerts.sort(["timestamp", "alert_id"])
    if max_alerts:
        alerts = alerts.head(max_alerts)
    tx = io.scan_transactions(pq, split=split, columns=ALERT_TX_COLUMNS).collect()
    if "typology" not in alerts.columns:
        alerts = alerts.with_columns(pl.lit(None, dtype=pl.String).alias("typology"))
    joined = alerts.select("alert_id", "tx_id", "bank", "label", "p0", "typology").join(tx, on="tx_id", how="left")
    if joined["timestamp"].null_count():
        raise ValueError("some alerts point at transactions outside the requested split")
    entity_of = _entities(pq)
    home_views: dict[int, pl.DataFrame] = {}
    ledgers: dict[int, pl.DataFrame] = {}

    def view(bank: int) -> pl.DataFrame:
        if bank not in ledgers:
            ledgers[bank] = io.scan_view(views, bank, split=split, columns=LEDGER_COLUMNS).collect()
        return ledgers[bank]

    out, skipped = [], 0
    for row in joined.iter_rows(named=True):
        home = int(row["bank"])
        if home not in home_views:
            home_views[home] = view(home).select("timestamp", "src_id", "dst_id")
        b = bind_alert(row["alert_id"], row, home, home_views[home])
        if b.partner_bank is None:
            skipped += 1
            continue
        out.append(BtAlert(row["alert_id"], bool(row["label"]), b, float(row["p0"]), row["typology"], split))
    partners = {a.binding.partner_bank for a in out}
    banks = {p: PartnerBank(p, view(p), bank_config or BankConfig.load(), seed, entity_of) for p in sorted(partners)}
    return RealWorld(banks, out, f"{data}-{split}-{len(out)}-{_digest(out)}", skipped, ledgers)


def _digest(alerts: list[BtAlert]) -> str:
    import hashlib

    h = hashlib.blake2b(digest_size=4)
    for a in alerts:
        h.update(a.alert_id.encode())
    return h.hexdigest()


def _entities(parquet_dir: Path) -> dict | None:
    p = parquet_dir / "accounts.parquet"
    if not p.exists():
        return None
    acc = io.read_accounts(parquet_dir).filter(pl.col("entity_id").is_not_null())
    return dict(zip(acc["node_id"].to_list(), acc["entity_id"].to_list(), strict=True))
