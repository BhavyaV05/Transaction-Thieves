"""Evidence views for the planner (C11). This is the week-2 STUB: top-K edges by amount around the alert.

Owner: T3 (Akshat) until week 4, then T1 (Abhishek) owns it (hand-off #11).  Spec: plan.md 6.12, tracks.md 3.8.

View "top-40 by amount" (the RQ4 baseline view and the default training view for the fine-tuned planner):
  1. the home bank's view up to `as_of` (the alert time), read ONLY through ftm.data.io (time guard + test lock);
  2. edges that touch the alert's sender or receiver (1 hop), inside [as_of - window_hours, as_of];
  3. the K largest by amount_paid (ties: lower tx_id first), always including the alerted transaction;
  4. re-sorted by (timestamp, tx_id) into one time-ordered trail;
  5. node ids replaced by aliases (invariant 4) before anything is formatted for an LLM.
The scorer-only, scorer+PCST and long-prompt views (T1, weeks 7-9) should keep `select_*` -> `format_trail`.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from ftm import guards
from ftm.data import alias, io

K_DEFAULT = 40
WINDOW_HOURS = 72
TRAIL_COLS = ["tx_id", "timestamp", "src_id", "dst_id", "from_bank", "to_bank", "amount_paid", "payment_currency",
              "payment_format"]  # fmt: skip
_FORBIDDEN = {"src_id", "dst_id", "from_account", "to_account", "is_laundering", "split"}


def _as_dt(as_of) -> datetime:
    return datetime.fromisoformat(as_of) if isinstance(as_of, str) else as_of


def select_top_k_by_amount(
    frame: pl.DataFrame | pl.LazyFrame,
    accounts: list[int],
    as_of,
    *,
    k: int = K_DEFAULT,
    window_hours: float = WINDOW_HOURS,
    alert_tx_id: int | None = None,
) -> pl.DataFrame:
    """Steps 2-4 on a frame with node ids (`src_id`, `dst_id`). Applies `timestamp <= as_of` itself as well,
    so a caller that skipped ftm.data.io still cannot select future rows."""
    if as_of is None:
        raise guards.TimeGuardError("as_of is required")
    as_of = _as_dt(as_of)
    lo = as_of - timedelta(hours=window_hours)
    lf = frame.lazy().filter(
        (pl.col("src_id").is_in(accounts) | pl.col("dst_id").is_in(accounts))
        & (pl.col("timestamp") >= lo)
        & (pl.col("timestamp") <= as_of)
    )
    rows = lf.sort(["amount_paid", "tx_id"], descending=[True, False]).collect()
    guards.assert_not_future(rows["timestamp"].max() if rows.height else None, as_of)
    top = rows.head(k)
    if alert_tx_id is not None and alert_tx_id not in set(top["tx_id"].to_list()):
        alert_row = rows.filter(pl.col("tx_id") == alert_tx_id)
        if alert_row.height:
            top = pl.concat([top.head(k - 1), alert_row])
    return top.sort(["timestamp", "tx_id"])


def format_trail(rows: pl.DataFrame, alert_tx_id: int | None = None, home_bank: int | None = None) -> str:
    """One line per edge for the prompt. Refuses frames that still carry raw ids or labels."""
    leaked = _FORBIDDEN & set(rows.columns)
    if leaked:
        raise ValueError(f"format_trail needs aliased rows without labels; found {sorted(leaked)}")
    lines = []
    for r in rows.iter_rows(named=True):
        mark = " <- ALERT" if alert_tx_id is not None and r.get("tx_id") == alert_tx_id else ""
        fb, tb = r["from_bank"], r["to_bank"]
        banks = f"bank {fb} -> bank {tb}"
        if home_bank is not None:
            banks += "" if fb == tb == home_bank else (" (outgoing)" if fb == home_bank else " (incoming)")
        lines.append(
            f"{r['timestamp']:%Y-%m-%d %H:%M} | {r['src_alias']} -> {r['dst_alias']} | "
            f"{r['amount_paid']:,.2f} {r['payment_currency']} | {r['payment_format']} | {banks}{mark}"
        )
    return "\n".join(lines)


def alert_summary(alert_row: dict, aliases: pl.DataFrame) -> dict:
    """Aliased fields of the alerted transaction for AlertContext.summary (no raw ids, no label)."""
    amap = dict(zip(aliases["node_id"].to_list(), aliases["alias"].to_list(), strict=True))
    return {
        "sender": amap[alert_row["src_id"]],
        "receiver": amap[alert_row["dst_id"]],
        "sender_bank": alert_row["from_bank"],
        "receiver_bank": alert_row["to_bank"],
        "amount": round(float(alert_row["amount_paid"]), 2),
        "currency": alert_row["payment_currency"],
        "format": alert_row["payment_format"],
        "time": f"{_as_dt(alert_row['timestamp']):%Y-%m-%d %H:%M}",
    }


def top_k_by_amount_view(
    views_dir: str | Path,
    bank: int,
    alert_row: dict,
    aliases: pl.DataFrame,
    *,
    k: int = K_DEFAULT,
    window_hours: float = WINDOW_HOURS,
) -> tuple[pl.DataFrame, str]:
    """Full pipeline for one alert. `alert_row` needs tx_id, timestamp, src_id, dst_id (node ids, server side).

    Returns (aliased rows, prompt text). The as_of is the alert timestamp.
    """
    as_of = alert_row["timestamp"]
    view = io.scan_view(views_dir, bank, as_of=as_of, columns=TRAIL_COLS)
    rows = select_top_k_by_amount(view, [alert_row["src_id"], alert_row["dst_id"]], as_of, k=k,
                                  window_hours=window_hours, alert_tx_id=alert_row["tx_id"])  # fmt: skip
    aliased = alias.apply_aliases(rows, aliases)
    return aliased, format_trail(aliased, alert_tx_id=alert_row["tx_id"], home_bank=bank)
