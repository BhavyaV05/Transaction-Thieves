"""Situation shape: the key the strategy library is filed under (C12). Never an account id.

Owner: T3 (Akshat).  Spec: plan.md 6.13, tracks.md 5.5.  SCHEMA AND KEYING FROZEN AT G3 (change = all four owners).

The key is a small set of buckets computed from ALL 1-hop edges of the alert's receiver in the 72 h window up to the
alert time (shape_from_edges), plus the planner's typology guess. It does not read the evidence rows the prompt shows
(top-40 by amount, scorer or PCST views), so it is the same under every view of the same alert (schema v2; v1 was
computed from the evidence rows and is refused on load):
    typology       planner hypothesis (or "any" before the first step)
    in_deg         distinct accounts paying the alert's receiver in the 1-hop edges      buckets 0 | 1 | 2-3 | 4-7 | 8+
    out_deg        distinct accounts the receiver pays in the 1-hop edges                  same buckets
    n_banks        distinct banks in the 1-hop edges                                    1 | 2 | 3-4 | 5+
    amount         median amount (log10 bucket) + "flat" if the coefficient of variation < 0.5, else "varied"
    span           time from the first 1-hop edge to the alert                          <6h | 6-24h | 1-3d | >3d
Buckets are coarse on purpose: a strategy should apply to many alerts, and the key must not single out an account.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from ftm import guards

SHAPE_SCHEMA_VERSION = 2
WINDOW_HOURS = 72
EDGE_COLS = ["tx_id", "timestamp", "src_id", "dst_id", "from_bank", "to_bank", "amount_paid"]
ANY = "any"
_DEG = ((0, "0"), (1, "1"), (3, "2-3"), (7, "4-7"))
_BANKS = ((1, "1"), (2, "2"), (4, "3-4"))
_AMT = ((100, "<100"), (1_000, "100-1k"), (10_000, "1k-10k"), (100_000, "10k-100k"))
_SPAN = ((6, "<6h"), (24, "6-24h"), (72, "1-3d"))


def _bucket(x: float, edges: tuple, top: str, inclusive: bool) -> str:
    """Counts use inclusive upper edges (x <= hi); amounts and hours use exclusive ones (x < hi)."""
    for hi, label in edges:
        if (x <= hi) if inclusive else (x < hi):
            return label
    return top


@dataclass(frozen=True)
class SituationShape:
    typology: str
    in_deg: str
    out_deg: str
    n_banks: str
    amount: str
    span: str

    def key(self) -> tuple:
        return (self.typology, self.in_deg, self.out_deg, self.n_banks, self.amount, self.span)

    def with_typology(self, typology: str | None) -> SituationShape:
        return SituationShape(typology or ANY, self.in_deg, self.out_deg, self.n_banks, self.amount, self.span)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SituationShape:
        return cls(**{k: str(d.get(k, ANY)) for k in ("typology", "in_deg", "out_deg", "n_banks", "amount", "span")})

    def vector(self) -> tuple[float, ...]:
        """Ordinal positions of the buckets (for nearest-shape fallback). Typology is matched, not embedded."""
        def pos(v, edges, top):
            labels = [lbl for _, lbl in edges] + [top]
            return float(labels.index(v)) if v in labels else -1.0

        amt = self.amount.split(" ")[0]
        return (pos(self.in_deg, _DEG, "8+"), pos(self.out_deg, _DEG, "8+"), pos(self.n_banks, _BANKS, "5+"),
                pos(amt, _AMT, "100k+"), 0.5 if self.amount.endswith("flat") else 0.0, pos(self.span, _SPAN, ">3d"))  # fmt: skip

    def describe(self) -> str:
        return (f"{self.typology if self.typology != ANY else 'any typology'}, {self.in_deg} payers in, "
                f"{self.out_deg} payees out, {self.n_banks} banks, amounts {self.amount}, over {self.span}")  # fmt: skip


def shape_from_rows(rows: list[dict], receiver: str, alert_time, typology: str | None = None) -> SituationShape:
    """`rows`: edges around `receiver` (src_alias, dst_alias, from_bank, to_bank, amount_paid, timestamp). Production
    callers go through shape_from_edges, which picks the rows independently of any evidence view."""
    payers = {r["src_alias"] for r in rows if r["dst_alias"] == receiver}
    payees = {r["dst_alias"] for r in rows if r["src_alias"] == receiver}
    banks = {r["from_bank"] for r in rows} | {r["to_bank"] for r in rows}
    amounts = [float(r["amount_paid"]) for r in rows] or [0.0]
    med = statistics.median(amounts)
    cv = statistics.pstdev(amounts) / statistics.mean(amounts) if len(amounts) > 1 and statistics.mean(amounts) > 0 else 0.0
    t_alert = datetime.fromisoformat(alert_time) if isinstance(alert_time, str) else alert_time
    first = min((r["timestamp"] for r in rows), default=t_alert)
    first = datetime.fromisoformat(first) if isinstance(first, str) else first
    span_h = max((t_alert - first).total_seconds() / 3600, 0.0)
    return SituationShape(
        typology=typology or ANY,
        in_deg=_bucket(len(payers), _DEG, "8+", inclusive=True),
        out_deg=_bucket(len(payees), _DEG, "8+", inclusive=True),
        n_banks=_bucket(len(banks), _BANKS, "5+", inclusive=True),
        amount=f"{_bucket(med, _AMT, '100k+', inclusive=False)} {'flat' if cv < 0.5 else 'varied'}",
        span=_bucket(span_h, _SPAN, ">3d", inclusive=False),
    )


def shape_from_edges(edges: pl.DataFrame, receiver_id, alert_time, *, window_hours: float = WINDOW_HOURS,
                     typology: str | None = None) -> SituationShape:  # fmt: skip
    """The library key from node-id edges (tx_id, timestamp, src_id, dst_id, from_bank, to_bank, amount_paid): the
    1-hop edges of `receiver_id` in [alert_time - window_hours, alert_time]. Pure: no files. Edges after the alert
    are dropped (and a frame whose kept rows are still in the future is refused), so the key has no look-ahead."""
    t_alert = datetime.fromisoformat(alert_time) if isinstance(alert_time, str) else alert_time
    kept = edges.filter(((pl.col("src_id") == receiver_id) | (pl.col("dst_id") == receiver_id))
                        & (pl.col("timestamp") >= t_alert - timedelta(hours=window_hours))
                        & (pl.col("timestamp") <= t_alert))  # fmt: skip
    guards.assert_not_future(kept["timestamp"].max() if kept.height else None, t_alert)
    rows = [{"src_alias": r["src_id"], "dst_alias": r["dst_id"], "from_bank": r["from_bank"], "to_bank": r["to_bank"],
             "amount_paid": r["amount_paid"], "timestamp": r["timestamp"]} for r in kept.iter_rows(named=True)]  # fmt: skip
    return shape_from_rows(rows, receiver_id, t_alert, typology)


def receiver_edges(views_dir: str | Path, bank: int, receiver_id, as_of) -> pl.DataFrame:
    """The alert bank's view up to `as_of`, read through ftm.data.io (time guard, test lock), narrowed to edges that
    touch the receiver. Feed to shape_from_edges."""
    from ftm.data import io

    lf = io.scan_view(views_dir, bank, as_of=as_of, columns=EDGE_COLS)
    return lf.filter((pl.col("src_id") == receiver_id) | (pl.col("dst_id") == receiver_id)).collect()
