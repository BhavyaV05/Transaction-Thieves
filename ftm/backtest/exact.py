"""Exact-mode back-testing (the ablation of plan.md 6.8) and the realistic-vs-exact comparison (G2).

Owner: T2 (Bhavya); statistics reviewed by T4.  Spec: plan.md 6.8, tracks.md 4.6 (week 6).

Realistic mode (the main results): the labels come from the home banks' own past *alerts* (the hard cases that passed
intake) and the partner banks answer with live noise. **Exact mode**: the labels of *every bank's* transactions, and no
noise (eps = inf). It tells how much of a question's value is lost to the noise and to the narrower, harder alert set.

`exact_alerts` draws a label-stratified sample of cross-bank transactions from a split (all or `n_pos` laundering ones,
`neg_ratio` times as many clean ones), treats each as an alert at one of its two banks, and binds it like a real alert.
Because the sample is stratified by label, the class-conditional answer distributions, and therefore the LRs, do not
depend on the sampling rates; only the base rate does, and LRs never use it. Train and val splits only.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from ftm.backtest.engine import BtAlert
from ftm.banks.binding import bind_alert

TX_COLUMNS = ["tx_id", "timestamp", "from_bank", "to_bank", "src_id", "dst_id", "amount_received", "receiving_currency"]


def exact_alerts(tx: pl.DataFrame, labels: pl.Series, ledgers: dict[int, pl.DataFrame], n_pos: int = 2000,
                 neg_ratio: int = 4, seed: int = 0, split: str = "train") -> list[BtAlert]:  # fmt: skip
    """Label-stratified sample of cross-bank transactions as back-test alerts.

    `tx` has TX_COLUMNS, `labels` is its is_laundering column, `ledgers` the bank views (every sampled transaction needs
    both of its banks in `ledgers`). Each transaction becomes an alert at the bank of its sender or its receiver, chosen
    by a seeded coin, with the other bank as the partner."""
    if split not in ("train", "val"):
        raise ValueError("exact-mode back-testing runs on train or val only; the test split is locked")
    ok = (pl.col("from_bank") != pl.col("to_bank")) & pl.col("from_bank").is_in(list(ledgers)) & pl.col("to_bank").is_in(list(ledgers))
    df = tx.with_columns(labels.alias("_y")).filter(ok)
    pos, neg = df.filter(pl.col("_y")), df.filter(~pl.col("_y"))
    rng = random.Random(seed)

    def pick(frame: pl.DataFrame, n: int) -> pl.DataFrame:
        n = min(n, frame.height)
        idx = sorted(rng.sample(range(frame.height), n))
        return frame[idx]

    chosen = pl.concat([pick(pos, n_pos), pick(neg, n_pos * neg_ratio)]).sort(["timestamp", "tx_id"])
    out = []
    for row in chosen.iter_rows(named=True):
        home = row["from_bank"] if random.Random(f"{seed}|{row['tx_id']}").random() < 0.5 else row["to_bank"]
        view = ledgers[home].select("timestamp", "src_id", "dst_id")
        aid = f"X{row['tx_id']:08d}"
        b = bind_alert(aid, row, home, view.filter(pl.col("timestamp") <= row["timestamp"]))
        out.append(BtAlert(aid, bool(row["_y"]), b, 0.5, None, split))
    return out


def compare_modes(realistic: Sequence[dict], exact: Sequence[dict]) -> dict:
    """Realistic (noisy, alert labels) vs exact (no noise, all labels) back-test entries of the same questions.

    Matched by canonical program (the realistic entries may be at any eps). Reports, over the questions eligible in both:
    how many are accepted in each, the agreement, the rank correlation of their strength (max |log LR|), and the median
    ratio of realistic to exact strength (what the noise and the harder alert set cost)."""
    ex = {e["program_canonical"]: e for e in exact if e["eligible"]}
    pairs = [(r, ex[r["program_canonical"]]) for r in realistic if r["eligible"] and r["program_canonical"] in ex]
    if not pairs:
        return {"n_common": 0}

    def strength(e: dict) -> float:
        vals = [abs(math.log(v)) for v in e["lr"].values() if v > 0 and math.isfinite(v)]
        return max(vals, default=0.0)

    s_r, s_e = np.array([strength(r) for r, _ in pairs]), np.array([strength(e) for _, e in pairs])
    a_r, a_e = [bool(r["accepted"]) for r, _ in pairs], [bool(e["accepted"]) for _, e in pairs]
    both = sum(x and y for x, y in zip(a_r, a_e, strict=True))
    rho = float(spearmanr(s_r, s_e).statistic) if len(pairs) > 2 and s_r.std() > 0 and s_e.std() > 0 else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = s_r[s_e > 0] / s_e[s_e > 0]
    return {
        "n_common": len(pairs), "accepted_realistic": sum(a_r), "accepted_exact": sum(a_e), "accepted_both": both,
        "accepted_only_realistic": sum(x and not y for x, y in zip(a_r, a_e, strict=True)),
        "accepted_only_exact": sum(y and not x for x, y in zip(a_r, a_e, strict=True)),
        "strength_rank_correlation": rho,
        "median_strength_ratio_realistic_to_exact": float(np.median(ratio)) if len(ratio) else float("nan"),
    }  # fmt: skip


def load_exact_world(data: str, split: str = "train", n_pos: int = 2000, neg_ratio: int = 4, seed: int = 0,
                     root: str = ".", max_banks: int | None = None):  # fmt: skip
    """The real data version: transactions through `ftm.data.io` (labels included, so train or val only), the bank views
    that exist, and a `RealWorld` whose alerts are the stratified sample."""
    from pathlib import Path

    from ftm.backtest.world import LEDGER_COLUMNS, RealWorld, _digest
    from ftm.banks.partner import BankConfig, PartnerBank
    from ftm.data import io

    if split not in ("train", "val"):
        raise ValueError("exact-mode back-testing runs on train or val only; the test split is locked")
    cfg = io.load_config(data)
    root = Path(root)
    pq, views = root / cfg["parquet_dir"], root / cfg["views_dir"]
    banks_tbl = io.read_banks(views)
    bank_ids = [int(b) for b in banks_tbl["bank"].to_list()][:max_banks]
    ledgers = {b: io.scan_view(views, b, split=split, columns=LEDGER_COLUMNS).collect() for b in bank_ids}
    tx = io.scan_transactions(pq, split=split, columns=[*TX_COLUMNS, "is_laundering"]).collect()
    alerts = exact_alerts(tx.select(TX_COLUMNS), tx["is_laundering"], ledgers, n_pos, neg_ratio, seed, split)
    partners = {a.binding.partner_bank for a in alerts}
    banks = {p: PartnerBank(p, ledgers[p], BankConfig.load(), seed) for p in sorted(partners)}
    return RealWorld(banks, alerts, f"{data}-{split}-exact-{len(alerts)}-{_digest(alerts)}", 0, ledgers)
