"""C11 evidence views for one alert: top-40 by amount, scorer only, scorer + PCST (hand-off #11, T1 owns from week 4).

Owner: T1 (Abhishek).  Spec: plan.md 6.12, tracks.md 3.8.  New file: T3's week-2 stub (`views.py`, PR #6) is untouched.

For an alert (home bank b, transaction, `as_of` = alert time) the candidates are the edges of view(b) in
[as_of - 72 h, as_of], read through `ftm.data.io` with `as_of` (time guard, no test rows). Every view returns the
K = 40 tx_ids (always including the alerted transaction); `format_trail` turns them into one time-ordered trail
with aliased account ids (`ftm.data.alias`), which is the only thing an LLM may see.

Evaluation (`python -m ftm.retrieval.evidence_views --data hi_small`) uses labels only to score the views:
a pattern edge = a candidate edge of the same laundering attempt (`patterns.parquet`) as the alerted transaction.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ftm.data import alias, io
from ftm.retrieval import pcst

K = 40
WINDOW_HOURS = 72
COLS = ["tx_id", "timestamp", "src_id", "dst_id", "from_bank", "to_bank", "amount_paid", "payment_currency", "payment_format"]
TRAIL_COLS = ["tx_id", "timestamp", "src_alias", "dst_alias", "from_bank", "to_bank", "amount_paid", "payment_currency", "payment_format"]
VIEWS = ("amount", "amount_1hop", "scorer", "scorer_greedy", "scorer_pcst")


def candidates(cfg: dict, bank: int, as_of, window_hours: float = WINDOW_HOURS) -> pl.DataFrame:
    """View(bank) edges in [as_of - window, as_of], time-sorted. Reads only train/val rows with timestamp <= as_of."""
    as_of = datetime.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    lf = io.scan_view(cfg["views_dir"], bank, split=["train", "val"], as_of=as_of, columns=COLS)
    return lf.filter(pl.col("timestamp") >= as_of - timedelta(hours=window_hours)).sort(["timestamp", "tx_id"]).collect()


def prizes(cand: pl.DataFrame, scores: pl.DataFrame | None) -> np.ndarray:
    """Edge scorer score in [0, 1]; without scores, log amount scaled to [0, 1] (the fallback)."""
    if scores is None:
        a = np.log1p(cand["amount_paid"].to_numpy())
        return a / max(a.max(), 1e-9)
    return cand.join(scores.select("tx_id", "score"), on="tx_id", how="left")["score"].fill_null(0.0).to_numpy()


def _top(order_key: np.ndarray, tx: np.ndarray, alert: int, k: int, pool: np.ndarray | None = None) -> np.ndarray:
    """Mask: the alerted edge plus the k-1 largest `order_key` among `pool` (ties: lower tx_id)."""
    ok = np.ones(len(tx), bool) if pool is None else pool.copy()
    ok[alert] = False
    idx = np.flatnonzero(ok)
    idx = idx[np.lexsort((tx[idx], -order_key[idx]))][: k - 1]
    m = np.zeros(len(tx), bool)
    m[idx] = True
    m[alert] = True
    return m


def select_views(cand: pl.DataFrame, alert: int, prize: np.ndarray, k: int = K, edge_cost: float = 0.05) -> dict[str, np.ndarray]:
    """{view name: bool mask over `cand`} for the alerted row index `alert`."""
    tx, src, dst = cand["tx_id"].to_numpy(), cand["src_id"].to_numpy(), cand["dst_id"].to_numpy()
    t = cand["timestamp"].dt.epoch("s").to_numpy().astype("f8")
    amt = cand["amount_paid"].to_numpy()
    hop = np.isin(src, [src[alert], dst[alert]]) | np.isin(dst, [src[alert], dst[alert]])
    pool = pcst.reachable_to_alert(src, dst, t, alert)
    out = {
        "amount": _top(amt, tx, alert, k),
        "amount_1hop": _top(amt, tx, alert, k, hop),  # T3's stub definition (1 hop of sender/receiver), own implementation
        "scorer": _top(prize, tx, alert, k),
        "scorer_greedy": pcst.greedy_trail(src, dst, t, prize, alert, k, allowed=pool),  # ablation: trail without PCST
    }
    out["scorer_pcst"], _ = pcst.select_pcst_trail(src, dst, t, prize, alert, k, edge_cost)
    return out


def format_trail(cand: pl.DataFrame, mask: np.ndarray, aliases: pl.DataFrame) -> pl.DataFrame:
    """The selected edges as one time-ordered trail with aliased ids (no raw account, no label, no node_id)."""
    sel = cand.filter(pl.Series(mask)).sort(["timestamp", "tx_id"])
    return alias.apply_aliases(sel, aliases).select(TRAIL_COLS)


def evaluate(ds: str, parts=("eval",), edge_cost: float = 0.05, prize_source: str = "scorer", k: int = K, limit: int | None = None) -> dict:
    """Pattern-edge recall among the K and trail checks over the val alerts of `parts` (data/alerts/<ds>/val.parquet)."""
    cfg = io.load_config(ds)
    pq = cfg["parquet_dir"]
    al = pl.read_parquet(Path(cfg["alerts_dir"]) / "val.parquet").filter(pl.col("val_part").is_in(list(parts)))  # time-guard: not transactions
    if limit:
        al = al.head(limit)
    scores = pl.read_parquet(Path("data/retrieval") / ds / "edge_scores.parquet") if prize_source == "scorer" else None  # time-guard: not transactions
    pat = io.scan_patterns(pq, split=["train", "val"]).select("tx_id", "attempt_id").collect()
    by_att = {a: set(g["tx_id"].to_list()) for (a,), g in pat.group_by("attempt_id")}
    rows, t0 = [], time.time()
    for r in al.iter_rows(named=True):
        cand = candidates(cfg, r["bank"], r["timestamp"])
        i = int(np.flatnonzero(cand["tx_id"].to_numpy() == r["tx_id"])[0])
        pr = prizes(cand, scores)
        masks = select_views(cand, i, pr, k, edge_cost)
        tx, src, dst = cand["tx_id"].to_numpy(), cand["src_id"].to_numpy(), cand["dst_id"].to_numpy()
        t = cand["timestamp"].dt.epoch("s").to_numpy().astype("f8")
        rel = np.isin(tx, list(by_att.get(r["attempt_id"], ()))) if r["attempt_id"] is not None else np.zeros(len(tx), bool)
        rel[i] = False
        reach = pcst.reachable_to_alert(src, dst, t, i)
        base = {"alert_id": r["alert_id"], "label": bool(r["label"]), "n_cand": cand.height, "n_relevant": int(rel.sum()), "n_relevant_reachable": int((rel & reach).sum())}
        for name, m in masks.items():
            assert (t[m] <= t[i]).all()  # nothing after as_of
            rows.append(base | {"view": name, "n_sel": int(m.sum()), "hits": int((m & rel).sum()),
                                "connected": pcst.is_connected(src, dst, m), "trail": pcst.is_one_trail(src, dst, t, i, m)})  # fmt: skip
    df = pl.DataFrame(rows)
    has = df.filter(pl.col("n_relevant") > 0)
    summ = []
    for name in VIEWS:
        d, h = df.filter(pl.col("view") == name), has.filter(pl.col("view") == name)
        summ.append({"view": name, "alerts": d.height, "alerts_with_pattern_edges": h.height,
                     "recall_at_k": float((h["hits"] / h["n_relevant"]).mean()), "mean_pattern_edges_in_k": float(h["hits"].mean()),
                     "connected": float(d["connected"].mean()), "one_time_respecting_trail": float(d["trail"].mean()),
                     "mean_selected": float(d["n_sel"].mean())})  # fmt: skip
    first = df.filter(pl.col("view") == "amount")
    out = {"dataset": ds, "parts": list(parts), "prize_source": prize_source, "edge_cost": edge_cost, "k": k, "window_hours": WINDOW_HOURS,
           "alerts": al.height, "alerts_in_attempt": int(al["attempt_id"].is_not_null().sum()),
           "mean_candidates": float(first["n_cand"].mean()), "median_candidates": float(first["n_cand"].median()),
           "mean_relevant_in_window": float(first.filter(pl.col("n_relevant") > 0)["n_relevant"].mean()),
           "relevant_reachable_share": float(first["n_relevant_reachable"].sum() / max(first["n_relevant"].sum(), 1)),
           "views": summ, "seconds": time.time() - t0}  # fmt: skip
    return out | {"_per_alert": df}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--parts", nargs="+", default=["eval"])
    ap.add_argument("--prize", choices=["scorer", "amount"], default="scorer")
    ap.add_argument("--edge-cost", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    res = evaluate(a.data, tuple(a.parts), a.edge_cost, a.prize, limit=a.limit)
    df = res.pop("_per_alert")
    tag = f"{a.data}_{a.prize}_{'_'.join(a.parts)}"
    Path("data/runs").mkdir(exist_ok=True)
    df.write_parquet(f"data/runs/evidence_views_{tag}.parquet")
    Path(f"data/runs/evidence_views_{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
