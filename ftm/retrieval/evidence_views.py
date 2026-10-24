"""C11 evidence views for one alert: top-40 by amount, scorer only, scorer + PCST (hand-off #11, T1 owns from week 4).

Owner: T1 (Abhishek).  Spec: plan.md 6.12, tracks.md 3.8.  New file: T3's week-2 stub (`views.py`, PR #6) is untouched.

For an alert (home bank b, transaction, `as_of` = alert time) the candidates are the edges of view(b) in
[as_of - 72 h, as_of], read through `ftm.data.io` with `as_of` (time guard, no test rows). Every view returns the
K = 40 tx_ids (always including the alerted transaction); `format_trail` turns them into one time-ordered trail
with aliased account ids (`ftm.data.alias`), which is the only thing an LLM may see.

Stable API for T3 (E8 / H6): `view(alert, name, k=40, data=...)` -> `EvidenceView(text, tx_ids, ...)`, name in `NAMES`.

Evaluation (`python -m ftm.retrieval.evidence_views --data hi_small`) uses labels only to score the views:
a pattern edge = a candidate edge of the same laundering attempt (`patterns.parquet`) as the alerted transaction.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl

from ftm import guards
from ftm.data import alias, io
from ftm.retrieval import pcst
from ftm.retrieval import views as stub_views

K = 40
WINDOW_HOURS = 72
COLS = ["tx_id", "timestamp", "src_id", "dst_id", "from_bank", "to_bank", "amount_paid", "payment_currency", "payment_format"]
TRAIL_COLS = ["tx_id", "timestamp", "src_alias", "dst_alias", "from_bank", "to_bank", "amount_paid", "payment_currency", "payment_format"]
VIEWS = ("amount", "amount_1hop", "scorer", "scorer_greedy", "scorer_pcst", "long_prompt")
# The four views of E8 under the names T3 calls: top-40 by amount, scorer only, scorer + time-respecting PCST trail, everything.
NAMES = ("amount", "scorer", "scorer_pcst", "long_prompt", "amount_1hop", "scorer_greedy")
# "Everything in a long prompt": all candidate edges of the 72 h window, serialised like the other views, newest kept when they
# do not fit. The harness prompt cap is `Limits.max_tokens` = 6,000 (configs/llm/default.yaml; the server has an 8,192 context).
# The instructions, the alert, the question menu and the answer history of T3's planner prompt take about 2,000 of those, so
# the evidence gets 4,000 tokens. Tokens are estimated at 1.7 characters per token (aliases and amounts tokenise badly; checked
# against the Qwen3 tokenizer in docs/t1_planner_view.md), the same rule T3's trim loop uses.
HARNESS_PROMPT_TOKENS = 6000
PROMPT_RESERVE_TOKENS = 2000
CHARS_PER_TOKEN = 1.7
EVIDENCE_TOKENS = HARNESS_PROMPT_TOKENS - PROMPT_RESERVE_TOKENS


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


def line_chars(cand: pl.DataFrame, alert: int, home_bank: int | None) -> np.ndarray:
    """Characters of each candidate's trail line (`views.format_trail`, newline included): lets the long-prompt view be cut
    without formatting 17,000 rows. Exact (a test compares it with the real lines)."""
    out = np.empty(cand.height, dtype=np.int64)
    for i, (a, cur, fmt, fb, tb) in enumerate(zip(cand["amount_paid"], cand["payment_currency"], cand["payment_format"],
                                                   cand["from_bank"], cand["to_bank"], strict=True)):  # fmt: skip
        suffix = 0 if home_bank is None or fb == tb == home_bank else 11  # " (outgoing)" and " (incoming)" are both 11
        out[i] = 16 + 3 + 8 + 4 + 8 + 3 + len(f"{a:,.2f}") + 1 + len(cur) + 3 + len(fmt) + 3 + len(f"bank {fb} -> bank {tb}") + suffix + 1
    out[alert] += 9  # " <- ALERT"
    return out


def long_prompt_mask(cand: pl.DataFrame, alert: int, home_bank: int | None, max_tokens: int = EVIDENCE_TOKENS) -> np.ndarray:
    """All candidate edges, newest first, until `max_tokens` (at `CHARS_PER_TOKEN`) is used; the alerted edge is always in."""
    chars = line_chars(cand, alert, home_bank)
    budget = int(max_tokens * CHARS_PER_TOKEN) - chars[alert]
    m = np.zeros(cand.height, bool)
    m[alert] = True
    for i in range(cand.height - 1, -1, -1):  # candidates are sorted by (timestamp, tx_id): newest last
        if i == alert:
            continue
        if chars[i] > budget:
            break
        budget -= chars[i]
        m[i] = True
    return m


def select_views(cand: pl.DataFrame, alert: int, prize: np.ndarray, k: int = K, edge_cost: float = 0.05,
                 home_bank: int | None = None, max_tokens: int = EVIDENCE_TOKENS, only: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:  # fmt: skip
    """{view name: bool mask over `cand`} for the alerted row index `alert` (only the views in `only`, default all)."""
    tx, src, dst = cand["tx_id"].to_numpy(), cand["src_id"].to_numpy(), cand["dst_id"].to_numpy()
    t = cand["timestamp"].dt.epoch("s").to_numpy().astype("f8")
    amt = cand["amount_paid"].to_numpy()
    hop = np.isin(src, [src[alert], dst[alert]]) | np.isin(dst, [src[alert], dst[alert]])
    pool = pcst.reachable_to_alert(src, dst, t, alert) if (only is None or "scorer_greedy" in only) else None
    make = {
        "amount": lambda: _top(amt, tx, alert, k),
        "amount_1hop": lambda: _top(amt, tx, alert, k, hop),  # T3's stub definition (1 hop of sender/receiver), own implementation
        "scorer": lambda: _top(prize, tx, alert, k),
        "scorer_greedy": lambda: pcst.greedy_trail(src, dst, t, prize, alert, k, allowed=pool),  # ablation: trail without PCST
        "scorer_pcst": lambda: pcst.select_pcst_trail(src, dst, t, prize, alert, k, edge_cost)[0],
        "long_prompt": lambda: long_prompt_mask(cand, alert, home_bank, max_tokens),
    }
    return {n: f() for n, f in make.items() if only is None or n in only}


def format_trail(cand: pl.DataFrame, mask: np.ndarray, aliases: pl.DataFrame) -> pl.DataFrame:
    """The selected edges as one time-ordered trail with aliased ids (no raw account, no label, no node_id)."""
    sel = cand.filter(pl.Series(mask)).sort(["timestamp", "tx_id"])
    return alias.apply_aliases(sel, aliases).select(TRAIL_COLS)


@dataclass(frozen=True)
class EvidenceView:
    """What `view` returns. `text` is the aliased, time-ordered trail (one line per edge, the alerted edge marked) that goes into
    a prompt; `tx_ids` are the same edges' transaction ids in the same order (server side: bookkeeping and scoring, never in a
    prompt). No label, no raw account id. `tokens_est` = len(text) / CHARS_PER_TOKEN."""

    name: str
    text: str
    tx_ids: list[int]
    n_candidates: int  # edges of the home bank's view in [as_of - 72 h, as_of]
    tokens_est: int


@lru_cache(maxsize=4)
def _scores(ds: str) -> pl.DataFrame:
    f = Path("data/retrieval") / ds / "edge_scores.parquet"
    if not f.exists():
        raise FileNotFoundError(f"{f}: train the edge scorer first (python -m ftm.retrieval.edge_scorer --data {ds} --train)")
    return pl.read_parquet(f, columns=["tx_id", "score"])  # time-guard: not transactions


@lru_cache(maxsize=4)
def _alert_rows(ds: str) -> dict:
    cfg = io.load_config(ds)
    al = pl.concat([pl.read_parquet(Path(cfg["alerts_dir"]) / f"{s}.parquet", columns=["alert_id", "tx_id", "bank", "timestamp"])  # time-guard: not transactions
                    for s in ("train", "val")])  # fmt: skip
    return {r["alert_id"]: r for r in al.iter_rows(named=True)}


@lru_cache(maxsize=4)
def _aliases(parquet_dir: str, seed: int) -> pl.DataFrame:
    return alias.read_aliases(parquet_dir, seed)


def resolve_alert(alert, ds: str) -> tuple[int, int, datetime]:
    """(tx_id, home bank, as_of) of an alert given as an `AlertContext`, an alert id, or a dict with `tx_id`, `bank`, `timestamp`.
    Alert ids are looked up in `data/alerts/<ds>/{train,val}.parquet`, which is the source of truth for the alert time."""
    if isinstance(alert, dict) and {"tx_id", "bank", "timestamp"} <= alert.keys():
        return int(alert["tx_id"]), int(alert["bank"]), alert["timestamp"]
    aid = alert if isinstance(alert, str) else (alert["alert_id"] if isinstance(alert, dict) else alert.alert_id)
    row = _alert_rows(ds).get(aid)
    if row is None:
        raise KeyError(f"alert {aid!r} is not in the {ds} train/val alert sets")
    as_of = getattr(alert, "as_of", None)
    if as_of is not None and datetime.fromisoformat(as_of) != row["timestamp"]:
        raise ValueError(f"alert {aid}: as_of {as_of} is not the alert's time {row['timestamp']}")
    return int(row["tx_id"]), int(row["bank"]), row["timestamp"]


def view(alert, name: str = "scorer_pcst", k: int = K, *, data: str = "hi_small", alias_seed: int = 0,
         max_evidence_tokens: int = EVIDENCE_TOKENS) -> EvidenceView:  # fmt: skip
    """The evidence of one alert under one view (the stable API for T3, E8 / H6).

    name: `amount` (top-k by amount over the whole window), `scorer` (top-k by edge-scorer score), `scorer_pcst` (the scorer's
    edges as a time-respecting PCST trail connected to the alerted edge, at most k), `long_prompt` (every candidate edge,
    newest kept until `max_evidence_tokens`; k is ignored), plus the ablations `amount_1hop` (the week-2 stub's definition) and
    `scorer_greedy`. The candidates are the home bank's edges in [as_of - 72 h, as_of], read through `ftm.data.io` (time
    guard, train/val only); nothing later than the alert is read, scored or returned. Accounts are aliased with
    `alias_seed` (the run seed). Raises FileNotFoundError for the scorer views when the scorer has not been trained.
    """
    if name not in NAMES:
        raise ValueError(f"unknown view {name!r}; choose one of {NAMES}")
    cfg = io.load_config(data)
    tx_id, bank, as_of = resolve_alert(alert, data)
    cand = candidates(cfg, bank, as_of)
    hit = np.flatnonzero(cand["tx_id"].to_numpy() == tx_id)
    if not len(hit):
        raise ValueError(f"alerted transaction {tx_id} is not in view({bank}) up to {as_of}")
    i = int(hit[0])
    pr = prizes(cand, _scores(data)) if name.startswith("scorer") else prizes(cand, None)
    mask = select_views(cand, i, pr, k, home_bank=bank, max_tokens=max_evidence_tokens, only=(name,))[name]
    sel = cand.filter(pl.Series(mask)).sort(["timestamp", "tx_id"])
    guards.assert_not_future(sel["timestamp"].max(), as_of)
    rows = format_trail(cand, mask, _aliases(cfg["parquet_dir"], alias_seed))
    text = stub_views.format_trail(rows, alert_tx_id=tx_id, home_bank=bank)
    return EvidenceView(name, text, sel["tx_id"].to_list(), cand.height, int(np.ceil(len(text) / CHARS_PER_TOKEN)))


H6_BASELINES = ("amount_1hop", "amount")  # the week-2 stub's top-40 by amount (1 hop), and the plain top-40 by amount
H6_VS = "scorer_pcst"
H6_MARGIN = 0.15


def h6_report(per_alert: pl.DataFrame, n_boot: int = 10_000, seed: int = 0, k: int = K) -> dict:
    """H6 (tracks.md 3.8): does scorer + PCST trail find >= 15 points more real pattern edges (recall in the 40) than the
    top-40-by-amount baseline? Paired over the alerts that have pattern edges in the window, 95 % percentile CI from a
    bootstrap over alerts and, since several alerts of one attempt are not independent, over attempts (alerts with no attempt
    count as their own cluster). Positive = scorer + PCST finds more. Reported for all such alerts and for the stratum where
    the baseline's 1-hop pool has at least k edges, the only place where choosing among edges can matter."""
    has = per_alert.filter(pl.col("n_relevant") > 0)
    wide = has.with_columns((pl.col("hits") / pl.col("n_relevant")).alias("recall")).pivot(on="view", index=["alert_id", "attempt_id"], values="recall")
    wide = wide.join(has.pivot(on="view", index="alert_id", values="hits"), on="alert_id", suffix="_hits")
    wide = wide.join(has.filter(pl.col("view") == "amount_1hop").select("alert_id", pl.col("n_sel").alias("pool_1hop")), on="alert_id")
    return {"margin": H6_MARGIN, "n_boot": n_boot, "all": _h6_block(wide, n_boot, seed),
            "pool_1hop_at_least_k": _h6_block(wide.filter(pl.col("pool_1hop") >= k), n_boot, seed)}


def _h6_block(wide: pl.DataFrame, n_boot: int, seed: int) -> dict:
    from ftm.eval import bootstrap as bs

    out = {"alerts_with_pattern_edges": wide.height, "comparisons": {}}
    if wide.height < 2:
        return out
    att = [a if a is not None else f"alert:{i}" for i, a in zip(wide["alert_id"], wide["attempt_id"], strict=True)]
    out["attempts"] = int(len(set(att)))
    cl_attempt, cl_alert = bs.cluster_ids(att, []), np.arange(wide.height)
    for other in (*H6_BASELINES, "scorer", "scorer_greedy", "long_prompt"):
        for kind, col in (("recall", other), ("pattern_edges", f"{other}_hits")):
            d = (wide[H6_VS if kind == "recall" else f"{H6_VS}_hits"] - wide[col]).to_numpy()
            res = {"diff": float(d.mean())}
            for cname, cl in (("alerts", cl_alert), ("attempts", cl_attempt)):
                b = bs.bootstrap_means(d[:, None], cl, n_boot=n_boot, seed=seed)[:, 0]
                res[f"ci95_{cname}"] = [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]
            if kind == "recall":
                res["meets_margin"] = bool(res["diff"] >= H6_MARGIN)
                res["ci_lower_above_margin_attempts"] = bool(res["ci95_attempts"][0] >= H6_MARGIN)
            out["comparisons"][f"{H6_VS}_minus_{other}_{kind}"] = res
    return out


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
        masks = select_views(cand, i, pr, k, edge_cost, home_bank=int(r["bank"]))
        tx, src, dst = cand["tx_id"].to_numpy(), cand["src_id"].to_numpy(), cand["dst_id"].to_numpy()
        t = cand["timestamp"].dt.epoch("s").to_numpy().astype("f8")
        rel = np.isin(tx, list(by_att.get(r["attempt_id"], ()))) if r["attempt_id"] is not None else np.zeros(len(tx), bool)
        rel[i] = False
        reach = pcst.reachable_to_alert(src, dst, t, i)
        lc = line_chars(cand, i, int(r["bank"]))
        base = {"alert_id": r["alert_id"], "attempt_id": r["attempt_id"], "label": bool(r["label"]), "n_cand": cand.height,
                "n_relevant": int(rel.sum()), "n_relevant_reachable": int((rel & reach).sum())}  # fmt: skip
        for name, m in masks.items():
            assert (t[m] <= t[i]).all()  # nothing after as_of
            rows.append(base | {"view": name, "n_sel": int(m.sum()), "hits": int((m & rel).sum()), "chars": int(lc[m].sum()),
                                "connected": pcst.is_connected(src, dst, m), "trail": pcst.is_one_trail(src, dst, t, i, m)})  # fmt: skip
    df = pl.DataFrame(rows)
    has = df.filter(pl.col("n_relevant") > 0)
    summ = []
    for name in VIEWS:
        d, h = df.filter(pl.col("view") == name), has.filter(pl.col("view") == name)
        summ.append({"view": name, "alerts": d.height, "alerts_with_pattern_edges": h.height,
                     "recall_at_k": float((h["hits"] / h["n_relevant"]).mean()), "mean_pattern_edges_in_k": float(h["hits"].mean()),
                     "connected": float(d["connected"].mean()), "one_time_respecting_trail": float(d["trail"].mean()),
                     "mean_selected": float(d["n_sel"].mean()), "mean_tokens_est": float(d["chars"].mean() / CHARS_PER_TOKEN),
                     "max_tokens_est": float(d["chars"].max() / CHARS_PER_TOKEN)})  # fmt: skip
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
    res["h6"] = h6_report(df)
    tag = f"{a.data}_{a.prize}_{'_'.join(a.parts)}"
    Path("data/runs").mkdir(exist_ok=True)
    df.write_parquet(f"data/runs/evidence_views_{tag}.parquet")
    Path(f"data/runs/evidence_views_{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
