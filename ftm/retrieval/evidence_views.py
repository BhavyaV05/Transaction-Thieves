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
from ftm.retrieval import explainers, pcst
from ftm.retrieval import views as stub_views

K = 40
WINDOW_HOURS = 72
COLS = ["tx_id", "timestamp", "src_id", "dst_id", "from_bank", "to_bank", "amount_paid", "payment_currency", "payment_format"]
TRAIL_COLS = ["tx_id", "timestamp", "src_alias", "dst_alias", "from_bank", "to_bank", "amount_paid", "payment_currency", "payment_format"]
VIEWS = ("amount", "amount_1hop", "scorer", "scorer_greedy", "scorer_pcst", "long_prompt", "ppr", "gnnexplainer", "pgexplainer")
# The four views of E8 under the names T3 calls: top-40 by amount, scorer only, scorer + time-respecting PCST trail, everything;
# then the ablations and the three explainer comparisons (personalised PageRank, GNNExplainer, PGExplainer: `ftm.retrieval.explainers`).
NAMES = ("amount", "scorer", "scorer_pcst", "long_prompt", "amount_1hop", "scorer_greedy", "ppr", "gnnexplainer", "pgexplainer")
EXPLAINER_VIEWS = ("gnnexplainer", "pgexplainer")  # scores come from the Multi-GNN one-bank model (cache or live), not from the candidates alone
# "Everything in a long prompt": all candidate edges of the 72 h window, serialised like the other views, newest kept when they
# do not fit. The harness prompt cap is `Limits.max_tokens` = 6,000 (configs/llm/default.yaml; the server has an 8,192 context).
# The instructions, the alert, the question menu and the answer history of T3's planner prompt take about 2,000 of those, so
# the evidence gets 4,000 tokens. Tokens are estimated at 1.5 characters per token: the Qwen3 tokenizer gives 1.54 to 1.62 on these lines (aliases and
# amounts tokenise badly; docs/t1_planner_view.md), so the estimate errs on the safe side by 3 to 8 percent.
HARNESS_PROMPT_TOKENS = 6000
PROMPT_RESERVE_TOKENS = 2000
CHARS_PER_TOKEN = 1.5
EVIDENCE_TOKENS = HARNESS_PROMPT_TOKENS - PROMPT_RESERVE_TOKENS


def splits_for(final: bool) -> list[str]:
    """Splits the evidence may come from: train and val; the final run (week 14, TEST alerts) adds test (locked: FTM_FINAL_RUN=1)."""
    return ["train", "val", "test"] if final else ["train", "val"]


def candidates(cfg: dict, bank: int, as_of, window_hours: float = WINDOW_HOURS, final: bool = False) -> pl.DataFrame:
    """View(bank) edges in [as_of - window, as_of], time-sorted. Reads only train/val rows (test too when `final`) with timestamp <= as_of."""
    as_of = datetime.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    lf = io.scan_view(cfg["views_dir"], bank, split=splits_for(final), as_of=as_of, columns=COLS)
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


def _positive_top(score: np.ndarray, tx: np.ndarray, alert: int, k: int) -> np.ndarray:
    """The alerted edge plus the k-1 highest `score` among the edges with score > 0."""
    return _top(score, tx, alert, k, np.asarray(score) > 0)


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
                 home_bank: int | None = None, max_tokens: int = EVIDENCE_TOKENS, only: tuple[str, ...] | None = None,
                 explain: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:  # fmt: skip
    """{view name: bool mask over `cand`} for the alerted row index `alert` (only the views in `only`, default all).

    `explain` = {explainer view name: score per candidate edge}; the explainer views exist only for the names it holds. Every
    score-ranked selector (`ppr` and the explainers) takes the k-1 best edges with a score above 0, so an edge the selector never saw
    (outside the alerted edge's component, or outside the model's receptive field) is not padded in."""
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
        "ppr": lambda: _positive_top(explainers.ppr_edge_scores(src, dst, alert), tx, alert, k),
    }
    for name, sc in (explain or {}).items():
        make[name] = lambda sc=sc: _positive_top(sc, tx, alert, k)
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
def _scores(ds: str, final: bool = False) -> pl.DataFrame:
    f = Path("data/retrieval") / ds / "edge_scores.parquet"
    if not f.exists():
        raise FileNotFoundError(f"{f}: train the edge scorer first (python -m ftm.retrieval.edge_scorer --data {ds} --train)")
    out = pl.read_parquet(f, columns=["tx_id", "score"])  # time-guard: not transactions
    if final:  # the test edges, scored by `edge_scorer --final`
        t = Path("data/retrieval") / ds / "final" / "edge_scores_test.parquet"
        if not t.exists():
            raise FileNotFoundError(f"{t}: run `python -m ftm.retrieval.edge_scorer --data {ds} --final` first")
        out = pl.concat([out, pl.read_parquet(t, columns=["tx_id", "score"])])  # time-guard: not transactions
    return out


@lru_cache(maxsize=8)
def _explainer_cache(ds: str, method: str, final: bool = False) -> dict:
    """{(bank, alerted tx_id): {candidate tx_id: score}} written by `scripts/run_explainers.py` (empty when nothing is cached)."""
    d = Path("data/retrieval") / ds / ("explainers_final" if final else "explainers") / method
    fs = sorted(d.glob("bank*.parquet"))
    out: dict = {}
    for f in fs:
        for (bank, alert_tx), g in pl.read_parquet(f, columns=["bank", "alert_tx", "tx_id", "score"]).group_by(["bank", "alert_tx"]):  # time-guard: not transactions
            out[(bank, alert_tx)] = dict(zip(g["tx_id"].to_list(), g["score"].to_list(), strict=True))
    return out


def explainer_scores(ds: str, method: str, bank: int, tx_id: int, as_of, cand: pl.DataFrame, final: bool = False) -> np.ndarray:
    """Score per candidate edge of the alert under an explainer view: the cached one if `run_explainers.py` explained this alert, else
    computed now (needs the Multi-GNN clone, the bank's checkpoint and, for PGExplainer, its trained weights)."""
    hit = _explainer_cache(ds, method, final).get((bank, tx_id))
    if hit is not None:
        return np.array([hit.get(int(t), 0.0) for t in cand["tx_id"].to_numpy()])
    return explainers.live_scores(ds, method, bank, tx_id, as_of, cand["tx_id"].to_numpy(), final=final)


@lru_cache(maxsize=4)
def _alert_rows(ds: str, final: bool = False) -> dict:
    cfg = io.load_config(ds)
    if final:
        guards.require_final_run("the test alert set")
    al = pl.concat([pl.read_parquet(Path(cfg["alerts_dir"]) / f"{s}.parquet", columns=["alert_id", "tx_id", "bank", "timestamp"])  # time-guard: not transactions
                    for s in (("train", "val", "test") if final else ("train", "val"))])  # fmt: skip
    return {r["alert_id"]: r for r in al.iter_rows(named=True)}


@lru_cache(maxsize=4)
def _aliases(parquet_dir: str, seed: int) -> pl.DataFrame:
    return alias.read_aliases(parquet_dir, seed)


def is_final_alert(alert) -> bool:
    """Test alerts have ids `F000123` (`ftm.intake.alerts.test_alert_ids`); only they are read from the test split."""
    aid = alert if isinstance(alert, str) else (alert.get("alert_id") if isinstance(alert, dict) else getattr(alert, "alert_id", None))
    return isinstance(aid, str) and aid.startswith("F")


def resolve_alert(alert, ds: str, final: bool = False) -> tuple[int, int, datetime]:
    """(tx_id, home bank, as_of) of an alert given as an `AlertContext`, an alert id, or a dict with `tx_id`, `bank`, `timestamp`.
    Alert ids are looked up in `data/alerts/<ds>/{train,val}.parquet`, which is the source of truth for the alert time."""
    if isinstance(alert, dict) and {"tx_id", "bank", "timestamp"} <= alert.keys():
        return int(alert["tx_id"]), int(alert["bank"]), alert["timestamp"]
    aid = alert if isinstance(alert, str) else (alert["alert_id"] if isinstance(alert, dict) else alert.alert_id)
    row = _alert_rows(ds, final).get(aid)
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
    newest kept until `max_evidence_tokens`; k is ignored), the ablations `amount_1hop` (the week-2 stub's definition) and
    `scorer_greedy`, and the explainer comparisons `ppr` (personalised PageRank from the alerted edge's two accounts, undirected),
    `gnnexplainer` and `pgexplainer` (edge masks of the home bank's Multi-GNN one-bank model for the class laundering;
    `ftm.retrieval.explainers`). The candidates are the home bank's edges in [as_of - 72 h, as_of], read through `ftm.data.io` (time
    guard, train/val only); nothing later than the alert is read, scored or returned. Accounts are aliased with
    `alias_seed` (the run seed). Raises FileNotFoundError for the scorer views when the scorer has not been trained.
    """
    if name not in NAMES:
        raise ValueError(f"unknown view {name!r}; choose one of {NAMES}")
    cfg = io.load_config(data)
    final = is_final_alert(alert)  # a test alert (week 14): its evidence may also come from test days up to its time
    tx_id, bank, as_of = resolve_alert(alert, data, final)
    cand = candidates(cfg, bank, as_of, final=final)
    hit = np.flatnonzero(cand["tx_id"].to_numpy() == tx_id)
    if not len(hit):
        raise ValueError(f"alerted transaction {tx_id} is not in view({bank}) up to {as_of}")
    i = int(hit[0])
    pr = prizes(cand, _scores(data, final)) if name.startswith("scorer") else prizes(cand, None)
    explain = {name: explainer_scores(data, name, bank, tx_id, as_of, cand, final)} if name in EXPLAINER_VIEWS else None
    mask = select_views(cand, i, pr, k, home_bank=bank, max_tokens=max_evidence_tokens, only=(name,), explain=explain)[name]
    sel = cand.filter(pl.Series(mask)).sort(["timestamp", "tx_id"])
    guards.assert_not_future(sel["timestamp"].max(), as_of)
    rows = format_trail(cand, mask, _aliases(cfg["parquet_dir"], alias_seed))
    text = stub_views.format_trail(rows, alert_tx_id=tx_id, home_bank=bank)
    return EvidenceView(name, text, sel["tx_id"].to_list(), cand.height, int(np.ceil(len(text) / CHARS_PER_TOKEN)))


H6_BASELINES = ("amount_1hop", "amount")  # the week-2 stub's top-40 by amount (1 hop), and the plain top-40 by amount
H6_VS = "scorer_pcst"
H6_MARGIN = 0.15


def h6_report(per_alert: pl.DataFrame, n_boot: int = 10_000, seed: int = 0, k: int = K) -> dict:
    """H6 as pre-registered (`prereg/hypotheses.yaml`): `pattern_edge_share(scorer_pcst) - pattern_edge_share(top40_by_amount)` with the
    rejection rule "share-difference CI lower bound >= 0.15 AND Brier CI upper bound < 0". The Brier half needs planner runs with
    each view (T3's E8) and is not computed here.

    `pattern_edge_share` = pattern edges in the view / edges counted. "Among the 40" is read two ways and both are reported
    (`denominator`): **k** (40, so a view that returns 15 edges is not rewarded for being short; the reading used for the
    verdict) and **selected** (the number of edges the view returned). The baseline is `amount`, the plain top-40 by amount
    over the window (not the 1-hop stub, which is a secondary comparison). Two populations: the alerts with at least one
    pattern edge in the window (the only ones where a share can be above 0) and all alerts. Paired, 95 % percentile CI from
    `n_boot` bootstrap resamples over alerts and over attempts (alerts with no attempt are their own cluster).

    Secondary context (`recall_secondary`): recall of the pattern edges in the window, scorer + PCST against the stub, the
    scorer alone, the greedy trail and the long prompt, on the alerts with pattern edges, and the same in the stratum where the
    stub's 1-hop pool has at least k edges."""
    from ftm.eval import bootstrap as bs

    def paired(att, d):
        cl = {"alerts": np.arange(len(d)), "attempts": bs.cluster_ids(att, [])}
        out = {"diff": float(d.mean())}
        for cname, c in cl.items():
            b = bs.bootstrap_means(d[:, None], c, n_boot=n_boot, seed=seed)[:, 0]
            out[f"ci95_{cname}"] = [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]
        return out

    hits = per_alert.pivot(on="view", index=["alert_id", "attempt_id", "n_relevant"], values="hits")
    present = [v for v in VIEWS if v in set(per_alert["view"])]
    nsel = per_alert.pivot(on="view", index="alert_id", values="n_sel").rename({v: f"{v}_n" for v in present})
    wide = hits.join(nsel, on="alert_id")
    out = {"margin": H6_MARGIN, "n_boot": n_boot, "k": k, "baseline": "amount (plain top-40 by amount)", "share": {}}
    for pop, w in (("alerts_with_pattern_edges", wide.filter(pl.col("n_relevant") > 0)), ("all_alerts", wide)):
        att = [a if a is not None else f"alert:{i}" for i, a in zip(w["alert_id"], w["attempt_id"], strict=True)]
        out["share"][pop] = {"alerts": w.height, "attempts": len(set(att))}
        for den in ("k", "selected"):
            def share(v, w=w, den=den):
                return (w[v] / (k if den == "k" else w[f"{v}_n"])).to_numpy()
            r = paired(att, share(H6_VS) - share("amount"))
            r |= {"share_scorer_pcst": float(share(H6_VS).mean()), "share_amount": float(share("amount").mean())}
            r["lower_bound_ge_margin_attempts"] = bool(r["ci95_attempts"][0] >= H6_MARGIN)
            out["share"][pop][f"denominator_{den}"] = r
    has = per_alert.filter(pl.col("n_relevant") > 0)
    rec = has.with_columns((pl.col("hits") / pl.col("n_relevant")).alias("recall")).pivot(on="view", index=["alert_id", "attempt_id"], values="recall")
    rec = rec.join(has.pivot(on="view", index="alert_id", values="hits"), on="alert_id", suffix="_hits")
    rec = rec.join(has.filter(pl.col("view") == "amount_1hop").select("alert_id", pl.col("n_sel").alias("pool_1hop")), on="alert_id")
    out["recall_secondary"] = {"all": _h6_block(rec, n_boot, seed), "pool_1hop_at_least_k": _h6_block(rec.filter(pl.col("pool_1hop") >= k), n_boot, seed)}
    return out


def _h6_block(wide: pl.DataFrame, n_boot: int, seed: int) -> dict:
    from ftm.eval import bootstrap as bs

    out = {"alerts_with_pattern_edges": wide.height, "comparisons": {}}
    if wide.height < 2:
        return out
    att = [a if a is not None else f"alert:{i}" for i, a in zip(wide["alert_id"], wide["attempt_id"], strict=True)]
    out["attempts"] = int(len(set(att)))
    cl_attempt, cl_alert = bs.cluster_ids(att, []), np.arange(wide.height)
    for other in (v for v in (*H6_BASELINES, "scorer", "scorer_greedy", "long_prompt", "ppr", *EXPLAINER_VIEWS) if v in wide.columns):
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


def evaluate(ds: str, parts=("eval",), edge_cost: float = 0.05, prize_source: str = "scorer", k: int = K, limit: int | None = None,
             explainer_views: tuple[str, ...] = (), final: bool = False) -> dict:  # fmt: skip
    """Pattern-edge recall among the K and trail checks over the val alerts of `parts` (data/alerts/<ds>/val.parquet).

    `explainer_views` (names from `EXPLAINER_VIEWS`) adds the explainer comparisons; their scores come from the cache of
    `scripts/run_explainers.py` and are computed live for an alert that is not in it (slow: seconds per alert).

    `final=True` (week 14): the alerts are the TEST alert set (`data/alerts/<ds>/test.parquet`, bulk and tail together, a `test_tail` column
    separates them), the evidence may come from train, val and test edges up to each alert's time, the scorer's test scores come from
    `edge_scorer --final`, and the pattern labels of the test split score the views. Locked: needs FTM_FINAL_RUN=1."""
    cfg = io.load_config(ds)
    pq = cfg["parquet_dir"]
    if final:
        guards.require_final_run("the evidence-view evaluation on test alerts")
        al = pl.read_parquet(Path(cfg["alerts_dir"]) / "test.parquet")  # time-guard: not transactions
    else:
        al = pl.read_parquet(Path(cfg["alerts_dir"]) / "val.parquet").filter(pl.col("val_part").is_in(list(parts)))  # time-guard: not transactions
    if limit:
        al = al.head(limit)
    scores = _scores(ds, final) if prize_source == "scorer" else None
    pat = io.scan_patterns(pq, split=splits_for(final)).select("tx_id", "attempt_id").collect()
    by_att = {a: set(g["tx_id"].to_list()) for (a,), g in pat.group_by("attempt_id")}
    rows, t0 = [], time.time()
    for r in al.iter_rows(named=True):
        cand = candidates(cfg, r["bank"], r["timestamp"], final=final)
        i = int(np.flatnonzero(cand["tx_id"].to_numpy() == r["tx_id"])[0])
        pr = prizes(cand, scores)
        explain = {n: explainer_scores(ds, n, int(r["bank"]), int(r["tx_id"]), r["timestamp"], cand, final) for n in explainer_views} or None
        masks = select_views(cand, i, pr, k, edge_cost, home_bank=int(r["bank"]), explain=explain)
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
    for name in (v for v in VIEWS if v in set(df["view"])):
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
    ap.add_argument("--explainers", nargs="*", choices=EXPLAINER_VIEWS, default=[], help="add gnnexplainer / pgexplainer (cache of scripts/run_explainers.py)")
    ap.add_argument("--final", action="store_true", help="week 14: the TEST alert set (needs FTM_FINAL_RUN=1, edge_scorer --final and the final explainer cache)")
    a = ap.parse_args()
    res = evaluate(a.data, tuple(a.parts), a.edge_cost, a.prize, limit=a.limit, explainer_views=tuple(a.explainers), final=a.final)
    df = res.pop("_per_alert")
    res["h6"] = h6_report(df)
    tag = f"{a.data}_{a.prize}_{'_'.join(a.parts)}" + ("_explainers" if a.explainers else "") + ("_FINAL" if a.final else "")
    Path("data/runs").mkdir(exist_ok=True)
    df.write_parquet(f"data/runs/evidence_views_{tag}.parquet")
    Path(f"data/runs/evidence_views_{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
