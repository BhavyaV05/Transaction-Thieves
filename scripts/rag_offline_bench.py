"""Offline (CPU) benchmark of the five RAG components with WITH vs WITHOUT arms (T3, docs/t3_rag_bench.md).

    nice -n 10 python scripts/rag_offline_bench.py [--sections evidence regulatory library memory finfre] [--n-boot 2000]
        [--train-n 3000] [--out outputs/rag_bench/offline.json] [--md outputs/rag_bench/offline.md]

Sections (a)-(e) of the task. The data used to CHOOSE anything is TRAIN (time-blocked) and val `fit` + `cal`. Val `eval` is not read at
all by this script (the per-section `eval` column of the plan is the GPU matrix's final, once-only step); nothing here touches test.

  evidence    C11 views x edge scorer v1/v2 (+ the NO-evidence arm): pattern-edge recall@k with attempt-cluster CI, connected-trail
              rate, prompt tokens (ftm.retrieval.evidence_views.evaluate on fit+cal; cached per scorer under outputs/rag_bench/)
  regulatory  C6: bm25 / keyword (/ hybrid when sentence-transformers is installed) x fixed vs alert-aware query x rules_first on/off,
              plus NO rules and the whole corpus: recall@k / hit@k of the red flags (expert-question set), citation validity/exact of the
              LLM-free citer, and on the real fit+cal alerts: citable rule in the top 5, distinct top-5 sets (the fixed query returns one
              set for every alert), typology hit@5, prompt size
  library     C12: hit rate (non-empty strategies) by distance cutoff for shape vs account libs (and a back-test-built shape library from
              TRAIN, the no-LLM rebuild of docs/t3_rag_bench.md), and the held-out realised gain of the returned notes' questions on fit+cal
  memory      case memory (Sharma / Torres / Co-Investigator): neighbour-label vote AUC, time-blocked on train and on fit+cal queries,
              vs a p0-only baseline, and whether the vote adds to p0 (logistic p0 vs p0 + vote, fitted on later train, applied to val)
  finfre      FinFRE-RAG pool: exemplar label balance and the kNN-vote AUC at positive fraction natural / 0.33 / 0.5, vs the zero-shot arm
              (no exemplars: the vote is the constant base rate, AUC 0.5; the score is p0 alone)

Owner: T3 (Akshat).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftm.eval import bootstrap as bs  # noqa: E402

OUT = ROOT / "outputs" / "rag_bench"
DATA = "hi_medium"
ALERT_DIR = ROOT / "data" / "alerts" / "hi_medium_v2"  # p0 v2: the tuned profile's
BACKTEST_TABLE = ROOT / "configs" / "fusion" / "headroom_v1_table.jsonl"
TRAIN_ROWS = ROOT / "data" / "backtest" / DATA / "7613b89b5dcc" / "alert_rows.parquet"
VAL_ROWS = ROOT / "data" / "backtest" / DATA / "f8b676b7d1fc" / "alert_rows.parquet"
LIBS = {"shape_frozen": "outputs/t3_library/frozen/shape_gemma.json", "account_frozen": "outputs/t3_library/frozen/account_gemma.json",
        "shape_teacher_only": "outputs/t3_library/frozen_v2/shape_teacher_only.json",
        "account_teacher_only": "outputs/t3_library/frozen_v2/account_teacher_only.json"}  # fmt: skip
CUTOFFS = (0.0, 1.0, 2.0, 3.0, 4.0, math.inf)
FIXED_QUERY = "sharing information with another financial institution about suspected money laundering"
PAST_SHARE = 0.6  # train: first 60% by time = memory / pool source, the rest fits the combiner (time-blocked)


# -- small numeric helpers (pure, tested) ---------------------------------------------------------------------------
def auc(y, s) -> float:
    from scipy.stats import rankdata

    y, s = np.asarray(y, "f8"), np.asarray(s, "f8")
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, "f8"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_logistic(X, y, ridge: float = 1e-3, steps: int = 40) -> np.ndarray:
    """Intercept + coefficients of a ridge logistic regression (Newton). X: n x d."""
    X = np.column_stack([np.ones(len(X)), np.asarray(X, "f8")])
    y = np.asarray(y, "f8")
    b = np.zeros(X.shape[1])
    pen = ridge * np.eye(X.shape[1])
    pen[0, 0] = 0.0
    for _ in range(steps):
        q = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        g = X.T @ (y - q) - pen @ b
        H = (X * (q * (1 - q))[:, None]).T @ X + pen + 1e-9 * np.eye(len(b))
        step = np.linalg.solve(H, g)
        b = b + step
        if np.abs(step).max() < 1e-8:
            break
    return b


def predict_logistic(b, X) -> np.ndarray:
    X = np.column_stack([np.ones(len(X)), np.asarray(X, "f8")])
    return 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))


def boot_ci(stat, clusters, n_boot: int, seed: int = 0) -> list[float]:
    """95% cluster-bootstrap percentile CI of stat(row_indices)."""
    s = bs.cluster_bootstrap(stat, clusters, n_boot=n_boot, seed=seed)
    s = s[~np.isnan(s)]
    return [float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))] if len(s) else [float("nan"), float("nan")]


def auc_with_ci(y, s, clusters, n_boot: int) -> dict:
    y, s = np.asarray(y, "f8"), np.asarray(s, "f8")
    return {"auc": auc(y, s), "ci": boot_ci(lambda ix: auc(y[ix], s[ix]), clusters, n_boot)}


def auc_diff_with_ci(y, s_a, s_b, clusters, n_boot: int) -> dict:
    """AUC(a) - AUC(b), paired cluster bootstrap (the same resamples for both)."""
    y, s_a, s_b = (np.asarray(v, "f8") for v in (y, s_a, s_b))
    return {"diff": auc(y, s_a) - auc(y, s_b), "ci": boot_ci(lambda ix: auc(y[ix], s_a[ix]) - auc(y[ix], s_b[ix]), clusters, n_boot)}


def time_split(timestamps, share: float = PAST_SHARE) -> tuple[np.ndarray, np.ndarray]:
    """(past, later) index arrays by time order: the memory / pool is built from `past`, queries come from `later`."""
    o = np.argsort(np.asarray(timestamps), kind="stable")
    cut = int(round(len(o) * share))
    return o[:cut], o[cut:]


# -- case-memory scores ----------------------------------------------------------------------------------------------
def _standardise(x_past: np.ndarray):
    mu, sd = x_past.mean(0), np.maximum(x_past.std(0), 1e-9)
    return mu, sd


def knn_scores(x_mem, y_mem, x_q, k: int = 10, per_class: int = 2, exclude_self: bool = False) -> dict[str, np.ndarray]:
    """Per query: `vote` = share of positives among the k nearest memory rows; `margin` = mean distance to the `per_class` nearest clean
    rows minus that to the nearest laundering rows (the Group C prompt shows 2 nearest of each class; higher = closer to laundering).
    Features are standardised on the memory (as ftm.external.case_memory.CaseMemory does)."""
    x_mem, x_q, y_mem = np.asarray(x_mem, "f8"), np.asarray(x_q, "f8"), np.asarray(y_mem)
    mu, sd = _standardise(x_mem)
    zm, zq = (x_mem - mu) / sd, (x_q - mu) / sd
    pos, neg = np.flatnonzero(y_mem == 1), np.flatnonzero(y_mem == 0)
    vote, margin = np.empty(len(zq)), np.empty(len(zq))
    for i, q in enumerate(zq):
        d = np.sqrt(((zm - q) ** 2).sum(1))
        if exclude_self:
            d[int(np.argmin(d))] = np.inf  # leave-one-out when the query is in the memory
        near = np.argpartition(d, min(k, len(d) - 1))[:k]
        vote[i] = y_mem[near].mean()
        dp, dn = np.sort(d[pos])[:per_class], np.sort(d[neg])[:per_class]
        margin[i] = dn.mean() - dp.mean()
    return {"vote": vote, "margin": margin}


def balanced_sample(labels, n: int, pos_frac: float | None, seed: int = 0) -> np.ndarray:
    """Indices of n rows with `pos_frac` positives (all of a class if it has fewer); None / <0 = every row (natural)."""
    labels = np.asarray(labels)
    if pos_frac is None or pos_frac < 0:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    npos = int(round(n * pos_frac))
    return np.sort(np.concatenate([rng.permutation(pos)[:npos], rng.permutation(neg)[: n - npos]]))


def beyond_p0(y_fit, p0_fit, s_fit, y_test, p0_test, s_test, clusters_test, n_boot: int) -> dict:
    """Does a retrieval score `s` carry information beyond p0? Logistic on logit(p0) vs on [logit(p0), s], both fitted on the later-train
    queries (`*_fit`) and applied to the test queries. Reports test AUC of each, the AUC difference with a paired cluster CI, and the
    coefficient on s."""
    b0 = fit_logistic(logit(p0_fit)[:, None], y_fit)
    b1 = fit_logistic(np.column_stack([logit(p0_fit), s_fit]), y_fit)
    z0 = predict_logistic(b0, logit(p0_test)[:, None])
    z1 = predict_logistic(b1, np.column_stack([logit(p0_test), s_test]))
    d = auc_diff_with_ci(y_test, z1, z0, clusters_test, n_boot)
    return {"auc_p0_only": auc(y_test, z0), "auc_p0_plus_s": auc(y_test, z1), "auc_gain": d["diff"], "auc_gain_ci": d["ci"],
            "coef_s": float(b1[2]), "brier_p0_only": float(np.mean((z0 - y_test) ** 2)), "brier_p0_plus_s": float(np.mean((z1 - y_test) ** 2))}  # fmt: skip


# -- data ----------------------------------------------------------------------------------------------------------------
def _load_rla():
    spec = importlib.util.spec_from_file_location("run_llm_agent", ROOT / "scripts" / "run_llm_agent.py")
    rla = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rla)
    return rla


def load_contexts(train_n: int, cache: Path = OUT / "contexts.pkl") -> dict:
    """TRAIN sample (time-ordered later) and val fit + cal as AlertContext + label + part + attempt (cached: 0.3 s per alert)."""
    if cache.exists():
        return pickle.loads(cache.read_bytes())  # noqa: S301 - our own cache
    import polars as pl

    rla = _load_rla()
    t0 = time.time()
    out = {"train": [], "val": []}
    for split, part, n in (("train", None, train_n), ("val", "fit", 10_000), ("val", "cal", 10_000)):
        table = pl.read_parquet(ALERT_DIR / f"{split}.parquet")  # time-guard: not transactions
        if part:
            table = table.filter(pl.col("val_part") == part)
        n = min(n, table.height)
        got = rla.data_alerts(DATA, split, n, 0, 40, val_part=part, evidence_view="amount_1hop", p0_variant="v2", scorer="v2")
        meta = {r["alert_id"]: r for r in table.iter_rows(named=True)}
        for ctx, y in got:
            m = meta[ctx.alert_id]
            out[split].append({"ctx": ctx, "y": int(y), "part": part or "train", "attempt": m["attempt_id"], "typology": m["typology"],
                               "ts": m["timestamp"].isoformat()})  # fmt: skip
        print(f"  contexts {split}/{part}: {len(got)} ({time.time() - t0:.0f}s)", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(out))
    return out


def clusters_of(rows: list[dict]) -> np.ndarray:
    return bs.cluster_ids([r["attempt"] for r in rows], [[r["ctx"].summary["sender"] for r in rows], [r["ctx"].summary["receiver"] for r in rows]])


# -- (a) evidence ---------------------------------------------------------------------------------------------------------
def section_evidence(n_boot: int, limit: int | None = None) -> dict:
    """evidence_views.evaluate on val fit+cal for scorer v1 and v2 (cached), plus the no-evidence arm."""
    import polars as pl

    from ftm.retrieval import evidence_views as ev

    out: dict = {"parts": ["fit", "cal"], "arms": {}}
    out_frames: dict = {}
    for scorer in ("v1", "v2"):
        cache = OUT / f"evidence_{scorer}.parquet"
        meta = OUT / f"evidence_{scorer}.json"
        if not cache.exists():
            res = ev.evaluate(DATA, ("fit", "cal"), scorer=scorer, limit=limit)
            df = res.pop("_per_alert")
            OUT.mkdir(parents=True, exist_ok=True)
            df.write_parquet(cache)
            meta.write_text(json.dumps(res, indent=1))
        df = pl.read_parquet(cache)
        for name in ev.VIEWS:
            d = df.filter(pl.col("view") == name)
            if not d.height:
                continue
            h = d.filter(pl.col("n_relevant") > 0)
            rec = (h["hits"] / h["n_relevant"]).to_numpy()
            att = [a if a is not None else f"alert:{i}" for i, a in enumerate(h["attempt_id"])]
            cl = bs.cluster_ids(att, [])
            ci = bs.percentile_ci(bs.bootstrap_means(rec, cl, n_boot, 0)[:, 0])
            arm = f"{name}[{scorer}]" if "scorer" in name or name == "ppr" else f"{name}"
            if arm in out["arms"]:
                continue
            out_frames[arm] = h.with_columns((pl.col("hits") / pl.col("n_relevant")).alias("recall")).select("alert_id", "attempt_id", "recall")
            out["arms"][arm] = {"recall_at_k": float(rec.mean()), "recall_ci": [float(ci[0]), float(ci[1])], "alerts_with_pattern_edges": h.height,
                                "connected": float(d["connected"].mean()), "one_time_respecting_trail": float(d["trail"].mean()),
                                "mean_selected": float(d["n_sel"].mean()), "tokens_est": float(d["chars"].mean() / ev.CHARS_PER_TOKEN),
                                "scorer": scorer if ("scorer" in name or name == "ppr") else "none"}  # fmt: skip
    out["paired_recall_vs_amount_1hop"] = paired_recall(out_frames, "amount_1hop", n_boot)
    out["arms"]["NO_EVIDENCE"] = {"recall_at_k": 0.0, "recall_ci": [0.0, 0.0], "connected": float("nan"), "one_time_respecting_trail": float("nan"),
                                  "mean_selected": 0.0, "tokens_est": 0.0, "scorer": "none",
                                  "note": "alert summary only: by construction no pattern edge reaches the prompt; the WITH/WITHOUT effect on Brier is the GPU arm"}  # fmt: skip
    return out


def paired_recall(frames: dict, base: str, n_boot: int) -> dict:
    """Recall(view) - recall(base) over the alerts with pattern edges, paired, attempt-cluster bootstrap 95% CI."""
    import polars as pl

    out = {}
    b = frames[base]
    for name, f in frames.items():
        if name == base:
            continue
        j = b.join(f, on="alert_id", suffix="_o")
        att = [a if a is not None else f"alert:{i}" for i, a in enumerate(j["attempt_id"])]
        d = (j["recall_o"] - j["recall"]).to_numpy()
        cl = bs.cluster_ids(att, [])
        lo, hi = bs.percentile_ci(bs.bootstrap_means(d, cl, n_boot, 0)[:, 0])
        out[name] = {"diff": float(d.mean()), "ci": [float(lo), float(hi)], "n": int(len(d))}
    del pl
    return out


# -- (b) regulatory ---------------------------------------------------------------------------------------------------------
def alert_query(row: dict, gate) -> str:
    from ftm.compliance.query import rules_query

    return rules_query(row["ctx"], gate)


def shape_typology_cue(shape: dict) -> str:
    """Typology guess from the situation shape alone (no label): many payers in and few out = fan-in, the reverse = fan-out, both many =
    scatter-gather. "" when the shape says nothing. A proposed extension of `rules_query` (its `typology_guess` is never set by the
    pipeline, so the alert-aware query carries no typology word today); measured here as the arm `alert_aware+shape_cue`."""
    high = ("4-7", "8+")
    i, o = shape.get("in_deg") in high, shape.get("out_deg") in high
    return "fan-in" if i and shape.get("out_deg") in ("0", "1") else "fan-out" if o and shape.get("in_deg") in ("0", "1") else "scatter-gather" if i and o else ""


def cue_query(row: dict, gate) -> str:
    from ftm.compliance.query import TYPOLOGY_WORDS

    cue = shape_typology_cue(row["ctx"].summary["shape"])
    return alert_query(row, gate) + (" " + TYPOLOGY_WORDS[cue] if cue else "")


def section_regulatory(contexts: dict | None, k_list=(3, 5, 10)) -> dict:
    from ftm.compliance import rag_eval
    from ftm.compliance.gate import Gate
    from ftm.compliance.query import JURISDICTION_WORDS

    gates = {"bm25": Gate.from_config(retrieval="bm25"), "keyword": Gate.from_config(retrieval="keyword")}
    hybrid_note = "measured"
    try:
        gates["hybrid"] = Gate.from_config(retrieval="hybrid")
        gates["hybrid"].lookup_rules("warm up", k=1)  # loads bge-small once (an error here = no sentence-transformers / weights)
    except Exception as e:  # noqa: BLE001 - the hybrid arm is then reported as unavailable, never silently dropped
        gates.pop("hybrid", None)
        hybrid_note = f"hybrid not measured here ({type(e).__name__}: {str(e)[:80]})"
    g0 = gates["bm25"]
    cases = rag_eval.citation_cases()
    whole_tokens = (sum(len(r.text) for r in g0.rules.values()) + sum(len(f.text) for f in g0.red_flags.values())) / 4
    res: dict = {"hybrid": hybrid_note, "n_docs": len(g0.rules) + len(g0.red_flags), "n_rules": len(g0.rules), "n_red_flags": len(g0.red_flags), "arms": {}}
    for mode, g in gates.items():  # question-level retrieval recall (each expert question cites its red flags), pure ranking
        for k in k_list:
            res["arms"][f"retrieval/{mode}/k={k}"] = rag_eval.retrieval_eval(lambda q, kk, g=g: g.lookup_rules(q, k=kk, rules_first=False), k=k)
    qa = {"fixed": lambda c: FIXED_QUERY, "question": lambda c: c["question"], "fixed+question": lambda c: f"{FIXED_QUERY} {c['question']}",
          "alert_aware": lambda c: f"{FIXED_QUERY} {JURISDICTION_WORDS.get(c['jurisdiction'], '')} {c['question']}"}  # fmt: skip
    for mode, g in gates.items():  # citation of the LLM-free citer (top-ranked citable rule), partner jurisdiction filter vs none
        for qn, qf in qa.items():
            for jur in ("partner", "none"):
                for pin in (False, True):
                    res["arms"][f"citation/{mode}/{qn}/jur={jur}/rules_first={pin}"] = rag_eval.citation_by_retrieval(
                        lambda q, j, k, g=g, pin=pin: g.lookup_rules(q, j, k, rules_first=pin), cases, qf, jur, gate=g0)
    res["arms"]["citation/NO_RULES"] = {"n": len(cases), "valid": 0.0, "exact": 0.0, "missing": 1.0,
                                        "note": "no rule in the prompt: nothing to cite"}  # fmt: skip
    res["arms"]["citation/whole_corpus"] = rag_eval.citation_whole_corpus(cases, g0)
    if contexts is not None:  # the real fit+cal alerts: what lookup_rules(rules_query(alert)) returns vs the fixed sentence
        rows = contexts["val"]
        tmatch = {t: {f.flag_id for f in g0.red_flags.values() if t in f.typologies} for t in {r["typology"] for r in rows if r["typology"]}}
        stats = {}
        for mode, g in gates.items():
            for qn in ("fixed", "alert_aware", "alert_aware+shape_cue"):
                for pin in (False, True):
                    sets, cit, th, size = [], 0, [], []
                    for row in rows:
                        q = FIXED_QUERY if qn == "fixed" else alert_query(row, g0) if qn == "alert_aware" else cue_query(row, g0)
                        hits = g.lookup_rules(q, k=5, rules_first=pin)
                        ids = tuple(h["rule_id"] for h in hits)
                        sets.append(ids)
                        cit += any(h["kind"] == "rule" for h in hits)
                        size.append(sum(len(h["text"]) for h in hits) / 4)
                        if row["typology"]:
                            th.append(bool(set(ids) & tmatch[row["typology"]]))
                    stats[f"alerts/{mode}/{qn}/rules_first={pin}"] = {
                        "n_alerts": len(rows), "distinct_top5_sets": len(set(sets)), "citable_rule_in_top5": cit / len(rows),
                        "typology_flag_hit_at_5": float(np.mean(th)) if th else float("nan"), "n_typology_alerts": len(th),
                        "mean_prompt_tokens_est": float(np.mean(size))}  # fmt: skip
        cues = [(shape_typology_cue(r["ctx"].summary["shape"]), r["typology"]) for r in rows if r["typology"]]
        stats["alerts/shape_cue_accuracy"] = {"n_typology_alerts": len(cues), "cue_given": float(np.mean([c != "" for c, _ in cues])),
                                              "cue_correct_when_given": float(np.mean([c.replace("-", "_") == t for c, t in cues if c] or [float("nan")]))}  # fmt: skip
        stats["alerts/NO_RULES"] = {"n_alerts": len(rows), "distinct_top5_sets": 1, "citable_rule_in_top5": 0.0, "mean_prompt_tokens_est": 0.0}
        stats["alerts/whole_corpus"] = {"n_alerts": len(rows), "distinct_top5_sets": 1, "citable_rule_in_top5": 1.0, "mean_prompt_tokens_est": whole_tokens}
        res["arms"].update(stats)
    return res


# -- (c) library ---------------------------------------------------------------------------------------------------------------
def load_gain_table(rows_path: Path, contexts_ids: set[str]) -> dict[tuple, float]:
    """(alert_id, canonical, eps) -> Brier gain of asking that question once from p0: (p0 - y)^2 - (post - y)^2, post = p0 odds x the
    train-fitted headroom table's LR of the answer bin. Only accepted table rows."""
    import polars as pl


    table = {}
    for ln in BACKTEST_TABLE.read_text().splitlines():
        r = json.loads(ln)
        if r.get("accepted"):
            table[(r["program_canonical"], float(r["eps"]))] = r["lr"]
    df = pl.read_parquet(rows_path).filter(pl.col("alert_id").is_in(sorted(contexts_ids)))  # time-guard: not transactions
    return {(a, c, float(e), b): (table[(c, float(e))].get(b), y) for a, c, e, b, y in
            df.select("alert_id", "canonical", "eps", "answer_bin", "label").iter_rows() if (c, float(e)) in table}  # fmt: skip


def realised_gains(table: dict, p0: dict[str, float]) -> dict[tuple, float]:
    out = {}
    for (a, c, e, _b), (lr, y) in table.items():
        if lr is None or a not in p0:
            continue
        p = p0[a]
        post = p * lr / (p * lr + 1 - p)
        out[(a, c, e)] = (p - y) ** 2 - (post - y) ** 2
    return out


def library_eval(lib, rows: list[dict], gains: dict, cutoffs=CUTOFFS, n_boot: int = 1000, k: int = 3) -> dict:
    """Hit rate by cutoff, and the realised (held-out) gain of the returned notes' questions, by note kind."""
    from ftm.library.shape import SituationShape

    cl = clusters_of(rows)
    res: dict = {"n_notes": len(lib), "keying": lib.keying, "by_cutoff": {}}
    for d in cutoffs if lib.keying == "shape" else (math.inf,):
        hit, per_kind = [], {"success": [], "failure": []}
        for row in rows:
            sh = row["ctx"].summary["shape"]
            notes = (lib.search(SituationShape.from_dict(sh), k=k, max_dist=d) if lib.keying == "shape"
                     else lib.search(account=sh.get("receiver"), k=k))  # fmt: skip
            hit.append(bool(notes))
            for kind in per_kind:
                g = [gains[(row["ctx"].alert_id, s.question, s.eps)] for s in notes if s.kind == kind and (row["ctx"].alert_id, s.question, s.eps) in gains]
                per_kind[kind].append(float(np.mean(g)) if g else float("nan"))
        entry = {"hit_rate": float(np.mean(hit)), "n_alerts_hit": int(sum(hit))}
        for kind, vals in per_kind.items():
            v = np.array(vals)
            ok = ~np.isnan(v)
            entry[f"{kind}_notes_alerts_with_gain"] = int(ok.sum())
            if ok.sum() >= 5:
                entry[f"{kind}_mean_realised_gain"] = float(v[ok].mean())
                entry[f"{kind}_gain_ci"] = boot_ci(lambda ix, v=v[ok]: float(v[ix].mean()), cl[ok], n_boot)
        res["by_cutoff"][str(d)] = entry
    return res


def build_backtest_library(train_rows: list[dict], n_min: int = 20, z: float = 2.0):
    """Shape library built WITHOUT any LLM run: records = (train alert, accepted table question, eps) with the single-question Brier gain
    from the TRAIN rows of the back-test (every alert x every table question, so no collapse onto the planner's one favourite question)."""
    from ftm.library.build import _family, build_library

    ids = {r["ctx"].alert_id for r in train_rows}
    gains = realised_gains(load_gain_table(TRAIN_ROWS, ids), {r["ctx"].alert_id: r["ctx"].p0 for r in train_rows})
    by_id = {r["ctx"].alert_id: r for r in train_rows}
    recs = [{"alert_id": a, "split": "train", "shape": by_id[a]["ctx"].summary["shape"], "receiver": by_id[a]["ctx"].summary["receiver"],
             "question": c, "family": _family(c), "eps": e, "accepted": True, "gain": g} for (a, c, e), g in gains.items()]  # fmt: skip
    return build_library(recs, keying="shape", n_min=n_min, z=z), len(recs)


def section_library(contexts: dict, n_boot: int, write_library: Path | None = None) -> dict:
    from ftm.library.store import StrategyLibrary

    val = contexts["val"]
    gains = realised_gains(load_gain_table(VAL_ROWS, {r["ctx"].alert_id for r in val}), {r["ctx"].alert_id: r["ctx"].p0 for r in val})
    res: dict = {"n_val_alerts": len(val), "n_gain_rows": len(gains), "arms": {"NO_LIBRARY": {"hit_rate": 0.0}}}
    for name, rel in LIBS.items():
        p = ROOT / rel
        if p.exists():
            res["arms"][name] = library_eval(StrategyLibrary.load(p), val, gains, n_boot=n_boot)
    for n_min in (20, 10):
        lib, n_rec = build_backtest_library(contexts["train"], n_min=n_min)
        out = library_eval(lib, val, gains, n_boot=n_boot)
        out["records"] = n_rec
        out["note"] = f"built from TRAIN back-test rows only (train sample {len(contexts['train'])} alerts), n_min {n_min}, z 2"
        if write_library is not None and n_min == 20:
            lib.freeze()
            out["written"] = {"path": str(write_library), "content_hash": lib.save(write_library)}
        res["arms"][f"shape_backtest_built_nmin{n_min}"] = out
    return res


# -- (d)/(e) memory and FinFRE ---------------------------------------------------------------------------------------------------
def _feat(rows):
    from ftm.external.case_memory import alert_features

    return np.array([alert_features(r["ctx"]) for r in rows], "f8")


def _prepare_splits(contexts: dict):
    train, val = contexts["train"], contexts["val"]
    past, later = time_split([r["ts"] for r in train])
    return train, val, past, later


def section_memory(contexts: dict, n_boot: int, mem_n: int = 500) -> dict:
    train, val, past, later = _prepare_splits(contexts)
    X_tr, X_va = _feat(train), _feat(val)
    y_tr, y_va = np.array([r["y"] for r in train]), np.array([r["y"] for r in val])
    cl_va = clusters_of(val)
    p0_tr, p0_va = X_tr[:, 0], X_va[:, 0]
    res: dict = {"train_alerts": len(train), "past": len(past), "later": len(later), "val_fit_cal": len(val),
                 "train_pos_rate": float(y_tr.mean()), "val_pos_rate": float(y_va.mean()), "arms": {}}  # fmt: skip
    res["p0_only"] = {"train_later_auc": auc(y_tr[later], p0_tr[later]), "val_auc": auc(y_va, p0_va),
                      "val_auc_ci": auc_with_ci(y_va, p0_va, cl_va, n_boot)["ci"]}  # fmt: skip
    everyone = np.arange(len(train))
    for label, frac in (("natural", None), ("balanced_0.5", 0.5)):
        idx = past[balanced_sample(y_tr[past], mem_n, frac)]  # blocked: memory = the first 60% of train in time, queries = the rest
        idx_full = everyone[balanced_sample(y_tr, mem_n, frac)]  # production: the memory is the whole train sample, queries are val (later in time)
        s_later = knn_scores(X_tr[idx], y_tr[idx], X_tr[later])
        s_val = knn_scores(X_tr[idx_full], y_tr[idx_full], X_va)
        arm = {"memory_rows_blocked": int(len(idx)), "memory_pos_rate_blocked": float(y_tr[idx].mean()), "memory_rows_val": int(len(idx_full)),
               "memory_pos_rate_val": float(y_tr[idx_full].mean())}  # fmt: skip
        for sc in ("vote", "margin"):
            arm[f"train_blocked_{sc}_auc"] = auc(y_tr[later], s_later[sc])
            arm[f"val_{sc}_auc"] = auc_with_ci(y_va, s_val[sc], cl_va, n_boot)
            arm[f"val_{sc}_minus_p0_auc"] = auc_diff_with_ci(y_va, s_val[sc], p0_va, cl_va, n_boot)
            arm[f"beyond_p0_{sc}"] = beyond_p0(y_tr[later], p0_tr[later], s_later[sc], y_va, p0_va, s_val[sc], cl_va, n_boot)
        res["arms"][f"memory_{label}"] = arm
    # the optimistic non-blocked leave-one-out (memory = all train sample, self excluded): shows what time-blocking removes
    sub = np.arange(len(train))[:: max(1, len(train) // 1500)]
    loo = knn_scores(X_tr[sub], y_tr[sub], X_tr[sub], exclude_self=True)
    res["arms"]["train_loo_unblocked_natural"] = {"vote_auc": auc(y_tr[sub], loo["vote"]), "rows": int(len(sub))}
    res["arms"]["NO_MEMORY"] = {"note": "empty memory: the prompt carries p0 only; GPU arm --no-memory", "val_auc": auc(y_va, p0_va)}
    return res


def section_finfre(contexts: dict, n_boot: int, pool_n: int = 500) -> dict:
    from ftm.external.finfre_rag import ExemplarPool, tabular_row

    train, val, past, later = _prepare_splits(contexts)
    y_tr, y_va = np.array([r["y"] for r in train]), np.array([r["y"] for r in val])
    p0_tr, p0_va = np.array([r["ctx"].p0 for r in train]), np.array([r["ctx"].p0 for r in val])
    cl_va = clusters_of(val)
    res: dict = {"pool_n": pool_n, "arms": {"zero_shot": {"note": "no exemplars: vote = constant, AUC 0.5; the score is p0 alone", "val_p0_auc": auc(y_va, p0_va)}}}
    qrows = [(r, tabular_row(r["ctx"])) for r in val]
    lrows = [(train[i], tabular_row(train[i]["ctx"])) for i in later]
    for label, frac in (("natural", None), ("pos_0.33", 0.33), ("pos_0.5", 0.5)):
        def pool_of(src, label_src=label, frac_src=frac):
            idx = src[balanced_sample(y_tr[src], pool_n, frac_src)]
            if frac_src is None and len(idx) > pool_n:  # natural: a random pool_n of the source at its natural rate
                idx = np.sort(np.random.default_rng(0).permutation(idx)[:pool_n])
            return idx, ExemplarPool.from_alerts([train[i]["ctx"] for i in idx], [int(y_tr[i]) for i in idx], seed=0)

        def vote(items, pool):
            v = []
            for _r, (num, cat) in items:
                lab = np.array([e["label"] for e in pool.retrieve(num, cat)])
                v.append(lab.mean() if len(lab) else 0.0)
            return np.array(v)

        idx_b, pool_b = pool_of(past)  # blocked (fits the combiner on the later train alerts)
        idx, pool = pool_of(np.arange(len(train)))  # production: the pool is the whole train sample, queries are val
        v_val, v_later = vote(qrows, pool), vote(lrows, pool_b)
        arm = {"pool_rows": int(len(idx)), "pool_pos_rate": float(y_tr[idx].mean()), "pool_pos_rate_blocked": float(y_tr[idx_b].mean()),
               "mean_positive_share_in_20_exemplars_val": float(v_val.mean()), "share_queries_with_any_positive_exemplar": float((v_val > 0).mean()),
               "val_vote_auc": auc_with_ci(y_va, v_val, cl_va, n_boot), "val_vote_minus_p0_auc": auc_diff_with_ci(y_va, v_val, p0_va, cl_va, n_boot),
               "train_blocked_vote_auc": auc(y_tr[later], v_later), "importance_method": pool.importance_method, "selected_features": pool.selected,
               "beyond_p0": beyond_p0(y_tr[later], p0_tr[later], v_later, y_va, p0_va, v_val, cl_va, n_boot)}  # fmt: skip
        res["arms"][f"pool_{label}"] = arm
    return res


# -- markdown ---------------------------------------------------------------------------------------------------------------------
def _f(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and x != x) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _ci(c):
    return "-" if not c else f"[{_f(c[0])}, {_f(c[1])}]"


def markdown(res: dict) -> str:
    L = ["# Offline RAG bench (CPU; train time-blocked + val fit+cal; val eval and test not read)", ""]
    if "evidence" in res:
        L += ["## Evidence views (val fit+cal, pattern-edge recall among the K=40)", "", "| arm | scorer | recall@40 [95% CI] | connected | one trail | selected | tokens |", "|---|---|---|---|---|---|---|"]
        for a, m in res["evidence"]["arms"].items():
            L.append(f"| {a} | {m['scorer']} | {_f(m['recall_at_k'])} {_ci(m['recall_ci'])} | {_f(m['connected'])} | {_f(m['one_time_respecting_trail'])} | {_f(m['mean_selected'], 1)} | {_f(m['tokens_est'], 0)} |")
        L.append("")
    if "regulatory" in res:
        r = res["regulatory"]
        L += [f"## Regulatory RAG ({r['n_rules']} rules, {r['n_red_flags']} red flags; hybrid: {r['hybrid']})", "", "| arm | metrics |", "|---|---|"]
        for a, m in r["arms"].items():
            L.append(f"| {a} | " + ", ".join(f"{k} {_f(v)}" for k, v in m.items() if isinstance(v, (int, float))) + " |")
        L.append("")
    if "library" in res:
        L += ["## Strategy library (val fit+cal alerts; realised gain = single-question Brier gain from the train-fitted table)", "", "| library | notes | cutoff | hit rate | success notes: mean gain [CI] | failure notes: mean gain [CI] |", "|---|---|---|---|---|---|"]
        for a, m in res["library"]["arms"].items():
            if "by_cutoff" not in m:
                L.append(f"| {a} | 0 | - | {_f(m['hit_rate'])} | - | - |")
                continue
            for d, e in m["by_cutoff"].items():
                L.append(f"| {a} | {m['n_notes']} | {d} | {_f(e['hit_rate'])} ({e['n_alerts_hit']}) | {_f(e.get('success_mean_realised_gain'), 4)} {_ci(e.get('success_gain_ci'))} | "
                         f"{_f(e.get('failure_mean_realised_gain'), 4)} {_ci(e.get('failure_gain_ci'))} |")
        L.append("")
    for key, title in (("memory", "Case memory (neighbour-label vote vs p0)"), ("finfre", "FinFRE-RAG pool (kNN vote vs p0)")):
        if key not in res:
            continue
        L += [f"## {title}", "", "```json", json.dumps(res[key], indent=1, default=float), "```", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs="+", default=["evidence", "regulatory", "library", "memory", "finfre"])
    ap.add_argument("--train-n", type=int, default=3000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--write-library", type=Path, default=None, help="save + freeze the back-test-built shape library (n_min 20) here")
    ap.add_argument("--out", type=Path, default=OUT / "offline.json")
    ap.add_argument("--md", type=Path, default=OUT / "offline.md")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"note": "descriptive; choose only from train / val fit+cal; val eval and test not read", "train_n": a.train_n}
    needs_ctx = {"regulatory", "library", "memory", "finfre"} & set(a.sections)
    ctx = load_contexts(a.train_n) if needs_ctx else None
    if "evidence" in a.sections:
        res["evidence"] = section_evidence(a.n_boot)
    if "regulatory" in a.sections:
        res["regulatory"] = section_regulatory(ctx)
    if "library" in a.sections:
        res["library"] = section_library(ctx, a.n_boot, a.write_library)
    if "memory" in a.sections:
        res["memory"] = section_memory(ctx, a.n_boot)
    if "finfre" in a.sections:
        res["finfre"] = section_finfre(ctx, a.n_boot)
    a.out.write_text(json.dumps(res, indent=1, default=float))
    a.md.write_text(markdown(res))
    print(f"-> {a.out}, {a.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
