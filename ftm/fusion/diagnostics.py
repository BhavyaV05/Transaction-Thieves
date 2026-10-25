"""Why does naive fusion move p0 the wrong way?  Three diagnostics on one dataset's validation alerts.

Owner: T4 (Shreyas).  Spec: plan.md 6.9, 13 (G2); docs/t4_fusion_findings.md.  Script: `scripts/fusion_diagnostics.py`.

G2 failed on real data because the posterior `logit p0 + sum of back-tested log LRs` is worse than p0, and worse with
more questions or less noise (docs/t1_handoffs.md). Hypotheses, each with its own table:

1. DOUBLE COUNTING.  The one-bank model behind p0 already uses the patterns the questions measure, so p0 and the answers
   are not independent given the label and Bayes' rule counts the evidence twice.  Test: the LR of every answer cell
   *within strata of p0* (Mantel-Haenszel pooled over strata, a risk ratio P(cell | launder) / P(cell | clean) like the
   back-test's LR) against the marginal LR on the same alerts.  `explained_by_p0` = 1 - log LR_conditional / log LR_marginal,
   clipped to [0, 1] per cell and averaged with weights n * |log LR_marginal|: 0 = the answers carry their evidence on top
   of p0, 1 = p0 already holds all of it.
2. SHIFT / WINNER'S CURSE.  The LRs were measured on train alerts and the same labels picked the accepted questions
   (`ftm/backtest/evalue.py`); validation is later in time.  Test: per answer cell, the claimed (train) LR against the
   empirical LR of the validation answers, with a CI, and how often the DIRECTION disagrees.
3. CALIBRATION.  Brier and ECE (equal-mass bins) of p0, naive fusion, shared-weight fusion, per-family fusion and
   per-family fusion with cross-validated ridge and prior; fitted variants are cross-fitted over whole clusters (as in
   `ftm.eval.g2`); the gain over p0 (Brier(p0) - Brier(fused), positive = the fusion helps) has a cluster-bootstrap CI.

The planner run, the pool, the clusters and the cross-fitting are `ftm.eval.g2`'s: nothing is re-implemented here.
Labels are used for scoring only, after the planner has finished; the test split is never touched.  Every count can be
resampled by alert (`weights`), which is how the CI on `explained_by_p0` is made (whole clusters resampled).

Result of `diagnose` / `run_diagnostics` (the script writes it as JSON):
  world, n_alerts, n_pos, n_answers, families_used, clusters, strata {k, edges}, cells [...], families [...],
  direction {...}, conditional {...}, calibration {...}, interpretation [str, ...], summary
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ftm.eval import bootstrap as bs
from ftm.eval import g2
from ftm.eval.metrics import brier, ece
from ftm.fusion.fuse import answer_features
from ftm.types import Reply

MIN_ABS = 0.05  # a cell whose log LR (claimed, or marginal on val) is smaller than this in size is "neutral": no direction, no share
HALF = 0.5  # continuity correction added to the launder and clean counts of a cell (split over the strata)
EXPLAINED_HIGH, EXPLAINED_SOME = 0.5, 0.25  # thresholds of the interpretation string
DISAGREE_HIGH, DISAGREE_SOME = 0.30, 0.15


@dataclass
class Answers:
    """The answered questions of the validation alerts, one entry per answer with a back-tested LR."""

    alert: np.ndarray  # index into the alert arrays
    question: np.ndarray  # "<program> @ eps <eps>"
    family: np.ndarray
    cell: np.ndarray  # answer cell (the LR table's key: yes/no, bin index, ...)
    claimed: np.ndarray  # log LR of the cell from the train back-test, as fusion uses it

    def __len__(self) -> int:
        return len(self.alert)


def answers_from_verdicts(alert_ids: Sequence[str], verdicts, pool) -> Answers:
    """Every answered question of the run with a back-tested LR for its cell (the same filter as `g2.answer_log_lrs`)."""
    idx = {a: i for i, a in enumerate(alert_ids)}
    by_question = {(c.program, c.eps): c for c in pool}
    rows = []
    for v in verdicts:
        for e in v.questions_log:
            c = by_question.get((e.program, e.eps))
            if c is None or not isinstance(e.response, Reply):
                continue
            x = c.log_lr(e.response.value)
            if x is not None:
                rows.append((idx[v.alert_id], f"{c.program} @ eps {c.eps:g}", c.family, c.cell(e.response.value), x))
    cols = list(zip(*rows, strict=True)) if rows else [[]] * 5
    return Answers(np.array(cols[0], dtype=int), np.array(cols[1], dtype=str), np.array(cols[2], dtype=str),
                   np.array(cols[3], dtype=str), np.array(cols[4], dtype="f8"))  # fmt: skip


def feature_matrix(ans: Answers, n_alerts: int) -> tuple[np.ndarray, list[str]]:
    """X (alerts x families): summed claimed log LR per family, as fusion sees it."""
    fams = sorted(set(ans.family.tolist()))
    if not fams:
        return np.zeros((n_alerts, 0)), []
    return answer_features(range(n_alerts), ans.alert.tolist(), ans.family.tolist(), ans.claimed, families=fams)


def p0_strata(p0, k: int = 3) -> tuple[np.ndarray, list[float]]:
    """Stratum per alert: k equal-mass bins of p0 (fewer if p0 has ties). Returns (labels 0.., interior edges)."""
    p0 = np.asarray(p0, "f8")
    edges = np.unique(np.quantile(p0, np.linspace(0, 1, k + 1)[1:-1])) if k > 1 else np.zeros(0)
    edges = edges[(edges > p0.min()) & (edges <= p0.max())]
    return np.searchsorted(edges, p0, side="right"), [float(e) for e in edges]


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of a rate k/n."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _log_mh(a, b, n1, n0, tot, h: float = HALF) -> float:
    """log of the Mantel-Haenszel risk ratio of 'in this cell' between launder and clean, pooled over strata.

    a, b: launder and clean answers in the cell, n1, n0: launder and clean answers of the whole question, tot: all
    answers, each per stratum (arrays). With one stratum it is (a + h) n0 / ((b + h) n1) = the back-test's LR. A stratum
    with one class missing carries no information (it contributes 0 to both sums, as in the textbook estimator)."""
    ok = (n1 > 0) & (n0 > 0)
    if not ok.any():
        return float("nan")
    hs = h * tot / tot[ok].sum()
    safe = np.where(tot > 0, tot, 1)
    num = ((a + hs) * n0 / safe)[ok].sum()
    den = ((b + hs) * n1 / safe)[ok].sum()
    return float(np.log(num / den))


def _by_stratum(rows, ans_alert, y, stratum, w, k):
    """Weighted launder and clean counts of the answer rows `rows`, per stratum."""
    wa, ya, sa = w[ans_alert[rows]], y[ans_alert[rows]], stratum[ans_alert[rows]]
    one = np.bincount(sa, weights=wa * ya, minlength=k)
    zero = np.bincount(sa, weights=wa * (1 - ya), minlength=k)
    return one, zero


def _estimates(cell_rows, group_rows, ans_alert, y, stratum, w, k) -> tuple[float, float, float]:
    """(marginal log LR, conditional log LR, SE of the marginal) with alert weights `w`."""
    n1s, n0s = _by_stratum(group_rows, ans_alert, y, stratum, w, k)
    as_, bs_ = _by_stratum(cell_rows, ans_alert, y, stratum, w, k)
    cond = _log_mh(as_, bs_, n1s, n0s, n1s + n0s)
    a, b, n1, n0 = as_.sum(), bs_.sum(), n1s.sum(), n0s.sum()
    marg = _log_mh(np.array([a]), np.array([b]), np.array([n1]), np.array([n0]), np.array([n1 + n0]))
    se = float("nan")
    if n1 > 0 and n0 > 0:
        se = math.sqrt(1 / (a + HALF) - 1 / (n1 + HALF) + 1 / (b + HALF) - 1 / (n0 + HALF))
    return marg, cond, se


def _structure(ans: Answers):
    """Row indices of every question and of every (question, cell)."""
    groups: dict[str, np.ndarray] = {}
    for q in dict.fromkeys(ans.question.tolist()):
        groups[q] = np.flatnonzero(ans.question == q)
    cells = []
    for q, rows in groups.items():
        for c in dict.fromkeys(ans.cell[rows].tolist()):
            r = rows[ans.cell[rows] == c]
            cells.append({"question": q, "family": str(ans.family[r[0]]), "cell": c, "rows": r, "group": rows,
                          "claimed": float(ans.claimed[r[0]])})  # fmt: skip
    return cells


def _explained(marg: np.ndarray, cond: np.ndarray, n: np.ndarray, min_abs: float = MIN_ABS) -> float:
    """Weighted mean over cells of clip(1 - cond / marg, 0, 1), weights n * |marg|; NaN when no cell qualifies."""
    ok = np.isfinite(marg) & np.isfinite(cond) & (np.abs(marg) >= min_abs)
    if not ok.any():
        return float("nan")
    s = np.clip(1 - cond[ok] / marg[ok], 0.0, 1.0)
    w = n[ok] * np.abs(marg[ok])
    return float((s * w).sum() / w.sum())


def _agrees(claimed: float, val: float) -> bool | None:
    if not np.isfinite(val) or abs(claimed) < MIN_ABS or abs(val) < MIN_ABS:
        return None
    return bool(np.sign(claimed) == np.sign(val))


def _wmean(x, w) -> float:
    x, w = np.asarray(x, "f8"), np.asarray(w, "f8")
    ok = np.isfinite(x)
    return float((x[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() and w[ok].sum() > 0 else float("nan")


def conditional_report(ans: Answers, y, p0, clusters, strata: int = 3, n_boot: int = 200, seed: int = 0) -> dict:
    """Hypotheses 1 and 2: the cell table (claimed, marginal and conditional LR, direction) and its summaries."""
    y, p0, clusters = np.asarray(y, "f8"), np.asarray(p0, "f8"), np.asarray(clusters)
    n_alerts = len(y)
    stratum, edges = p0_strata(p0, strata)
    k = int(stratum.max()) + 1
    cells = _structure(ans)
    ones = np.ones(n_alerts)
    rows = []
    for c in cells:
        marg, cond, se = _estimates(c["rows"], c["group"], ans.alert, y, stratum, ones, k)
        n = len(c["rows"])
        yy = y[ans.alert[c["rows"]]]
        k_pos = int(yy.sum())
        lo, hi = wilson(k_pos, n)
        gy = y[ans.alert[c["group"]]]
        gs = stratum[ans.alert[c["group"]]]
        cs = stratum[ans.alert[c["rows"]]]
        by_s = []
        for s in range(k):
            n1, n0 = gy[gs == s].sum(), (gs == s).sum() - gy[gs == s].sum()
            a_, b_ = yy[cs == s].sum(), (cs == s).sum() - yy[cs == s].sum()
            ok = n1 > 0 and n0 > 0
            by_s.append({"stratum": s, "n": int((cs == s).sum()),
                         "log_lr": float(np.log((a_ + HALF) * n0 / ((b_ + HALF) * n1))) if ok else None})  # fmt: skip
        agrees = _agrees(c["claimed"], marg)
        rows.append({
            "question": c["question"], "family": c["family"], "cell": c["cell"], "n": n, "n_launder": k_pos,
            "launder_rate": k_pos / n, "launder_rate_ci": [lo, hi], "claimed_lr": math.exp(c["claimed"]),
            "claimed_log_lr": c["claimed"], "val_lr": math.exp(marg) if np.isfinite(marg) else None,
            "val_log_lr": marg, "val_log_lr_ci": [marg - 1.96 * se, marg + 1.96 * se],
            "cond_lr": math.exp(cond) if np.isfinite(cond) else None, "cond_log_lr": cond,
            "cond_by_stratum": by_s, "mean_p0": float(p0[ans.alert[c["rows"]]].mean()),
            "direction_agrees": agrees,
            "contradicted": bool(agrees is False and (marg - 1.96 * se) * (marg + 1.96 * se) > 0),
        })  # fmt: skip

    # direction summary (hypothesis 2): only cells with a direction on both sides count
    judged = [r for r in rows if r["direction_agrees"] is not None]
    nj = sum(r["n"] for r in judged)
    n_dis = sum(not r["direction_agrees"] for r in judged)
    direction = {"cells_judged": len(judged), "cells_disagree": n_dis,
                 "cells_contradicted": sum(r["contradicted"] for r in judged), "answers_judged": nj,
                 "share_answers_disagree": sum(r["n"] for r in judged if not r["direction_agrees"]) / nj if nj else float("nan"),
                 "share_cells_disagree": n_dis / len(judged) if judged else float("nan")}  # fmt: skip

    # conditional summary (hypothesis 1), with a cluster-bootstrap CI on the share
    marg = np.array([r["val_log_lr"] for r in rows], "f8")
    cond = np.array([r["cond_log_lr"] for r in rows], "f8")
    nn = np.array([r["n"] for r in rows], "f8")
    claimed = np.array([r["claimed_log_lr"] for r in rows], "f8")
    share = _explained(marg, cond, nn)
    ci = [float("nan"), float("nan")]
    if rows and n_boot and np.isfinite(share) and len(np.unique(clusters)) > 1:

        def stat(idx):
            w = np.bincount(idx, minlength=n_alerts).astype("f8")
            est = np.array([_estimates(c["rows"], c["group"], ans.alert, y, stratum, w, k)[:2] for c in cells])
            nw = np.array([w[ans.alert[c["rows"]]].sum() for c in cells])
            return _explained(est[:, 0], est[:, 1], nw)

        samples = bs.cluster_bootstrap(stat, clusters, n_boot, seed)
        if np.isfinite(samples).any():
            ci = list(bs.percentile_ci(samples))
    cond_summary = {"explained_by_p0": share, "explained_by_p0_ci95": ci, "n_cells": len(rows),
                    "explained_vs_claimed": _explained(claimed, cond, nn),
                    "mean_abs_log_lr": {"claimed": _wmean(np.abs(claimed), nn), "val_marginal": _wmean(np.abs(marg), nn),
                                        "val_conditional": _wmean(np.abs(cond), nn)}}  # fmt: skip
    return {"strata": {"k": k, "edges": edges}, "cells": rows, "families": _family_rows(rows),
            "direction": direction, "conditional": cond_summary}  # fmt: skip


def _family_rows(cells: list[dict]) -> list[dict]:
    """Per family: the cells pooled, answers-weighted."""
    out = []
    for f in dict.fromkeys(r["family"] for r in cells):
        rs = [r for r in cells if r["family"] == f]
        n = np.array([r["n"] for r in rs], "f8")
        m = np.array([r["val_log_lr"] for r in rs], "f8")
        c = np.array([r["cond_log_lr"] for r in rs], "f8")
        j = [r for r in rs if r["direction_agrees"] is not None]
        nj = sum(r["n"] for r in j)
        out.append({"family": f, "questions": len({r["question"] for r in rs}), "cells": len(rs), "n": int(n.sum()),
                    "launder_rate": float(sum(r["n_launder"] for r in rs) / n.sum()),
                    "mean_abs_claimed_log_lr": _wmean(np.abs([r["claimed_log_lr"] for r in rs]), n),
                    "mean_abs_val_log_lr": _wmean(np.abs(m), n), "mean_abs_cond_log_lr": _wmean(np.abs(c), n),
                    "share_answers_disagree": sum(r["n"] for r in j if not r["direction_agrees"]) / nj if nj else None,
                    "explained_by_p0": _explained(m, c, n), "mean_p0": _wmean([r["mean_p0"] for r in rs], n)})  # fmt: skip
    return out


# --- hypothesis 3: calibration ------------------------------------------------------------------------------------
def calibration_report(X: np.ndarray, y, p0, clusters, n_boot: int = 1000, seed: int = 0, n_folds: int = 5,
                       ece_bins: int = 10) -> dict:  # fmt: skip
    """Brier and ECE of p0 and four fusions; gain = Brier(p0) - Brier(fused) with a paired cluster-bootstrap CI."""
    y, p0, clusters = np.asarray(y, "f8"), np.asarray(p0, "f8"), np.asarray(clusters)
    variants: dict[str, np.ndarray | str] = {"p0": p0}
    chosen_by: dict[str, list] = {}
    if X.shape[1]:
        variants["naive"] = g2.crossfit_posterior(p0, X, y, clusters, "naive")
        for name, mode, sel in (("shared", "shared", "fixed"), ("per_family", "per_family", "fixed"),
                                ("per_family_cv", "per_family", "cv")):  # fmt: skip
            try:
                chosen: list = []
                variants[name] = g2.crossfit_posterior(p0, X, y, clusters, mode, n_folds, seed, select=sel, chosen=chosen)
                if chosen:
                    chosen_by[name] = chosen
            except ValueError as e:
                variants[name] = f"not run: {e}"
    out = {}
    for name, p in variants.items():
        if isinstance(p, str):
            out[name] = {"note": p}
            continue
        row = {"brier": brier(y, p), "ece": ece(y, p, ece_bins), "mean_p": float(p.mean())}
        if name != "p0":
            d = bs.brier_diff(y, p0, p, clusters, n_boot=n_boot, seed=seed)
            row |= {"gain": float(d["diff"]), "gain_ci95": [float(v) for v in d["ci"]],
                    "ece_change": row["ece"] - ece(y, p0, ece_bins)}  # fmt: skip
        if name in chosen_by:
            row["chosen_per_fold"] = chosen_by[name]
        out[name] = row
    return out


# --- the whole thing ----------------------------------------------------------------------------------------------
def diagnose(y, p0, clusters, ans: Answers, world: str = "synth", strata: int = 3, n_boot: int = 1000,
             n_boot_share: int = 200, seed: int = 0, n_folds: int = 5) -> dict:  # fmt: skip
    """All three diagnostics for the validation alerts (`y`, `p0`, `clusters`, one entry per alert) and their answers."""
    y, p0 = np.asarray(y, "f8"), np.asarray(p0, "f8")
    X, fams = feature_matrix(ans, len(y))
    res = {"world": world, "n_alerts": len(y), "n_pos": int(y.sum()), "n_answers": len(ans), "families_used": fams,
           "clusters": bs.cluster_report(clusters)}  # fmt: skip
    if len(ans):
        res |= conditional_report(ans, y, p0, clusters, strata, n_boot_share, seed)
    else:
        res |= {"strata": {"k": 0, "edges": []}, "cells": [], "families": [], "direction": {}, "conditional": {}}
    res["calibration"] = calibration_report(X, y, p0, clusters, n_boot, seed, n_folds)
    res["interpretation"] = interpret(res)
    res["summary"] = summary_line(res)
    return res


def interpret(r: dict) -> list[str]:
    """Plain-language reading of the three tables, with the thresholds written out. Hypotheses, not verdicts."""
    if not r["n_answers"]:
        return ["No answered question with a back-tested LR: nothing to diagnose."]
    out = []
    c, d, cal = r["conditional"], r["direction"], r["calibration"]
    s, ci = c["explained_by_p0"], c["explained_by_p0_ci95"]
    if not np.isfinite(s):
        out.append(f"Double counting: no cell with a marginal |log LR| >= {MIN_ABS:g} on validation, share not defined.")
    else:
        wide = np.isfinite(ci[0]) and ci[1] - ci[0] > 0.6
        verdict = ("p0 already holds most of the evidence the answers carry (naive Bayes counts it twice)"
                   if s >= EXPLAINED_HIGH else "p0 holds part of the evidence the answers carry" if s >= EXPLAINED_SOME
                   else "the answers carry their evidence on top of p0; double counting is not what explains the loss")  # fmt: skip
        out.append(f"Double counting: {s:.0%} of the answers' evidence is explained by p0 (95% CI {ci[0]:.0%} to {ci[1]:.0%}"
                   f"; cutoffs {EXPLAINED_SOME:.0%} / {EXPLAINED_HIGH:.0%}): {verdict}."
                   + (" The CI is wide, so this is inconclusive." if wide else ""))  # fmt: skip
    sd = d.get("share_answers_disagree", float("nan"))
    if not np.isfinite(sd):
        out.append("Shift: no cell with a direction on both the train and the validation side.")
    else:
        verdict = ("the train LRs point the wrong way on a large share of validation answers (shift or winner's curse)"
                   if sd >= DISAGREE_HIGH else "some train LRs flip on validation" if sd >= DISAGREE_SOME
                   else "the directions of the train LRs mostly hold on validation")  # fmt: skip
        out.append(f"Shift: the direction disagrees on {sd:.0%} of the {d['answers_judged']} judged answers "
                   f"({d['cells_disagree']} of {d['cells_judged']} cells, {d['cells_contradicted']} contradicted with a "
                   f"CI excluding 0; cutoffs {DISAGREE_SOME:.0%} / {DISAGREE_HIGH:.0%}): {verdict}.")  # fmt: skip
    if "naive" in cal:
        p0, nv = cal["p0"], cal["naive"]
        fitted = {k: v for k, v in cal.items() if k not in ("p0", "naive") and "gain" in v}
        best = max(fitted, key=lambda k: fitted[k]["gain"]) if fitted else None
        out.append(f"Calibration: p0 Brier {p0['brier']:.4f} ECE {p0['ece']:.3f}; naive Brier {nv['brier']:.4f} "
                   f"ECE {nv['ece']:.3f} (gain {nv['gain']:+.4f}, mean p {nv['mean_p']:.3f} vs {p0['mean_p']:.3f})"
                   + (f"; best fitted: {best} gain {fitted[best]['gain']:+.4f} "
                      f"(CI {fitted[best]['gain_ci95'][0]:+.4f} to {fitted[best]['gain_ci95'][1]:+.4f})" if best else "")
                   + ".")  # fmt: skip
    return out


def summary_line(r: dict) -> str:
    if not r["n_answers"]:
        return f"fusion diagnostics [{r['world']}]: no answers"
    c, d, cal = r["conditional"], r["direction"], r["calibration"]
    sh = c["explained_by_p0"]
    sd = d.get("share_answers_disagree", float("nan"))
    g = cal.get("naive", {}).get("gain", float("nan"))
    best = max((v["gain"] for k, v in cal.items() if k not in ("p0", "naive") and "gain" in v), default=float("nan"))
    return (f"fusion diagnostics [{r['world']}]: {r['n_alerts']} alerts, {r['n_answers']} answers; explained by p0 "
            f"{sh:.0%}, direction disagrees {sd:.0%}; Brier gain naive {g:+.4f}, best fitted {best:+.4f}")  # fmt: skip


def run_diagnostics(banks, entries: list[dict], val, world: str = "synth", budget: float = g2.BUDGET,
                    ask_eps: float = g2.ASK_EPS, eps_levels: Sequence[float] = g2.EPS_LEVELS,
                    attempt_of: dict | None = None, hub_cap: int | None = None, seed: int = 0, fdr: float = 0.10,
                    **kw) -> dict:  # fmt: skip
    """The G2 planner run on `val` (`g2.play`), then `diagnose` on its answers. `kw`: strata, n_boot, n_boot_share, n_folds."""
    val = list(val)
    if not val:
        raise ValueError("no validation alerts")
    _, pool, res = g2.play(banks, entries, val, budget, ask_eps, eps_levels, seed, fdr)
    ans = answers_from_verdicts([a.alert_id for a in val], res.verdicts, pool)
    y = np.array([float(a.label) for a in val])
    p0 = np.array([a.p0 for a in val])
    out = diagnose(y, p0, g2.clusters_for(val, attempt_of, hub_cap), ans, world, seed=seed, **kw)
    out |= {"ask_eps": ask_eps, "budget": budget, "pool": len(pool)}
    return out


# --- the markdown table -------------------------------------------------------------------------------------------
def _f(x, spec=".2f") -> str:
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)


def to_markdown(r: dict) -> str:
    """The three tables as markdown, with the reading on top."""
    L = [f"# Fusion diagnostics: {r['world']}", "", r["summary"], "", "## Reading", ""]
    L += [f"- {t}" for t in r["interpretation"]]
    if r["n_answers"]:
        L += ["", f"## 1 and 2. Answer cells ({r['strata']['k']} strata of p0)", "",
              "claimed = LR from the train back-test; val = marginal LR of the validation answers (95% CI); cond = "
              "Mantel-Haenszel LR within strata of p0; dir = claimed and val point the same way.", "",
              "| question | cell | n | launder rate | claimed LR | val LR (95% CI) | cond LR | dir | mean p0 |",
              "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for c in r["cells"]:
            lo, hi = (math.exp(v) if np.isfinite(v) else None for v in c["val_log_lr_ci"])
            d = "-" if c["direction_agrees"] is None else "yes" if c["direction_agrees"] else "NO"
            L.append(f"| {c['question']} | {c['cell']} | {c['n']} | {c['launder_rate']:.2f} | {c['claimed_lr']:.2f} | "
                     f"{_f(c['val_lr'])} ({_f(lo)} to {_f(hi)}) | {_f(c['cond_lr'])} | {d} | {c['mean_p0']:.3f} |")  # fmt: skip
        L += ["", "### Per family", "",
              "| family | questions | cells | n | launder rate | mean abs log LR claimed / val / cond | "
              "answers with flipped direction | explained by p0 |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for f in r["families"]:
            L.append(f"| {f['family']} | {f['questions']} | {f['cells']} | {f['n']} | {f['launder_rate']:.2f} | "
                     f"{_f(f['mean_abs_claimed_log_lr'])} / {_f(f['mean_abs_val_log_lr'])} / "
                     f"{_f(f['mean_abs_cond_log_lr'])} | {_f(f['share_answers_disagree'], '.0%')} | "
                     f"{_f(f['explained_by_p0'], '.0%')} |")  # fmt: skip
        c = r["conditional"]
        L += ["", f"Explained by p0: {_f(c['explained_by_p0'], '.0%')} (95% CI {_f(c['explained_by_p0_ci95'][0], '.0%')} "
              f"to {_f(c['explained_by_p0_ci95'][1], '.0%')}); against the claimed LRs instead of the validation ones: "
              f"{_f(c['explained_vs_claimed'], '.0%')}."]  # fmt: skip
    L += ["", "## 3. Calibration (fitted fusions are cross-fitted over whole clusters)", "",
          "gain = Brier(p0) - Brier(fused), positive = the fusion helps; CI = paired cluster bootstrap.", "",
          "| posterior | Brier | ECE | mean p | gain over p0 (95% CI) |", "|---|---|---|---|---|"]  # fmt: skip
    for name, v in r["calibration"].items():
        if "note" in v:
            L.append(f"| {name} | {v['note']} | | | |")
            continue
        g = f"{v['gain']:+.4f} ({v['gain_ci95'][0]:+.4f} to {v['gain_ci95'][1]:+.4f})" if "gain" in v else ""
        L.append(f"| {name} | {v['brier']:.4f} | {v['ece']:.3f} | {v['mean_p']:.3f} | {g} |")
    ch = r["calibration"].get("per_family_cv", {}).get("chosen_per_fold")
    if ch:
        L += ["", "per_family_cv chose (ridge, prior) per outer fold: " + ", ".join(f"({c['ridge']:g}, {c['prior']})" for c in ch)]
    return "\n".join(L) + "\n"
