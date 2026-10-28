"""T3 validation failure analysis and offline replay (docs/t3_val_analysis.md).

    python scripts/t3_val_analysis.py [--n-boot 2000] [--out-dir outputs/t3_val/analysis]

CPU only, no LLM calls, no server. Reads the logged runs `outputs/t3_val/eval/<spec>/c###` (val `eval`) and the tuning runs
`outputs/t3_val/fitcal/{fit,cal}/c###` (compose mode), the realistic back-test table, and the attempt+accounts clusters.
Never reads the test split. Everything that is fitted (shrinkage, caps, thresholds, dropped families, recalibration) is fitted
on val `fit`+`cal` ONLY; `eval` is reported once per pre-declared variant. Deterministic: bootstrap seed 0, 2,000 resamples.

Writes `<out-dir>/analysis.json` (also printed) and `<out-dir>/tables.md` (the tables pasted into the doc).

Replay limit: the logged answers are those of the live run. A replayed posterior does not change which questions were asked or
when the planner stopped. The posterior is shown to the model in modes compose, act_only, llm_decides (prompt line "current
probability ... back-tested formula") and the `used_in_posterior` tag is shown in every mode that lists answers, so the model's
next choice depends on the posterior in those modes; fixed_list and batched modes pick from a list with the posterior shown too.
A replay is therefore an approximation of what a live run with the changed fusion would do.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftm.eval import bootstrap as bs  # noqa: E402
from ftm.llm.investigation import LRTable  # noqa: E402
from ftm.planners import e18  # noqa: E402

N_BOOT = 2000
P_CLIP = 1e-6
Z_CAP = 5.0  # |logit| cap used only when recalibrating p0
EVAL = ROOT / "outputs" / "t3_val" / "eval"
FIT = [ROOT / "outputs" / "t3_val" / "fitcal" / "fit", ROOT / "outputs" / "t3_val" / "fitcal" / "cal"]
TABLE = ROOT / "data" / "backtest" / "hi_medium" / "7df6bad5337d" / "lr_table.jsonl"
CLUSTERS = ROOT / "outputs" / "e18" / "clusters.json"
FUSION_MODES = ["ours_compose", "ours_fixed_list", "ours_act_only", "ours_batched_2_1", "ours_library_account"]
BELIEF_MODES = ["ours_llm_decides", "ours_no_backtest"]
OWN_PROB = ["react", "reflexion", "uot"]
P0_BUCKETS = [(0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 0.99), (0.99, 1.01)]


# ---------------------------------------------------------------------------------------------------------------------
# replay math (pure, tested in tests/test_t3_val_analysis.py)
def logit(p: float) -> float:
    p = min(max(p, P_CLIP), 1 - P_CLIP)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def replay_prob(p0: float, contribs: list[float], *, lam: float = 1.0, cap_answer: float | None = None,
                cap_total: float | None = None, tau: float = 0.0) -> float:
    """Posterior from p0 and the log-LRs of the answers that update it.

    Each log-LR is clipped to +-cap_answer, summed, the sum clipped to +-cap_total, and the alert keeps p0 unless |sum| > tau
    (abstain rule). Otherwise sigmoid(logit(p0) + lam * sum). No contributions = p0 exactly (as in the live run).
    """
    if not contribs:
        return p0
    c = [max(-cap_answer, min(cap_answer, x)) for x in contribs] if cap_answer is not None else list(contribs)
    s = sum(c)
    if cap_total is not None:
        s = max(-cap_total, min(cap_total, s))
    if abs(s) <= tau:
        return p0
    return sigmoid(logit(p0) + lam * s)


def select_contribs(steps: list[dict], *, accepted_only: bool = True, drop_families: frozenset | set = frozenset(),
                    include_nonreply: bool = False) -> list[float]:
    """The log-LRs a variant uses from one alert's logged steps.

    accepted_only=True: the logged `log_lr` (the live fusion uses accepted entries of replies only).
    accepted_only=False: also the table LR of unaccepted (question, eps) entries (`table_lr`).
    include_nonreply=True would also let timeouts / declines update; they have no answer value so nothing is added
    (reported as a no-op by construction and checked against the logs).
    """
    out = []
    for s in steps:
        if s["family"] in drop_families:
            continue
        v = s["log_lr"] if accepted_only else s["table_log_lr"]
        if v is None:
            continue
        if s["response_kind"] != "reply" and not include_nonreply:
            continue
        out.append(v)
    return out


def brier(y, p) -> float:
    y, p = np.asarray(y, "f8"), np.asarray(p, "f8")
    return float(np.mean((p - y) ** 2))


def fit_grid(score, grid):
    """Smallest score wins; ties go to the earlier grid entry."""
    best = None
    for g in grid:
        s = score(g)
        if best is None or s < best[1] - 1e-15:
            best = (g, s)
    return best


def logistic_offset_slope(y, offset, x, iters: int = 50) -> float:
    """MLE of b in P(y=1) = sigmoid(offset + b*x) (1-D Newton, tiny ridge). b = 1: the claimed LRs are right; 0: no information."""
    y, offset, x = (np.asarray(a, "f8") for a in (y, offset, x))
    b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(offset + b * x)))
        g = float(np.sum((y - p) * x)) - 1e-6 * b
        h = float(np.sum(p * (1 - p) * x * x)) + 1e-6
        step = g / h
        b += max(-5.0, min(5.0, step))
        if abs(step) < 1e-9:
            break
    return b


def logistic_ab(y, z, iters: int = 100, ridge: float = 1e-3) -> tuple[float, float]:
    """MLE of (a, b) in P(y=1) = sigmoid(a + b*z) (Newton, small ridge on b toward 1 only through the tiny penalty)."""
    y, z = np.asarray(y, "f8"), np.asarray(z, "f8")
    th = np.array([0.0, 1.0])
    X = np.column_stack([np.ones_like(z), z])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ th)))
        g = X.T @ (y - p) - ridge * (th - np.array([0.0, 1.0]))
        H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(2)
        d = np.linalg.solve(H, g)
        th = th + np.clip(d, -3, 3)
        if np.max(np.abs(d)) < 1e-9:
            break
    return float(th[0]), float(th[1])


# ---------------------------------------------------------------------------------------------------------------------
# loading
def _parse_value(s):
    if s is None:
        return None
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    try:
        return float(s)
    except ValueError:
        return s


def load_table() -> LRTable:
    return LRTable.from_rows([json.loads(ln) for ln in TABLE.read_text().splitlines() if ln])


def load_run(dirs: list[Path], table: LRTable) -> list[dict]:
    """One dict per alert: label, p0, prob_raw (None = no verdict), prob (end-to-end: no verdict = p0), steps."""
    alerts: dict[str, dict] = {}
    for root in dirs:
        for c in sorted(root.glob("c[0-9][0-9][0-9]")):
            if not (c / "summary.json").exists():
                continue
            for r in e18.read_per_alert(c):
                alerts[r["alert_id"]] = {"alert_id": r["alert_id"], "label": int(r["label"]), "p0": float(r["p0"]),
                                         "prob_raw": r["probability"], "steps": [], "chunk": f"{root.name}/{c.name}"}
            df = pl.read_parquet(c / "runlog.parquet").filter(pl.col("event") == "step").with_row_index("_row")  # time-guard: not transactions
            for s in df.sort(["alert_id", "step", "_row"]).to_dicts():
                a = alerts.get(s["alert_id"])
                if a is None:
                    continue
                ent = table.get(s["program_canonical"], s["eps"]) if s["program_canonical"] else None
                val = _parse_value(s["answer_value"])
                tl = None
                if ent is not None and s["response_kind"] == "reply" and s["gate"] == "allowed":
                    lr = ent.lr_for(val)
                    tl = math.log(lr) if lr is not None and lr > 0 else None
                a["steps"].append({"step": s["step"], "program": s["program_canonical"], "eps": s["eps"], "gate": s["gate"],
                                   "response_kind": s["response_kind"] or "none", "value": s["answer_value"],
                                   "posterior": s["posterior"], "log_lr": s["log_lr"], "family": s["family"] or "none",
                                   "qkind": s["kind"], "accepted": bool(ent.accepted) if ent else False, "in_table": ent is not None,
                                   "table_log_lr": tl})
    out = list(alerts.values())
    for a in out:
        a["prob"] = a["p0"] if a["prob_raw"] is None else float(a["prob_raw"])
        a["nq"] = sum(1 for s in a["steps"] if s["gate"] == "allowed")
    return out


def arr(alerts, key):
    return np.array([a[key] for a in alerts], dtype="f8")


def clusters_of(alerts, mapping):
    return e18.clusters_for(tuple(a["alert_id"] for a in alerts), mapping)[0]


def diff(alerts, p, mapping, n_boot, p_ref=None) -> dict:
    y = arr(alerts, "label")
    ref = arr(alerts, "p0") if p_ref is None else np.asarray(p_ref, "f8")
    d = bs.brier_diff(y, np.asarray(p, "f8"), ref, clusters_of(alerts, mapping), n_boot=n_boot)
    return {"brier": brier(y, p), "ref": brier(y, ref), "diff": d["diff"], "lo": d["ci"][0], "hi": d["ci"][1]}


def fmt_d(d: dict) -> str:
    return f"{d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"


def md_table(head: list[str], rows: list[list]) -> str:
    L = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    L += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------------------------------
# decomposition
def bucket_of(p0: float) -> str:
    for lo, hi in P0_BUCKETS:
        if lo <= p0 < hi:
            return f"{lo:g}-{min(hi, 1):g}"
    return "?"


def split_table(alerts, p, key_fn, order=None) -> list[dict]:
    """Contribution of each group to the total dBrier = mean over ALL alerts of (p-y)^2 - (p0-y)^2 (the groups sum to the total)."""
    n = len(alerts)
    groups = defaultdict(list)
    for a, pi in zip(alerts, p, strict=True):
        d = (pi - a["label"]) ** 2 - (a["p0"] - a["label"]) ** 2
        groups[key_fn(a)].append((d, abs(pi - a["p0"]) > 1e-9, pi - a["p0"], a["label"]))
    keys = order or sorted(groups)
    rows = []
    for k in keys:
        g = groups.get(k, [])
        if not g:
            continue
        rows.append({"group": str(k), "n": len(g), "moved": int(sum(x[1] for x in g)), "dBrier_contrib": sum(x[0] for x in g) / n,
                     "dBrier_mean": float(np.mean([x[0] for x in g])), "mean_shift_to_y": float(np.mean([(x[2]) * (1 if x[3] else -1) for x in g]))})
    return rows


def step_attribution(alerts) -> list[dict]:
    """Telescoped per-step dBrier: (post_i - y)^2 - (post_{i-1} - y)^2, sums to the fusion posterior's dBrier. Fusion modes only."""
    rows = []
    for a in alerts:
        prev = a["p0"]
        for s in a["steps"]:
            post = s["posterior"]
            if post is None or s["gate"] != "allowed":
                d = 0.0
                post = prev
            else:
                d = (post - a["label"]) ** 2 - (prev - a["label"]) ** 2
            rows.append({**s, "alert_id": a["alert_id"], "label": a["label"], "d": d, "p_before": prev, "p_after": post})
            prev = post
    return rows


def group_steps(rows: list[dict], key_fn, n_alerts: int) -> list[dict]:
    g = defaultdict(list)
    for r in rows:
        g[key_fn(r)].append(r)
    out = []
    for k in sorted(g, key=str):
        rs = g[k]
        upd = [r for r in rs if r["log_lr"] is not None]
        hit = [r for r in upd if (r["log_lr"] > 0) == (r["label"] == 1)]
        out.append({"group": str(k), "steps": len(rs), "updating": len(upd), "dBrier_contrib": sum(r["d"] for r in rs) / n_alerts,
                    "mean_log_lr": float(np.mean([r["log_lr"] for r in upd])) if upd else None,
                    "direction_hit": (len(hit) / len(upd)) if upd else None})
    return out


def sign_check(rows: list[dict]) -> dict:
    """Step level: among updating answers, mean(y - posterior before the step) when the claimed log-LR is > 0 and when it is < 0.
    With correct, independent LRs the first is > 0 and the second < 0."""
    up = [r for r in rows if r["log_lr"] is not None]
    pos = [r["label"] - r["p_before"] for r in up if r["log_lr"] > 0]
    neg = [r["label"] - r["p_before"] for r in up if r["log_lr"] < 0]
    return {"n_up": len(pos), "mean_y_minus_p_claimed_up": float(np.mean(pos)) if pos else None,
            "n_down": len(neg), "mean_y_minus_p_claimed_down": float(np.mean(neg)) if neg else None}


def concentration(alerts, p) -> dict:
    d = np.array([(pi - a["label"]) ** 2 - (a["p0"] - a["label"]) ** 2 for a, pi in zip(alerts, p, strict=True)])
    n = len(d)
    order = np.argsort(-d)
    pos = d[d > 0].sum()
    neg = d[d < 0].sum()
    top = [{"alert_id": alerts[i]["alert_id"], "label": alerts[i]["label"], "p0": round(alerts[i]["p0"], 4), "p": round(float(p[i]), 4),
            "d": round(float(d[i]), 4)} for i in order[:5]]
    return {"n_moved": int((np.abs(d) > 1e-12).sum()), "n_worse": int((d > 1e-12).sum()), "n_better": int((d < -1e-12).sum()),
            "sum_worse_over_n": float(pos / n), "sum_better_over_n": float(neg / n), "total": float(d.sum() / n),
            "top5_share_of_worse": float(d[order[:5]].clip(0).sum() / pos) if pos > 0 else None,
            "total_without_top5": float((d.sum() - d[order[:5]].sum()) / n), "top5": top}


def lr_calibration(alerts, mapping, n_boot: int) -> dict:
    """Do the summed realised log-LRs predict labels as much as they claim? lam_hat: slope of logit(p0)+lam*sum (1 = as claimed)."""
    y = arr(alerts, "label")
    off = np.array([logit(a["p0"]) for a in alerts])
    x = np.array([sum(select_contribs(a["steps"])) for a in alerts])
    m = x != 0
    out = {"n_alerts": len(alerts), "n_with_update": int(m.sum())}
    if m.sum() < 5:
        return out
    lam = logistic_offset_slope(y, off, x)
    cl = clusters_of(alerts, mapping)
    rng = np.random.default_rng(0)
    uniq = np.unique(cl)
    members = [np.flatnonzero(cl == u) for u in uniq]
    bsl = []
    for _ in range(n_boot):
        idx = np.concatenate([members[j] for j in rng.integers(0, len(uniq), len(uniq))])
        bsl.append(logistic_offset_slope(y[idx], off[idx], x[idx]))
    lo, hi = bs.percentile_ci(bsl)
    out.update(lam_hat=lam, lam_lo=lo, lam_hi=hi, mean_abs_sum_loglr=float(np.abs(x[m]).mean()),
               mean_label_when_sum_pos=float(y[x > 0].mean()) if (x > 0).any() else None, n_pos=int((x > 0).sum()),
               mean_p0_when_sum_pos=float(np.mean([a["p0"] for a, v in zip(alerts, x, strict=True) if v > 0])) if (x > 0).any() else None,
               mean_label_when_sum_neg=float(y[x < 0].mean()) if (x < 0).any() else None, n_neg=int((x < 0).sum()),
               mean_p0_when_sum_neg=float(np.mean([a["p0"] for a, v in zip(alerts, x, strict=True) if v < 0])) if (x < 0).any() else None)
    return out


def reliability(alerts, bins=(0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.01)) -> list[dict]:
    p0, y = arr(alerts, "p0"), arr(alerts, "label")
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        m = (p0 >= lo) & (p0 < hi)
        if m.any():
            rows.append({"bin": f"{lo:g}-{min(hi, 1):g}", "n": int(m.sum()), "mean_p0": float(p0[m].mean()), "label_rate": float(y[m].mean())})
    return rows


# ---------------------------------------------------------------------------------------------------------------------
# replay
def prob_vector(alerts, **kw) -> np.ndarray:
    sel = {k: kw.pop(k) for k in ("accepted_only", "drop_families") if k in kw}
    return np.array([replay_prob(a["p0"], select_contribs(a["steps"], **sel), **kw) for a in alerts])


def harmful_families(fit_alerts, min_used: int = 3) -> tuple[set, list[dict]]:
    """Greedy backward elimination on fit/cal Brier: drop the family whose removal lowers it most, while it lowers it."""
    y = arr(fit_alerts, "label")
    fams = sorted({s["family"] for a in fit_alerts for s in a["steps"] if s["log_lr"] is not None})
    used = {f: sum(1 for a in fit_alerts for s in a["steps"] if s["family"] == f and s["log_lr"] is not None) for f in fams}
    dropped: set = set()
    trace = []
    cur = brier(y, prob_vector(fit_alerts))
    while True:
        best = None
        for f in fams:
            if f in dropped or used[f] < min_used:
                continue
            b = brier(y, prob_vector(fit_alerts, drop_families=dropped | {f}))
            if b < cur - 1e-9 and (best is None or b < best[1]):
                best = (f, b)
        if best is None:
            break
        dropped.add(best[0])
        trace.append({"family": best[0], "fit_brier_after": best[1], "used_answers": used[best[0]]})
        cur = best[1]
    return dropped, trace


def family_table(fit_alerts) -> list[dict]:
    """Per accepted-question family on fit/cal: used answers, mean claimed log-LR, direction hit, Brier effect of dropping it."""
    y = arr(fit_alerts, "label")
    base = brier(y, prob_vector(fit_alerts))
    fams = sorted({s["family"] for a in fit_alerts for s in a["steps"] if s["log_lr"] is not None})
    rows = []
    for f in fams:
        st = [(s, a["label"]) for a in fit_alerts for s in a["steps"] if s["family"] == f and s["log_lr"] is not None]
        hit = sum((s["log_lr"] > 0) == (lab == 1) for s, lab in st)
        rows.append({"family": f, "used": len(st), "mean_log_lr": float(np.mean([s["log_lr"] for s, _ in st])),
                     "direction_hit": hit / len(st), "brier_if_dropped_minus_base": brier(y, prob_vector(fit_alerts, drop_families={f})) - base})
    return rows


def replay_variants(fit_alerts, ev_by_mode: dict[str, list[dict]], mapping, n_boot: int) -> dict:
    """All variants are fixed in advance; every parameter comes from fit/cal compose. eval is only scored."""
    y = arr(fit_alerts, "label")
    res: dict = {"params": {}, "fit": {}, "eval": defaultdict(dict)}

    def score(**kw):
        return brier(y, prob_vector(fit_alerts, **kw))

    base_fit = score()
    res["fit"]["p0"] = brier(y, arr(fit_alerts, "p0"))
    res["fit"]["identity"] = base_fit
    lam, b = fit_grid(lambda g: score(lam=g), [round(i * 0.05, 2) for i in range(0, 31)])
    res["params"]["shrink"] = {"lam": lam}
    res["fit"]["shrink"] = b
    drop, trace = harmful_families(fit_alerts)
    res["params"]["drop_families"] = {"dropped": sorted(drop), "trace": trace}
    res["fit"]["drop_families"] = score(drop_families=drop)
    (ca, ct), b = fit_grid(lambda g: score(cap_answer=g[0], cap_total=g[1]),
                           [(a, t) for a in (0.1, 0.25, 0.5, 0.75, 1.0, None) for t in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, None)])
    res["params"]["cap"] = {"cap_answer": ca, "cap_total": ct}
    res["fit"]["cap"] = b
    tau, b = fit_grid(lambda g: score(tau=g), [round(i * 0.1, 1) for i in range(0, 31)])
    res["params"]["abstain"] = {"tau": tau}
    res["fit"]["abstain"] = b
    res["fit"]["all_table_entries"] = score(accepted_only=False)
    res["fit"]["all_table_entries_shrink"] = None
    lam2, b2 = fit_grid(lambda g: score(accepted_only=False, lam=g), [round(i * 0.05, 2) for i in range(0, 31)])
    res["params"]["all_table_entries_shrink"] = {"lam": lam2}
    res["fit"]["all_table_entries_shrink"] = b2
    nonreply = sum(1 for a in fit_alerts for s in a["steps"] if s["response_kind"] != "reply" and s["log_lr"] is not None)
    res["params"]["nonreply_with_log_lr_in_fit"] = nonreply  # (f): 0 means timeouts/declines never update: the variant is a no-op
    res["fit"]["no_update_on_timeout_refusal"] = base_fit

    variants = {
        "identity (live posterior)": {},
        "(a) shrink lambda": {"lam": lam},
        "(b) accepted entries only [live rule; no-op]": {"accepted_only": True},
        "(b') ALL table entries incl. unaccepted, lambda=1": {"accepted_only": False},
        "(b'') all table entries, shrink lambda'": {"accepted_only": False, "lam": lam2},
        "(c) drop harmful families": {"drop_families": drop},
        "(d) cap per answer / per alert": {"cap_answer": ca, "cap_total": ct},
        "(e) abstain unless |sum| > tau": {"tau": tau},
        "(f) no update on timeout/refusal [no-op]": {},
    }
    res["variants"] = list(variants)
    fit_scores = {"identity (live posterior)": base_fit, "(a) shrink lambda": res["fit"]["shrink"],
                  "(b) accepted entries only [live rule; no-op]": base_fit, "(b') ALL table entries incl. unaccepted, lambda=1": res["fit"]["all_table_entries"],
                  "(b'') all table entries, shrink lambda'": b2, "(c) drop harmful families": res["fit"]["drop_families"],
                  "(d) cap per answer / per alert": res["fit"]["cap"], "(e) abstain unless |sum| > tau": res["fit"]["abstain"],
                  "(f) no update on timeout/refusal [no-op]": base_fit}
    res["fit_scores"] = fit_scores
    # fit/cal paired CI vs p0 for the same variants (in-sample, optimistic)
    res["fit_vs_p0"] = {}
    for name, kw in variants.items():
        res["fit_vs_p0"][name] = diff(fit_alerts, prob_vector(fit_alerts, **dict(kw)), mapping, n_boot)
    for mode, al in ev_by_mode.items():
        for name, kw in variants.items():
            p = prob_vector(al, **dict(kw))
            res["eval"][mode][name] = diff(al, p, mapping, n_boot)
    res["eval"] = dict(res["eval"])
    return res


def recal_block(fit_alerts, ev_alerts, mapping, n_boot, lam: float) -> dict:
    """Recalibrating p0 alone (NOT the planner's merit). a, b from logistic regression of y on logit(p0) over fit/cal."""
    y = arr(fit_alerts, "label")
    z = np.array([max(-Z_CAP, min(Z_CAP, logit(a["p0"]))) for a in fit_alerts])  # p0 of exactly 0/1 would give logit +-13.8 and dominate the fit
    a_, b_ = logistic_ab(y, z)

    def recal(alerts, with_updates: bool, lam_):
        out = []
        for a in alerts:
            zz = max(-Z_CAP, min(Z_CAP, logit(a["p0"]))) + (lam_ * sum(select_contribs(a["steps"])) if with_updates else 0.0)
            out.append(sigmoid(a_ + b_ * zz))
        return np.array(out)

    # intercept-free variant: only the slope (the base rate differs between fit/cal and eval)
    ys = arr(fit_alerts, "label")
    b_only = logistic_offset_slope(ys, np.zeros_like(z), z)
    out = {"a": a_, "b": b_, "b_only": b_only, "fit_brier_p0": brier(y, arr(fit_alerts, "p0")),
           "fit_brier_recal": brier(y, recal(fit_alerts, False, 1.0)),
           "fit_label_rate": float(y.mean()), "eval_label_rate": float(arr(ev_alerts, "label").mean()),
           "fit_mean_p0": float(arr(fit_alerts, "p0").mean()), "eval_mean_p0": float(arr(ev_alerts, "p0").mean())}
    out["eval_p0_recal_vs_p0"] = diff(ev_alerts, recal(ev_alerts, False, 1.0), mapping, n_boot)
    out["eval_recal_plus_updates_lam1_vs_p0"] = diff(ev_alerts, recal(ev_alerts, True, 1.0), mapping, n_boot)
    out["eval_recal_plus_updates_lam1_vs_recal_p0"] = diff(ev_alerts, recal(ev_alerts, True, 1.0), mapping, n_boot, p_ref=recal(ev_alerts, False, 1.0))
    out["eval_recal_plus_updates_shrink_vs_recal_p0"] = diff(ev_alerts, recal(ev_alerts, True, lam), mapping, n_boot, p_ref=recal(ev_alerts, False, 1.0))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# own-probability agents (section 3)
def own_prob_summary(alerts, mapping, n_boot) -> dict:
    p = np.array([a["prob"] for a in alerts])
    raw = [a["prob_raw"] for a in alerts]
    y, p0 = arr(alerts, "label"), arr(alerts, "p0")
    have = np.array([r is not None for r in raw])
    vals = np.array([r for r in raw if r is not None], "f8")
    top = defaultdict(int)
    for v in vals:
        top[round(float(v), 3)] += 1
    common = sorted(top.items(), key=lambda kv: -kv[1])[:4]
    ph = p[have]
    out = {"n": len(alerts), "verdicts": int(have.sum()), "diff_vs_p0": diff(alerts, p, mapping, n_boot),
           "mean_p_y1": float(ph[y[have] == 1].mean()), "mean_p_y0": float(ph[y[have] == 0].mean()),
           "mean_p0_y1": float(p0[have][y[have] == 1].mean()), "mean_p0_y0": float(p0[have][y[have] == 0].mean()),
           "corr_p_p0": float(np.corrcoef(ph, p0[have])[0, 1]) if have.sum() > 2 and np.std(ph) > 0 else None,
           "corr_p_y": float(np.corrcoef(ph, y[have])[0, 1]) if have.sum() > 2 and np.std(ph) > 0 else None,
           "corr_p0_y": float(np.corrcoef(p0[have], y[have])[0, 1]),
           "most_common_values": [{"p": k, "share": v / have.sum()} for k, v in common],
           "share_p_ge_0.8": float((ph >= 0.8).mean()), "mean_abs_p_minus_p0": float(np.abs(ph - p0[have]).mean()),
           "brier_on_verdict_alerts": brier(y[have], ph), "p0_brier_on_verdict_alerts": brier(y[have], p0[have])}
    return out


def belief_summary(alerts, mapping, n_boot) -> dict:
    p, p0 = np.array([a["prob"] for a in alerts]), arr(alerts, "p0")
    shadow = prob_vector(alerts)
    moved = np.abs(p - p0) > 1e-9
    return {"n": len(alerts), "diff_vs_p0": diff(alerts, p, mapping, n_boot), "moved": int(moved.sum()),
            "mean_abs_shift": float(np.abs(p - p0).mean()), "corr_p_p0": float(np.corrcoef(p, p0)[0, 1]),
            "diff_formula_shadow_vs_p0": diff(alerts, shadow, mapping, n_boot),
            "diff_belief_vs_shadow": diff(alerts, p, mapping, n_boot, p_ref=shadow)}


# ---------------------------------------------------------------------------------------------------------------------
def build(n_boot: int) -> tuple[dict, str]:
    table = load_table()
    mapping = e18.load_clusters(CLUSTERS)
    fit = load_run(FIT, table)
    ev = {m.name: load_run([m], table) for m in sorted(EVAL.iterdir()) if m.is_dir() and (m.name in FUSION_MODES + BELIEF_MODES + OWN_PROB)}
    for m, al in ev.items():
        assert len(al) == 160, (m, len(al))
    acc_eps: dict = {}
    for e in table.entries():
        d = acc_eps.setdefault(f"{e.eps:g}", {"entries": 0, "accepted": 0, "max_abs_log_lr_accepted": 0.0})
        d["entries"] += 1
        if e.accepted:
            d["accepted"] += 1
            d["max_abs_log_lr_accepted"] = max(d["max_abs_log_lr_accepted"], max(abs(math.log(v)) for v in e.lr.values() if v > 0))
    res: dict = {"n_boot": n_boot, "table": {"rows": len(table), "accepted": sum(e.accepted for e in table.entries()), "by_eps": acc_eps},
                 "n_fit": len(fit), "n_eval": 160}
    md: list[str] = ["### Table 0. Back-test table entries by eps", "",
                     md_table(["eps", "(question, eps) entries", "accepted", "max |log LR| among accepted"],
                              [[k, v["entries"], v["accepted"], f"{v['max_abs_log_lr_accepted']:.2f}"] for k, v in sorted(acc_eps.items(), key=lambda kv: float(kv[0]))]), ""]

    # reproduction check: the replay of the logged log-LRs gives back the logged verdict probability (fusion modes)
    rep = {}
    for m in FUSION_MODES:
        if m in ev:
            p = prob_vector(ev[m])
            rep[m] = float(np.max(np.abs(p - arr(ev[m], "prob"))))
    rep["fit_cal_compose"] = float(np.max(np.abs(prob_vector(fit) - arr(fit, "prob"))))
    res["replay_reproduces_live_max_abs_diff"] = rep

    # --- 1a overall per mode
    overall = []
    for m in FUSION_MODES + BELIEF_MODES:
        if m not in ev:
            continue
        al = ev[m]
        p = arr(al, "prob")
        d = diff(al, p, mapping, n_boot)
        steps = [s for a in al for s in a["steps"]]
        upd = sum(1 for s in steps if s["log_lr"] is not None)
        overall.append({"mode": m, "brier": d["brier"], "diff": d["diff"], "lo": d["lo"], "hi": d["hi"],
                        "alerts_moved": int((np.abs(p - arr(al, "p0")) > 1e-9).sum()), "steps": len(steps), "steps_updating_fusion": upd,
                        "questions_per_alert": len(steps) / len(al), "updating_per_alert": upd / len(al),
                        "alerts_with_update": sum(1 for a in al if any(s["log_lr"] is not None for s in a["steps"]))})
    d = diff(fit, arr(fit, "prob"), mapping, n_boot)
    overall.append({"mode": "fit+cal ours_compose (tuning)", "brier": d["brier"], "diff": d["diff"], "lo": d["lo"], "hi": d["hi"],
                    "alerts_moved": int((np.abs(arr(fit, "prob") - arr(fit, "p0")) > 1e-9).sum()),
                    "steps": sum(len(a["steps"]) for a in fit), "steps_updating_fusion": sum(1 for a in fit for s in a["steps"] if s["log_lr"] is not None),
                    "questions_per_alert": sum(len(a["steps"]) for a in fit) / len(fit),
                    "updating_per_alert": sum(1 for a in fit for s in a["steps"] if s["log_lr"] is not None) / len(fit),
                    "alerts_with_update": sum(1 for a in fit if any(s["log_lr"] is not None for s in a["steps"]))})
    res["overall"] = overall
    md += ["### Table 1. Each mode on val `eval` (160 alerts) and compose on `fit`+`cal` (300 alerts)", "",
           md_table(["mode", "Brier", "vs p0 [95% CI]", "alerts moved", "questions/alert", "updating answers/alert", "alerts with >=1 update"],
                    [[o["mode"], f"{o['brier']:.4f}", f"{o['diff']:+.4f} [{o['lo']:+.4f}, {o['hi']:+.4f}]", o["alerts_moved"],
                      f"{o['questions_per_alert']:.2f}", f"{o['updating_per_alert']:.2f}", o["alerts_with_update"]] for o in overall]), ""]

    # --- 1b splits
    res["splits"] = {}
    sets = {"eval ours_compose": ev["ours_compose"], "fit+cal ours_compose": fit}
    for m in ("ours_fixed_list", "ours_act_only", "ours_llm_decides"):
        sets[f"eval {m}"] = ev[m]
    md += ["### Table 2. Where the dBrier comes from (contribution to the total = group sum / all alerts; groups add to the total)", ""]
    for name, al in sets.items():
        p = arr(al, "prob")
        res["splits"][name] = {}
        for dim, fn, order in (("label", lambda a: f"y={a['label']}", None), ("p0 bucket", lambda a: bucket_of(a["p0"]), [f"{lo:g}-{min(hi, 1):g}" for lo, hi in P0_BUCKETS]),
                               ("questions asked", lambda a: "3+" if a["nq"] >= 3 else str(a["nq"]), ["0", "1", "2", "3+"])):
            t = split_table(al, p, fn, order)
            res["splits"][name][dim] = t
            md += [f"**{name}, by {dim}**", "", md_table([dim, "alerts", "moved", "dBrier contribution", "mean dBrier per alert", "mean shift toward the label"],
                                                        [[r["group"], r["n"], r["moved"], f"{r['dBrier_contrib']:+.4f}", f"{r['dBrier_mean']:+.4f}", f"{r['mean_shift_to_y']:+.4f}"] for r in t]), ""]

    # --- 1c step-level attribution (fusion modes: telescoped)
    res["steps"] = {}
    md += ["### Table 3. Step-level attribution (telescoped dBrier of the formula posterior; steps that did not move it contribute 0)", ""]
    for name, al in (("eval ours_compose", ev["ours_compose"]), ("fit+cal ours_compose", fit), ("eval ours_fixed_list", ev["ours_fixed_list"]),
                     ("eval ours_act_only", ev["ours_act_only"])):
        rows = step_attribution(al)
        res["steps"][name] = {}
        md += [f"**{name}**", ""]
        for dim, fn in (("response", lambda r: r["response_kind"] if r["gate"] == "allowed" else r["gate"]),
                        ("in back-test table / accepted", lambda r: "accepted" if r["accepted"] else ("in table, rejected" if r["in_table"] else "not in table")),
                        ("eps", lambda r: f"{r['eps']:g}" if r["eps"] is not None else "none"), ("family (updating or asked)", lambda r: r["family"])):
            t = group_steps(rows, fn, len(al))
            if dim.startswith("family"):
                t = [r for r in t if r["updating"] > 0]
            res["steps"][name][dim] = t
            md += [md_table([dim, "steps", "updating", "dBrier contribution", "mean claimed log-LR", "direction hit"],
                            [[r["group"], r["steps"], r["updating"], f"{r['dBrier_contrib']:+.4f}",
                              "" if r["mean_log_lr"] is None else f"{r['mean_log_lr']:+.2f}",
                              "" if r["direction_hit"] is None else f"{r['direction_hit']:.2f}"] for r in t]), ""]
    res["sign_check"] = {}
    for name, al in (("fit+cal ours_compose", fit), ("eval ours_compose", ev["ours_compose"]), ("eval ours_fixed_list", ev["ours_fixed_list"]),
                     ("eval ours_act_only", ev["ours_act_only"])):
        rows = step_attribution(al)
        res["sign_check"][name] = {"all": sign_check(rows), "eps 1": sign_check([r for r in rows if r["eps"] == 1.0]),
                                   "eps 2": sign_check([r for r in rows if r["eps"] == 2.0])}
    md += ["### Table 3b. Sign check per updating answer: mean(label - posterior before) when the claimed log-LR is positive / negative "
           "(correct independent LRs give > 0 / < 0)", "",
           md_table(["run", "eps", "answers claimed up", "mean(y - p) up", "answers claimed down", "mean(y - p) down"],
                    [[n, k, v["n_up"], "" if v["mean_y_minus_p_claimed_up"] is None else f"{v['mean_y_minus_p_claimed_up']:+.3f}", v["n_down"],
                      "" if v["mean_y_minus_p_claimed_down"] is None else f"{v['mean_y_minus_p_claimed_down']:+.3f}"]
                     for n, d in res["sign_check"].items() for k, v in d.items()]), ""]
    # consistency checks stated in the doc
    chk = {}
    for name, al in (("eval ours_compose", ev["ours_compose"]), ("fit+cal ours_compose", fit)):
        steps = [s for a in al for s in a["steps"]]
        chk[name] = {"updating_steps": sum(s["log_lr"] is not None for s in steps),
                     "updating_steps_not_accepted": sum(s["log_lr"] is not None and not s["accepted"] for s in steps),
                     "updating_nonreply": sum(s["log_lr"] is not None and s["response_kind"] != "reply" for s in steps),
                     "accepted_replies_without_log_lr": sum(s["accepted"] and s["response_kind"] == "reply" and s["log_lr"] is None for s in steps),
                     "accepted_steps": sum(s["accepted"] for s in steps), "steps": len(steps),
                     "steps_in_table_rejected": sum(s["in_table"] and not s["accepted"] for s in steps),
                     "steps_not_in_table": sum(not s["in_table"] for s in steps)}
    res["checks"] = chk

    # --- 1d concentration
    res["concentration"] = {name: concentration(al, arr(al, "prob")) for name, al in sets.items()}
    md += ["### Table 4. Concentration of the loss (alerts whose posterior moved)", "",
           md_table(["run", "moved", "worse", "better", "sum worse / N", "sum better / N", "total", "top-5 share of worse", "total without worst 5"],
                    [[n, c["n_moved"], c["n_worse"], c["n_better"], f"{c['sum_worse_over_n']:+.4f}", f"{c['sum_better_over_n']:+.4f}", f"{c['total']:+.4f}",
                      "" if c["top5_share_of_worse"] is None else f"{c['top5_share_of_worse']:.2f}", f"{c['total_without_top5']:+.4f}"] for n, c in res["concentration"].items()]), ""]
    cc = res["concentration"]["eval ours_compose"]["top5"]
    md += ["Worst five alerts, eval ours_compose:", "", md_table(["alert", "label", "p0", "p final", "dBrier"], [[t["alert_id"], t["label"], t["p0"], t["p"], f"{t['d']:+.4f}"] for t in cc]), ""]

    # --- 1e LR calibration
    res["lr_calibration"] = {"fit+cal compose": lr_calibration(fit, mapping, n_boot), "eval compose": lr_calibration(ev["ours_compose"], mapping, n_boot),
                             "eval fixed_list": lr_calibration(ev["ours_fixed_list"], mapping, n_boot), "eval act_only": lr_calibration(ev["ours_act_only"], mapping, n_boot)}
    pool = ev["ours_compose"] + ev["ours_fixed_list"] + ev["ours_act_only"]
    md += ["### Table 5. Do the realised summed log-LRs predict the labels as claimed? (slope lambda of logit(p0) + lambda * sum; 1 = as claimed, 0 = no information)", "",
           md_table(["run", "alerts with update", "lambda_hat [95% cluster CI]", "mean |sum log-LR|", "alerts sum>0: n, label rate, mean p0", "alerts sum<0: n, label rate, mean p0"],
                    [[n, c["n_with_update"], "" if "lam_hat" not in c else f"{c['lam_hat']:+.2f} [{c['lam_lo']:+.2f}, {c['lam_hi']:+.2f}]",
                      "" if "lam_hat" not in c else f"{c['mean_abs_sum_loglr']:.2f}",
                      "" if "lam_hat" not in c or not c["n_pos"] else f"{c['n_pos']}, {c['mean_label_when_sum_pos']:.2f}, {c['mean_p0_when_sum_pos']:.2f}",
                      "" if "lam_hat" not in c or not c["n_neg"] else f"{c['n_neg']}, {c['mean_label_when_sum_neg']:.2f}, {c['mean_p0_when_sum_neg']:.2f}"]
                     for n, c in res["lr_calibration"].items()]), ""]
    del pool
    fam_rows = family_table(fit)
    res["family_table_fit"] = fam_rows
    md += ["### Table 6. Updating families on `fit`+`cal` (compose): answers used, claimed mean log-LR, share whose sign matches the label, and Brier change if the family is dropped", "",
           md_table(["family", "used answers", "mean claimed log-LR", "direction hit", "dBrier if dropped"],
                    [[r["family"], r["used"], f"{r['mean_log_lr']:+.2f}", f"{r['direction_hit']:.2f}", f"{r['brier_if_dropped_minus_base']:+.4f}"] for r in fam_rows]), ""]

    # --- 1f p0 reliability
    allev = ev["ours_compose"]
    res["p0_reliability"] = {"fit+cal": reliability(fit), "eval": reliability(allev)}
    md += ["### Table 7. Reliability of p0 on the askable alerts", ""]
    for n, al in (("fit+cal (300 alerts)", fit), ("eval (160 alerts)", allev)):
        md += [f"**{n}**", "", md_table(["p0 bin", "alerts", "mean p0", "label rate"], [[r["bin"], r["n"], f"{r['mean_p0']:.3f}", f"{r['label_rate']:.3f}"] for r in reliability(al)]), ""]

    # --- 2 replay
    ev_modes = {m: ev[m] for m in FUSION_MODES + BELIEF_MODES if m in ev}
    rv = replay_variants(fit, ev_modes, mapping, n_boot)
    res["replay"] = rv
    md += ["### Table 8. Replay variants, parameters fitted on `fit`+`cal` (compose, 300 alerts), scored once on eval", "",
           md_table(["variant", "chosen on fit/cal", "fit/cal Brier (p0 = {:.4f})".format(rv["fit"]["p0"]), "eval compose Brier", "eval compose vs p0 [CI]", "eval llm_decides formula shadow vs p0 [CI]"],
                    [[v, json.dumps({k: x for k, x in (rv["params"].get({"(a) shrink lambda": "shrink", "(c) drop harmful families": "drop_families",
                                                                         "(d) cap per answer / per alert": "cap", "(e) abstain unless |sum| > tau": "abstain",
                                                                         "(b'') all table entries, shrink lambda'": "all_table_entries_shrink"}.get(v, ""), {})).items() if k != "trace"}) if v in (
                        "(a) shrink lambda", "(c) drop harmful families", "(d) cap per answer / per alert", "(e) abstain unless |sum| > tau", "(b'') all table entries, shrink lambda'") else "",
                      f"{rv['fit_scores'][v]:.4f}", f"{rv['eval']['ours_compose'][v]['brier']:.4f}", fmt_d(rv["eval"]["ours_compose"][v]),
                      fmt_d(rv["eval"]["ours_llm_decides"][v])] for v in rv["variants"]]), ""]
    md += ["Same variants on the other modes (eval, vs p0 [95% CI]). For llm_decides and no_backtest the verdict is the model's own belief; the replay "
           "is of the formula posterior over the questions they asked, so the row is a different system (formula verdict) from the logged one.", "",
           md_table(["variant"] + [m.replace("ours_", "") for m in ev_modes],
                    [[v] + [fmt_d(rv["eval"][m][v]) for m in ev_modes] for v in rv["variants"]]), ""]

    # --- p0 recalibration, separately
    lam = rv["params"]["shrink"]["lam"]
    res["recal"] = recal_block(fit, ev["ours_compose"], mapping, n_boot, lam)
    rc = res["recal"]
    md += ["### Table 9. Recalibrating p0 alone (not the planner's merit), logistic recalibration fitted on fit/cal", "",
           md_table(["quantity", "value"], [
               ["fit/cal: intercept a, slope b on logit(p0)", f"{rc['a']:+.3f}, {rc['b']:.3f}"],
               ["fit/cal: slope-only fit (no intercept)", f"{rc['b_only']:.3f}"],
               ["label rate fit/cal vs eval", f"{rc['fit_label_rate']:.3f} vs {rc['eval_label_rate']:.3f}"],
               ["mean p0 fit/cal vs eval", f"{rc['fit_mean_p0']:.3f} vs {rc['eval_mean_p0']:.3f}"],
               ["fit/cal Brier p0 -> recalibrated p0", f"{rc['fit_brier_p0']:.4f} -> {rc['fit_brier_recal']:.4f}"],
               ["eval: recalibrated p0 vs p0", fmt_d(rc["eval_p0_recal_vs_p0"])],
               ["eval: recalibrated p0 + updates (lambda=1) vs p0", fmt_d(rc["eval_recal_plus_updates_lam1_vs_p0"])],
               ["eval: recalibrated p0 + updates (lambda=1) vs recalibrated p0 alone", fmt_d(rc["eval_recal_plus_updates_lam1_vs_recal_p0"])],
               ["eval: recalibrated p0 + shrunk updates vs recalibrated p0 alone", fmt_d(rc["eval_recal_plus_updates_shrink_vs_recal_p0"])]]), ""]

    # --- 3 own-probability agents and llm_decides belief
    res["own_prob"] = {m: own_prob_summary(ev[m], mapping, n_boot) for m in OWN_PROB if m in ev}
    res["belief"] = {m: belief_summary(ev[m], mapping, n_boot) for m in BELIEF_MODES if m in ev}
    md += ["### Table 10. Agents that output their own probability (eval)", "",
           md_table(["agent", "verdicts / 160", "Brier vs p0 [CI]", "mean p (y=1 / y=0)", "mean p0 (y=1 / y=0)", "corr(p, p0)", "corr(p, y) vs corr(p0, y)", "most common p", "share p>=0.8", "mean |p - p0|"],
                    [[m, f"{o['verdicts']}", fmt_d(o["diff_vs_p0"]), f"{o['mean_p_y1']:.2f} / {o['mean_p_y0']:.2f}", f"{o['mean_p0_y1']:.2f} / {o['mean_p0_y0']:.2f}",
                      "" if o["corr_p_p0"] is None else f"{o['corr_p_p0']:+.2f}", "" if o["corr_p_y"] is None else f"{o['corr_p_y']:+.2f} vs {o['corr_p0_y']:+.2f}",
                      ", ".join(f"{c['p']:g} ({c['share']:.0%})" for c in o["most_common_values"][:3]), f"{o['share_p_ge_0.8']:.2f}", f"{o['mean_abs_p_minus_p0']:.2f}"]
                     for m, o in res["own_prob"].items()]), ""]
    md += ["### Table 11. Belief-verdict modes (eval): the model's belief against the formula posterior over the same questions", "",
           md_table(["mode", "belief vs p0", "formula shadow vs p0", "belief vs shadow", "alerts where belief != p0", "mean |belief - p0|", "corr(belief, p0)"],
                    [[m, fmt_d(o["diff_vs_p0"]), fmt_d(o["diff_formula_shadow_vs_p0"]), fmt_d(o["diff_belief_vs_shadow"]), o["moved"], f"{o['mean_abs_shift']:.3f}", f"{o['corr_p_p0']:+.2f}"]
                     for m, o in res["belief"].items()]), ""]
    return res, "\n".join(md)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "t3_val" / "analysis")
    a = ap.parse_args(argv)
    res, md = build(a.n_boot)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "analysis.json").write_text(json.dumps(res, indent=1, default=str))
    (a.out_dir / "tables.md").write_text(md + "\n")
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
