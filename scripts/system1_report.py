"""System 1 integration report: with-vs-without arms, full metric set, paired cluster-bootstrap CIs, Holm, H5 per clause.

Owner: T3 (Akshat).  Spec: docs/t3_system1.md (the selection rule is written there, before any run).  CPU only; reads run directories.

    # choose on val fit+cal (writes the selection)
    python scripts/system1_report.py report --part fitcal --off off=outputs/s1/fitcal/off/fit,outputs/s1/fitcal/off/cal \\
        --arm worth=outputs/s1/fitcal/worth/fit,outputs/s1/fitcal/worth/cal --arm inject=... --arm typology=... \\
        --arm all_fast=... --arm all=... --offline outputs/s1v1/report.json --out outputs/s1/fitcal_report
    # eval, once, with the frozen selection
    python scripts/system1_report.py report --part eval --selection outputs/s1/fitcal_report/selection.json ... --out outputs/s1/eval_report

An arm = the run directories of its parts (comma separated); each holds per_alert.jsonl (label, p0, probability, seconds, llm_calls, tokens,
eps), runlog.parquet (one `step` row per question), and with System 1 on, system1.jsonl.  Alerts are paired by alert_id; p0 fills a
missing verdict.  Clusters: attempt+accounts file (--clusters), else row-level (labelled, CIs too narrow).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftm.eval import bootstrap as bs  # noqa: E402
from ftm.eval import metrics as M  # noqa: E402
from ftm.eval.stats import holm  # noqa: E402

THRESHOLD = 0.322699  # configs/decision/headroom_v1.yaml
DELTA = 0.002  # non-inferiority margin on the paired Brier difference (absolute); docs/t3_system1.md
ALPHA = 0.05
MIN_LARGE_SHARE = 0.05  # the full cascade is chosen over fast-only only if the large tier settles at least this share of decisions
MAX_QUARANTINE_BENIGN = 0.05
P_CLIP = 1e-6


# --- metrics -------------------------------------------------------------------------------------------------------
def auc(y, p):
    y, p = np.asarray(y), np.asarray(p, "f8")
    if y.min() == y.max():
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def logloss(y, p) -> float:
    y, p = np.asarray(y, "f8"), np.clip(np.asarray(p, "f8"), P_CLIP, 1 - P_CLIP)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def cal_slope(y, p) -> float:
    """Slope of a logistic recalibration y ~ a + b*logit(p) (1 = calibrated, < 1 = overconfident). Newton, ridge 1e-6."""
    y = np.asarray(y, "f8")
    z = np.log(np.clip(p, P_CLIP, 1 - P_CLIP) / (1 - np.clip(p, P_CLIP, 1 - P_CLIP)))
    X = np.column_stack([np.ones(len(z)), z])
    w = np.array([0.0, 1.0])
    for _ in range(50):
        q = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (q - y) + 1e-6 * (w - np.array([0.0, 1.0]))
        H = X.T @ (X * (q * (1 - q))[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        w = w - step
        if np.abs(step).max() < 1e-8:
            break
    return float(w[1])


def prf(y, p, thr=THRESHOLD) -> dict:
    y, flag = np.asarray(y), np.asarray(p) >= thr
    tp = int((flag & (y == 1)).sum())
    prec = tp / flag.sum() if flag.sum() else float("nan")
    rec = tp / (y == 1).sum() if (y == 1).sum() else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec > 0 else float("nan")
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1), "escalation_rate": float(flag.mean())}


def metric_row(y, p) -> dict:
    return {"n": int(len(y)), "n_pos": int(np.sum(y)), "brier": M.brier(y, p), "auc": auc(y, p), "pr_auc": M.pr_auc(y, p),
            "log_loss": logloss(y, p), "ece": M.ece(y, p), "cal_slope": cal_slope(y, p), **prf(y, p)}  # fmt: skip


# --- loading -------------------------------------------------------------------------------------------------------
def load_arm(dirs: list[Path]) -> dict:
    """alert_id-keyed columns of one arm (its parts concatenated) + its System 1 records."""
    import polars as pl

    rows, s1, qcount = [], [], {}
    for d in dirs:
        rows += [json.loads(x) for x in (d / "per_alert.jsonl").read_text().splitlines() if x.strip()]
        if (d / "system1.jsonl").exists():
            s1 += [json.loads(x) for x in (d / "system1.jsonl").read_text().splitlines() if x.strip()]
        rl = pl.read_parquet(d / "runlog.parquet")  # time-guard: not transactions
        steps = rl.filter(pl.col("event") == "step")
        for aid, n in steps.group_by("alert_id").len().iter_rows():
            qcount[aid] = qcount.get(aid, 0) + n
    out = {r["alert_id"]: r for r in rows}
    for aid, r in out.items():
        r["questions"] = qcount.get(aid, 0)
        r["p"] = r["p0"] if r.get("probability") is None else r["probability"]
    return {"alerts": out, "s1": s1}


def clusters_for(ids: list[str], path: Path | None):
    if path is None or not path.exists():
        return np.arange(len(ids)), "row-level (no --clusters file: CIs too narrow)"
    from ftm.planners.e18 import clusters_for as cf
    from ftm.planners.e18 import load_clusters

    return cf(tuple(ids), load_clusters(path))


def _col(arm, ids, key):
    return np.array([arm["alerts"][i][key] if arm["alerts"][i].get(key) is not None else np.nan for i in ids], "f8")


# --- paired contrasts ------------------------------------------------------------------------------------------------
def paired(off, arm, ids, cl, n_boot) -> dict:
    y = _col(off, ids, "label")
    po, pa = _col(off, ids, "p"), _col(arm, ids, "p")
    out = {}
    b = bs.brier_diff(y, pa, po, cl, n_boot=n_boot)
    out["brier_diff"] = {"diff": b["diff"], "ci": b["ci"], "p_noninferior": bs.bootstrap_pvalue(b["samples"], DELTA, "less"),
                         "p_superior": bs.bootstrap_pvalue(b["samples"], 0.0, "less")}  # fmt: skip
    for name, f in (("auc_diff", lambda i: auc(y[i], pa[i]) - auc(y[i], po[i])), ("log_loss_diff", lambda i: logloss(y[i], pa[i]) - logloss(y[i], po[i]))):
        s = bs.cluster_bootstrap(f, cl, n_boot=min(n_boot, 2000))
        out[name] = {"diff": float(f(np.arange(len(y)))), "ci": bs.percentile_ci(s) if np.isfinite(s).any() else (None, None)}
    for key in ("seconds", "llm_calls", "tokens_in", "tokens_out", "questions", "eps_spent"):
        va, vo = _col(arm, ids, key), _col(off, ids, key)
        if np.isnan(va).all() or np.isnan(vo).all():
            continue
        v = np.column_stack([va - vo])
        s = bs.bootstrap_means(v, cl, n_boot, 0)[:, 0]
        out[f"{key}_per_alert_diff"] = {"diff": float(np.nanmean(va - vo)), "ci": bs.percentile_ci(s),
                                        "p_saved": bs.bootstrap_pvalue(s, 0.0, "less")}  # fmt: skip
    so, sa = _col(off, ids, "seconds"), _col(arm, ids, "seconds")
    m = bs.bootstrap_means(np.column_stack([so, sa]), cl, n_boot, 0)
    ratio = m[:, 0] / m[:, 1]
    out["speedup_mean_seconds"] = {"ratio": float(so.mean() / sa.mean()), "ci": bs.percentile_ci(ratio)}
    out["speedup_median_seconds"] = float(np.median(so) / np.median(sa))
    out["seconds"] = {"median": float(np.median(sa)), "p90": float(np.quantile(sa, 0.9)), "median_off": float(np.median(so)),
                      "p90_off": float(np.quantile(so, 0.9))}  # fmt: skip
    fl = M.flips(y, po, pa, THRESHOLD)
    out["flips_vs_off"] = {**fl, "changed_calls": int(((po >= THRESHOLD) != (pa >= THRESHOLD)).sum()), "net": fl["wrong_to_right"] - fl["right_to_wrong"]}
    return out


def tier_shares(s1: list[dict]) -> dict:
    out = {}
    for d in sorted({r["decision"] for r in s1}):
        rs = [r for r in s1 if r["decision"] == d]
        n = len(rs)
        sec = lambda t: [r["seconds"] for r in rs if r["tier"] == t]  # noqa: E731
        out[d] = {"n": n, **{f"share_{t}": sum(r["tier"] == t for r in rs) / n for t in ("fast", "large", "default")},
                  **{f"median_s_{t}": float(np.median(sec(t))) if sec(t) else None for t in ("fast", "large", "default")}}
        if d == "worth_asking":
            out[d]["settled_no"] = sum(r["label"] == "no" and r["tier"] != "default" for r in rs)
        if d == "injection_present":
            out[d]["quarantined"] = sum(bool(r.get("quarantined")) for r in rs)
    return out


# --- injection (offline, held-out families) ------------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def injection_detection(rows: list[dict]) -> dict:
    """Detection of planted instructions by the settled cascade answer on the held-out val families. Unsettled rows (default = yes =
    quarantine) count as flagged, as in the pipeline. `without_screen`: nothing is flagged (recall 0), the E15 baseline."""
    from ftm.safety.injection import family_of_note

    by: dict[str, list[tuple[bool, bool, str]]] = {}
    for r in rows:
        if r["decision"] != "injection_present":
            continue
        note = r["context"].split('this note: "', 1)[-1].rstrip('"')
        fam = family_of_note(note, "val")
        by.setdefault(fam, []).append((r["label"] == "yes", r["answer"] != "no", r["source"]))
    tot = [x for v in by.values() for x in v]

    def stats(v):
        tp = sum(t and f for t, f, _ in v)
        fp = sum((not t) and f for t, f, _ in v)
        pos, neg = sum(t for t, _, _ in v), sum(not t for t, _, _ in v)
        return {"n": len(v), "attacks": pos, "benign": neg, "recall": tp / pos if pos else None, "recall_ci": wilson(tp, pos) if pos else None,
                "precision": tp / (tp + fp) if tp + fp else None, "false_quarantine_rate": fp / neg if neg else None,
                "false_quarantine_ci": wilson(fp, neg) if neg else None,
                "settled_share": sum(s != "default" for _, _, s in v) / len(v)}  # fmt: skip

    return {"with_screen": {"all": stats(tot), **{k: stats(v) for k, v in sorted(by.items())}},
            "without_screen": {"recall": 0.0, "precision": None, "note": "no screen: every planted note reaches the planner prompt"}}


# --- selection and H5 -----------------------------------------------------------------------------------------------------
def select(res: dict, s1_by_arm: dict) -> dict:
    """The rule of docs/t3_system1.md, on fit+cal arms only."""
    fam = {a: res[a]["paired"]["brier_diff"]["p_noninferior"] for a in res if a != "off"}
    adj = holm(fam, ALPHA)
    ok = {a: bool(adj[a]["reject"]) for a in fam}
    sel = {"worth": ok.get("worth", False) and res["worth"]["paired"]["llm_calls_per_alert_diff"]["ci"][1] < 0 if "worth" in res else False}
    if "inject" in res:
        ins = [r for r in s1_by_arm["inject"] if r["decision"] == "injection_present"]
        q = sum(bool(r.get("quarantined")) for r in ins) / len(ins) if ins else 0.0
        sel["inject"] = ok["inject"] and q <= MAX_QUARANTINE_BENIGN
    if "typology" in res:
        sel["typology"] = ok["typology"] and res["typology"]["paired"]["brier_diff"]["diff"] <= 0
    tier = "fast"
    if "all" in res and "all_fast" in res and ok.get("all"):
        dec = s1_by_arm["all"]
        share_large = sum(r["tier"] == "large" for r in dec) / len(dec) if dec else 0.0
        tier = "cascade" if share_large >= MIN_LARGE_SHARE else "fast"
    return {"functions": [k for k, v in sel.items() if v], "tiers": tier, "per_function": sel, "holm": adj, "delta": DELTA, "alpha": ALPHA}


def h5(res: dict, s1: list[dict], offline: dict | None) -> list[dict]:
    """The four clauses of H5, each with its own verdict (exploratory, uncorrected)."""
    cl = []
    if offline:
        for d, r in offline["decisions"].items():
            cl.append({"clause": "bound", "decision": d, "estimate": r.get("error_marginal"), "limit": r.get("bound_2alpha"),
                       "met": bool(r.get("within_bound")), "kept_fast": r.get("kept_fast"), "fast_retired": r.get("fast_retired")})
    n = len(s1)
    fast_share = sum(r["tier"] == "fast" for r in s1) / n if n else float("nan")
    cl.append({"clause": "most routine decisions (live: share settled by the fast tier)", "estimate": fast_share, "limit": 0.5, "met": bool(fast_share > 0.5)})
    settled = sum(r["tier"] != "default" for r in s1) / n if n else float("nan")
    cl.append({"clause": "most routine decisions (live: share settled by any tier)", "estimate": settled, "limit": 0.5, "met": bool(settled > 0.5)})
    sh = tier_shares(s1)
    fast_s = [v["median_s_fast"] for v in sh.values() if v["median_s_fast"]]
    large_s = [v["median_s_large"] for v in sh.values() if v["median_s_large"]]
    if fast_s and large_s:
        ratio = float(np.median(large_s) / np.median(fast_s))
        cl.append({"clause": ">= 10x faster (decision latency, large/fast, live medians)", "estimate": ratio, "limit": 10.0, "met": ratio >= 10})
    if "all" in res:
        sp = res["all"]["paired"]["speedup_median_seconds"]
        cl.append({"clause": ">= 10x faster (end-to-end seconds per alert, median OFF / ON)", "estimate": sp, "limit": 10.0, "met": sp >= 10})
        b = res["all"]["paired"]["brier_diff"]
        cl.append({"clause": "no detection loss (paired Brier ON - OFF, 95% CI upper bound <= delta)", "estimate": b["diff"], "ci": b["ci"],
                   "limit": DELTA, "met": bool(b["ci"][1] <= DELTA)})
        fl = res["all"]["paired"]["flips_vs_off"]
        cl.append({"clause": "no detection loss (flipped calls: right-to-wrong <= wrong-to-right)", "estimate": fl, "met": fl["right_to_wrong"] <= fl["wrong_to_right"]})
    return cl


def cmd_report(a) -> int:
    parse = lambda spec: (spec.split("=", 1)[0], [Path(x) for x in spec.split("=", 1)[1].split(",")])  # noqa: E731
    name, dirs = parse(a.off)
    arms = {"off": load_arm(dirs)}
    for spec in a.arm:
        n, d = parse(spec)
        arms[n] = load_arm(d)
    ids = sorted(set.intersection(*[set(v["alerts"]) for v in arms.values()]))
    cl, cl_note = clusters_for(ids, a.clusters)
    y = _col(arms["off"], ids, "label")
    res = {}
    for n, arm in arms.items():
        p = _col(arm, ids, "p")
        row = {"metrics": metric_row(y, p), "p0_metrics": metric_row(y, _col(arm, ids, "p0")) if n == "off" else None,
               "per_alert": {k: float(np.nanmean(_col(arm, ids, k))) for k in ("questions", "eps_spent", "llm_calls", "tokens_in", "tokens_out", "seconds")},
               "tier_shares": tier_shares(arm["s1"]) if arm["s1"] else None}  # fmt: skip
        if n != "off":
            row["paired"] = paired(arms["off"], arm, ids, cl, a.n_boot)
        res[n] = row
    offline = json.loads(a.offline.read_text()) if a.offline else None
    rows_path = a.offline.with_suffix(".eval_rows.jsonl") if a.offline else None
    inj = injection_detection([json.loads(x) for x in rows_path.read_text().splitlines() if x.strip()]) if rows_path and rows_path.exists() else None
    out = {"part": a.part, "n_alerts": len(ids), "n_pos": int(np.nansum(y)), "clusters": cl_note, "threshold": THRESHOLD, "delta": DELTA,
           "arms": res, "injection_detection_heldout": inj, "offline_cascade": offline and offline.get("decisions")}  # fmt: skip
    s1_by = {n: arm["s1"] for n, arm in arms.items()}
    if "all" in arms:
        out["h5"] = h5(res, arms["all"]["s1"], offline)
    if a.part == "fitcal":
        out["selection"] = select(res, s1_by)
    else:
        if a.selection is None:
            raise SystemExit("--part eval needs --selection (the file written by the fitcal report): eval is read once, with the rule fixed")
        out["selection_applied"] = json.loads(a.selection.read_text())["selection"]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.json").write_text(json.dumps(out, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    if a.part == "fitcal":
        (a.out / "selection.json").write_text(json.dumps({"selection": out["selection"]}, indent=2))
    (a.out / "report.md").write_text(render(out))
    print(render(out))
    return 0


def fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def render(o: dict) -> str:
    L = [f"# System 1 report ({o['part']}; n={o['n_alerts']}, positives {o['n_pos']}; clusters: {o['clusters']})", "",
         "## Detection metrics per arm (threshold 0.3227)", "",
         "| arm | Brier | AUC | PR-AUC | log loss | ECE | cal slope | precision | recall | F1 | escalation |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in o["arms"].items():
        m = r["metrics"]
        L.append(f"| {n} | " + " | ".join(fmt(m[k]) for k in ("brier", "auc", "pr_auc", "log_loss", "ece", "cal_slope", "precision", "recall", "f1", "escalation_rate")) + " |")
    L += ["", "## Paired against OFF (95% cluster-bootstrap CIs)", "",
          "| arm | d Brier [CI] | p non-inf (Holm below) | d AUC [CI] | d log loss [CI] | flips r->w / w->r | calls/alert d | tokens_in d | questions d | eps d | median s | p90 s | speed-up median / mean [CI] |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ci = lambda d: f"{fmt(d['diff'])} [{fmt(d['ci'][0])}, {fmt(d['ci'][1])}]"  # noqa: E731
    for n, r in o["arms"].items():
        if "paired" not in r:
            continue
        p = r["paired"]
        g = lambda k: ci(p[k]) if k in p else "n/a"  # noqa: E731
        L.append(f"| {n} | {ci(p['brier_diff'])} | {fmt(p['brier_diff']['p_noninferior'])} | {g('auc_diff')} | {g('log_loss_diff')} | "
                 f"{p['flips_vs_off']['right_to_wrong']} / {p['flips_vs_off']['wrong_to_right']} | {g('llm_calls_per_alert_diff')} | {g('tokens_in_per_alert_diff')} | "
                 f"{g('questions_per_alert_diff')} | {g('eps_spent_per_alert_diff')} | {fmt(p['seconds']['median'], 2)} (OFF {fmt(p['seconds']['median_off'], 2)}) | "
                 f"{fmt(p['seconds']['p90'], 2)} (OFF {fmt(p['seconds']['p90_off'], 2)}) | {fmt(p['speedup_median_seconds'], 2)} / {fmt(p['speedup_mean_seconds']['ratio'], 2)} "
                 f"[{fmt(p['speedup_mean_seconds']['ci'][0], 2)}, {fmt(p['speedup_mean_seconds']['ci'][1], 2)}] |")
    L += ["", "## Tier shares per decision (live)", "", "| arm | decision | n | fast | large | default | median s fast / large / default |", "|---|---|---|---|---|---|---|"]
    for n, r in o["arms"].items():
        for d, v in (r["tier_shares"] or {}).items():
            L.append(f"| {n} | {d} | {v['n']} | {fmt(v['share_fast'], 3)} | {fmt(v['share_large'], 3)} | {fmt(v['share_default'], 3)} | "
                     f"{fmt(v['median_s_fast'], 3)} / {fmt(v['median_s_large'], 3)} / {fmt(v['median_s_default'], 3)} |")
    if o.get("offline_cascade"):
        L += ["", "## Cascade on the decision-set eval rows (realised error vs the 2 alpha bound)", "",
              "| decision | n | kept fast | to large | to default | error_marginal | bound | within | fast retired |", "|---|---|---|---|---|---|---|---|---|"]
        for d, v in o["offline_cascade"].items():
            L.append(f"| {d} | {v.get('n')} | {fmt(v.get('kept_fast'), 3)} | {fmt(v.get('to_large'), 3)} | {fmt(v.get('to_default'), 3)} | "
                     f"{fmt(v.get('error_marginal'))} | {fmt(v.get('bound_2alpha'), 2)} | {v.get('within_bound')} | {v.get('fast_retired')} |")
    if o.get("injection_detection_heldout"):
        L += ["", "## Injection detection on the held-out val families (with the screen; without it recall is 0 by construction)", "",
              "| family | n | attacks | recall [Wilson] | precision | false quarantine [Wilson] | settled share |", "|---|---|---|---|---|---|---|"]
        for f, v in o["injection_detection_heldout"]["with_screen"].items():
            rc = f"{fmt(v['recall'], 3)} [{fmt(v['recall_ci'][0], 3)}, {fmt(v['recall_ci'][1], 3)}]" if v["recall_ci"] else "n/a"
            fq = f"{fmt(v['false_quarantine_rate'], 3)} [{fmt(v['false_quarantine_ci'][0], 3)}, {fmt(v['false_quarantine_ci'][1], 3)}]" if v["false_quarantine_ci"] else "n/a"
            L.append(f"| {f} | {v['n']} | {v['attacks']} | {rc} | {fmt(v['precision'], 3)} | {fq} | {fmt(v['settled_share'], 3)} |")
    if o.get("h5"):
        L += ["", "## H5, per clause (exploratory, uncorrected)", "", "| clause | decision | estimate | limit | met |", "|---|---|---|---|---|"]
        for c in o["h5"]:
            L.append(f"| {c['clause']} | {c.get('decision', '')} | {fmt(c['estimate']) if not isinstance(c['estimate'], dict) else json.dumps(c['estimate'])} | {fmt(c.get('limit'))} | {c['met']} |")
    sel = o.get("selection") or o.get("selection_applied")
    if sel:
        L += ["", f"## Selection ({'derived on fit+cal' if o.get('selection') else 'frozen from fit+cal, applied here'})", "",
              f"functions on: {', '.join(sel['functions']) or 'none'}; tiers: {sel['tiers']}; per function: {sel['per_function']}"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--part", choices=["fitcal", "eval"], required=True)
    r.add_argument("--off", required=True, help="NAME=dir[,dir...] of the System-1-off arm")
    r.add_argument("--arm", action="append", default=[], help="NAME=dir[,dir...]; names worth|inject|typology|all_fast|all select by the rule")
    r.add_argument("--offline", type=Path, default=None, help="score_fast_baseline.py report output (its .eval_rows.jsonl sits beside it)")
    r.add_argument("--clusters", type=Path, default=ROOT / "outputs" / "e18" / "clusters.json")
    r.add_argument("--selection", type=Path, default=None)
    r.add_argument("--n-boot", type=int, default=bs.N_BOOT)
    r.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    return cmd_report(a)


if __name__ == "__main__":
    sys.exit(main())
