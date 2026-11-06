"""E18 backbone refresh (2 Nov): profile lookup, non-LLM EIG reference, and the selection report.  CPU only.  Owner: T3 (Akshat).

    python scripts/t3_e18_refresh_report.py profile gemma4-12b                  # repo <TAB> revision <TAB> hub cache dir
    python scripts/t3_e18_refresh_report.py eig-ref --out outputs/e18_refresh/eig_ref.json   # label-free EIG policy on val fit+cal
    python scripts/t3_e18_refresh_report.py report --root outputs/e18_refresh [--best]       # selection rule -> report.json / report.md

Runs live in <root>/<candidate>/{fit,cal}/c###  (scripts/run_llm_agent.py run dirs).  Validation `fit` + `cal` only: the `eval` part is never
read here (the EIG reference drops it on load, as `scripts/t3_headroom.py select` does).  The selection rule is FIXED in
docs/t3_e18.md ("E18 refresh (2 Nov)") before any run and implemented by `select` below.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ftm.eval import bootstrap as bs  # noqa: E402
from ftm.llm.profile import decision_threshold  # noqa: E402
from ftm.planners import e18  # noqa: E402

SERVING = ROOT / "configs" / "llm" / "serving.yaml"
CLUSTERS = ROOT / "outputs" / "e18" / "clusters.json"
MIN_VALID = 0.95  # the prereg stage-1 filter on the valid-DSL output rate, kept as an eligibility gate
N_BOOT, LEVEL, SEED = 10_000, 0.95, 0
EIG = "eig_reference"

# parameters in billions: (total, active).  Used only to order equivalent models ("smaller wins").
SIZE_B = {
    "gemma4-12b": (12.0, 12.0), "qwen35-9b": (9.0, 9.0), "qwen35-4b": (4.0, 4.0), "gemma4-e4b": (8.0, 4.0),
    "qwen3-14b": (14.8, 14.8), "gpt-oss-20b": (21.0, 3.6), "phi4-14b": (14.7, 14.7),
}  # fmt: skip


def profile(name: str, path: Path = SERVING) -> tuple[str, str, str]:
    """(repo, revision, hub cache directory name) of a serving profile."""
    from omegaconf import OmegaConf

    models = OmegaConf.to_container(OmegaConf.load(path))["models"]
    if name not in models:
        raise SystemExit(f"unknown profile {name!r}; choose from {sorted(models)}")
    repo, rev = models[name]["repo"], models[name]["revision"]
    return repo, rev, "models--" + repo.replace("/", "--")


# -- the non-LLM EIG reference -------------------------------------------------------------------------------------
def eig_reference(out: Path) -> dict:
    """Frozen headroom_v1 'primary' EIG policy on val fit+cal, honest cross-part ((a, b) fitted on one part, scored on the other).
    The `eval` part is dropped on load.  The primary variant has base `p0` (a = 0, b = 1), so nothing is fitted on the scored part."""
    import t3_headroom as th

    from ftm.fusion.headroom import fit_train, log_lr_matrix, scorer_of, subsplit

    dec = json.loads(th.DECISIONS.read_text())
    v = dec["variants"]["primary"]
    hyps, tr, va = th.load_all()
    va = subsplit(va, np.flatnonzero(va.part != "eval"))  # eval labels are never read below this line
    assert not (va.part == "eval").any()
    m, des, lrs = fit_train(tr, hyps, l2=dec["l2"], use=None)
    xv = log_lr_matrix(va, hyps, lrs)
    sc0 = scorer_of(m, des, hyps, 1.0)
    p = th.cross_part(va, sc0, v["base"], v["shrink"], v["alloc"], xv, hyps)
    res = {"alert_ids": list(va.alert_ids), "p": [float(x) for x in p], "p0": [float(x) for x in va.p0], "y": [int(x) for x in va.y],
           "part": [str(x) for x in va.part], "variant": "headroom_v1 primary", "decisions_sha256": th.sha(th.DECISIONS),
           "brier": float(np.mean((p - va.y) ** 2)), "brier_p0": float(np.mean((va.p0 - va.y) ** 2))}  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    return res


# -- arms ----------------------------------------------------------------------------------------------------------
@dataclass
class Arm:
    name: str
    ids: tuple
    y: np.ndarray
    p: np.ndarray
    valid_dsl: float | None = None
    median_s: float | None = None
    median_gpu_s: float | None = None
    tokens_out: float | None = None
    no_verdict: float | None = None
    p0: np.ndarray | None = None
    resources: dict | None = None


def load_arm(root: Path, name: str, parts=("fit", "cal")) -> Arm | None:
    """Pool all finished chunks of the candidate's fit and cal runs; None if there is no finished chunk."""
    rows = []
    for part in parts:
        for c in sorted((root / name / part).glob("c[0-9][0-9][0-9]")):
            if (c / "summary.json").exists():
                rows += e18.read_per_alert(c)
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r["alert_id"])
    tot = {k: sum(r[k] for r in rows) for k in ("outputs", "json_ok", "questions", "dsl_ok")}
    p = np.array([r["p0"] if r["probability"] is None else r["probability"] for r in rows], "f8")  # no verdict = p0
    return Arm(name, tuple(r["alert_id"] for r in rows), np.array([r["label"] for r in rows], "f8"), p,
               e18.valid_dsl_output_rate(tot) if tot["outputs"] else None,
               float(np.median([r["seconds"] for r in rows])), float(np.median([r["gpu_seconds"] for r in rows])),
               float(np.mean([r["tokens_out"] for r in rows])), float(np.mean([r["probability"] is None for r in rows])),
               np.array([r["p0"] for r in rows], "f8"),
               {k: float(np.mean([r.get(k) or 0 for r in rows])) for k in ("questions", "eps_spent", "llm_calls", "tokens_in", "tokens_out",
                                                                          "seconds", "gpu_seconds", "duplicates", "refusals")})


def eig_arm(path: Path) -> Arm:
    d = json.loads(path.read_text())
    o = np.argsort(d["alert_ids"])
    return Arm(EIG, tuple(np.array(d["alert_ids"])[o]), np.array(d["y"], "f8")[o], np.array(d["p"], "f8")[o],
               p0=np.array(d["p0"], "f8")[o])


def diff(a: Arm, b: Arm, clusters: np.ndarray, n_boot: int) -> dict:
    """Brier(a) - Brier(b), paired cluster bootstrap 95% CI.  Negative = a better."""
    r = bs.brier_diff(a.y, a.p, b.p, clusters, n_boot=n_boot, seed=SEED, level=LEVEL)
    return {"diff": r["diff"], "ci": [float(r["ci"][0]), float(r["ci"][1])]}


# -- the selection rule (fixed 2 Nov, before any run) ---------------------------------------------------------------
def select(arms: dict[str, Arm], eig: Arm, clusters: np.ndarray, *, n_boot: int = N_BOOT, min_valid: float = MIN_VALID) -> dict:
    """1. eligible = valid-DSL output rate >= min_valid.  2. useful = eligible and the CI of Brier(model) - Brier(EIG) is not entirely above 0
    (beats: upper < 0; ties: spans 0).  3. best = lowest Brier among useful; equivalent = useful models whose CI of Brier(model) - Brier(best)
    includes 0 (lower <= 0).  4. pick the equivalent model with the fewest active parameters; ties: lower median seconds per alert, fewer
    total parameters, name.  No useful model: no LLM pick, the non-LLM EIG policy stays."""
    for a in arms.values():
        if a.ids != eig.ids or not np.array_equal(a.y, eig.y):
            raise ValueError(f"{a.name} does not cover the same alerts/labels as the EIG reference")
    rows, useful = {}, []
    for n, a in arms.items():
        d = diff(a, eig, clusters, n_boot)
        eligible = a.valid_dsl is None or a.valid_dsl >= min_valid
        verdict = "beats" if d["ci"][1] < 0 else ("worse" if d["ci"][0] > 0 else "ties")
        rows[n] = {"brier": float(np.mean((a.p - a.y) ** 2)), "vs_eig": d, "vs_eig_verdict": verdict, "eligible": eligible,
                   "useful": bool(eligible and verdict != "worse"), "valid_dsl": a.valid_dsl, "median_s": a.median_s,
                   "median_gpu_s": a.median_gpu_s, "tokens_out_per_alert": a.tokens_out, "no_verdict_rate": a.no_verdict,
                   "params_b": SIZE_B.get(n, (None, None))[0], "active_b": SIZE_B.get(n, (None, None))[1]}  # fmt: skip
        if rows[n]["useful"]:
            useful.append(n)
    out = {"n_alerts": len(eig.ids), "eig_brier": float(np.mean((eig.p - eig.y) ** 2)), "n_boot": n_boot, "models": rows,
           "useful": useful, "best": None, "equivalent": [], "pick": None}  # fmt: skip
    if not useful:
        return out
    best = min(useful, key=lambda n: (rows[n]["brier"], n))
    out["best"] = best
    for n in useful:
        d = diff(arms[n], arms[best], clusters, n_boot) if n != best else {"diff": 0.0, "ci": [0.0, 0.0]}
        rows[n]["vs_best"] = d
        if n == best or d["ci"][0] <= 0:
            out["equivalent"].append(n)
    big = float("inf")
    out["pick"] = min(out["equivalent"], key=lambda n: (rows[n]["active_b"] or big, rows[n]["median_s"] or big, rows[n]["params_b"] or big, n))
    return out


# -- the full metric set (RIGOR rule, brief.md): every metric with paired cluster-bootstrap CIs and Holm ------------------
METRICS = ("brier", "log_loss", "auc", "pr_auc", "ece", "f1", "precision", "recall", "escalation_rate", "catch_top10", "cal_slope")
LOWER_BETTER = {"brier", "log_loss", "ece"}  # for the rest, higher is better (cal_slope: closer to 1 is better, reported only)


def _auc(y, p) -> float:
    from scipy.stats import rankdata

    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _cal_slope(y, p) -> float:
    """Slope of a logistic regression of y on logit(p) (1 = calibrated); Newton, 30 steps, tiny ridge."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    if y.sum() == 0 or y.sum() == len(y) or z.std() == 0:
        return float("nan")
    X = np.column_stack([np.ones(len(y)), z])
    b = np.array([0.0, 1.0])
    for _ in range(30):
        q = 1 / (1 + np.exp(-X @ b))
        g = X.T @ (y - q) - 1e-6 * b
        H = (X * (q * (1 - q))[:, None]).T @ X + 1e-6 * np.eye(2)
        b = b + np.linalg.solve(H, g)
    return float(b[1])


def metric_set(y, p, thr: float) -> dict:
    """Brier, log loss, AUC, PR-AUC, ECE, calibration slope, precision / recall / F1 / escalation rate at `thr`, and the share of
    positives among the top 10% scored alerts (catch at a 10% review capacity; the alert table has no day for a per-day capacity)."""
    from ftm.eval.metrics import ece, pr_auc

    y, p = np.asarray(y, "f8"), np.asarray(p, "f8")
    q = np.clip(p, 1e-6, 1 - 1e-6)
    esc = p >= thr
    tp = float((esc & (y == 1)).sum())
    prec = tp / esc.sum() if esc.any() else float("nan")
    rec = tp / y.sum() if y.sum() else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec > 0 else 0.0
    k = max(1, int(np.ceil(0.1 * len(y))))
    top = np.argsort(-p, kind="stable")[:k]
    return {"brier": float(np.mean((p - y) ** 2)), "log_loss": float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))), "auc": _auc(y, p),
            "pr_auc": pr_auc(y, p), "ece": ece(y, p), "f1": f1, "precision": prec, "recall": rec, "escalation_rate": float(esc.mean()),
            "catch_top10": float(y[top].sum() / y.sum()) if y.sum() else float("nan"), "cal_slope": _cal_slope(y, p)}  # fmt: skip


def boot_metrics(y, preds: dict[str, np.ndarray], clusters, thr: float, n_boot: int) -> dict[str, dict[str, np.ndarray]]:
    """Per resample of whole clusters (the same resamples for every arm, so differences are paired): every arm's metric set."""
    cl = np.unique(np.asarray(clusters), return_inverse=True)[1]
    members = np.split(np.argsort(cl, kind="stable"), np.cumsum(np.bincount(cl))[:-1])
    rng = np.random.default_rng(SEED)
    out = {n: {m: np.empty(n_boot) for m in METRICS} for n in preds}
    for b in range(n_boot):
        ix = np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
        for n, p in preds.items():
            ms = metric_set(y[ix], p[ix], thr)
            for m in METRICS:
                out[n][m][b] = ms[m]
    return out


def paired(point: dict, boot: dict, a: str, b: str) -> dict:
    """Metric(a) - metric(b) per metric: point difference, percentile CI, two-sided bootstrap p."""
    from ftm.eval.bootstrap import bootstrap_pvalue, percentile_ci

    res = {}
    for m in METRICS:
        d = boot[a][m] - boot[b][m]
        ok = d[~np.isnan(d)]
        if not len(ok):
            res[m] = {"diff": None, "ci": [None, None], "p": None}
            continue
        lo, hi = percentile_ci(d)
        p = min(1.0, 2 * min(bootstrap_pvalue(d, 0.0, "greater"), bootstrap_pvalue(d, 0.0, "less")))
        res[m] = {"diff": float(point[a][m] - point[b][m]), "ci": [lo, hi], "p": p}
    return res


def full_report(arms: dict[str, Arm], eig: Arm, clusters, thr: float, n_boot: int, incumbent: str = "gemma4-12b") -> dict:
    """Metric sets and paired differences vs the EIG reference, vs p0 and vs the incumbent; Holm over the LLM models within each (comparison,
    metric) family.  Reported next to the fixed selection rule, which stays CI-based (`select`)."""
    from ftm.eval.stats import holm

    preds = {n: a.p for n, a in arms.items()} | {EIG: eig.p, "p0": eig.p0}
    point = {n: metric_set(eig.y, p, thr) for n, p in preds.items()}
    boot = boot_metrics(eig.y, preds, clusters, thr, n_boot)
    out = {"threshold": thr, "n_boot": n_boot, "metrics": point, "vs_eig": {}, "vs_p0": {}, "vs_incumbent": {}, "holm": {}}
    for n in arms:
        out["vs_eig"][n] = paired(point, boot, n, EIG)
        out["vs_p0"][n] = paired(point, boot, n, "p0")
        if incumbent in arms and n != incumbent:
            out["vs_incumbent"][n] = paired(point, boot, n, incumbent)
    for fam, src in (("vs_eig", out["vs_eig"]), ("vs_incumbent", out["vs_incumbent"])):
        if not src:
            continue
        for m in METRICS:
            ps = {n: src[n][m]["p"] for n in src if src[n][m]["p"] is not None}
            if ps:
                for n, h in holm(ps).items():
                    src[n][m]["p_holm"], src[n][m]["holm_reject"] = h["p_adj"], bool(h["reject"])
                out["holm"][f"{fam}:{m}"] = {"family_size": len(ps)}
    for n, a in arms.items():  # flipped calls at the threshold, versus p0 and versus the EIG reference
        from ftm.eval.metrics import flips

        out.setdefault("flips_vs_p0", {})[n] = flips(eig.y, eig.p0, a.p, thr)
        out.setdefault("flips_vs_eig", {})[n] = flips(eig.y, eig.p, a.p, thr)
        out.setdefault("resources", {})[n] = a.resources
    return out


def full_table(rep: dict) -> str:
    """Markdown: metrics per arm, then the paired differences with CIs (vs EIG, vs incumbent) and Holm-adjusted p."""
    cols = ["brier", "log_loss", "auc", "pr_auc", "ece", "cal_slope", "precision", "recall", "f1", "escalation_rate", "catch_top10"]
    f = lambda x: "-" if x is None or x != x else f"{x:.3f}"  # noqa: E731
    t = "| arm | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n"
    for n, m in rep["metrics"].items():
        t += f"| {n} | " + " | ".join(f(m[c]) for c in cols) + " |\n"
    for fam, title in (("vs_eig", "EIG reference"), ("vs_incumbent", "incumbent")):
        if not rep[fam]:
            continue
        t += f"\nPaired difference (model - {title}), 95% cluster-bootstrap CI, Holm-adjusted p within the model family:\n\n"
        t += "| model | " + " | ".join(cols[:6]) + " |\n|" + "---|" * 7 + "\n"
        for n, d in rep[fam].items():
            cells = []
            for c in cols[:6]:
                x = d[c]
                cells.append("-" if x["diff"] is None else f"{x['diff']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}] p_holm {x.get('p_holm', float('nan')):.3f}")
            t += f"| {n} | " + " | ".join(cells) + " |\n"
    t += "\nFlipped calls at the threshold (right-to-wrong / wrong-to-right) versus p0 and versus the EIG reference:\n\n| model | vs p0 | vs EIG |\n|---|---|---|\n"
    for n in rep.get("flips_vs_p0", {}):
        a, b = rep["flips_vs_p0"][n], rep["flips_vs_eig"][n]
        t += f"| {n} | {a['right_to_wrong']} / {a['wrong_to_right']} | {b['right_to_wrong']} / {b['wrong_to_right']} |\n"
    t += "\nPer alert (means): questions, eps spent, LLM calls, tokens in / out, wall seconds, GPU seconds.\n\n| model | questions | eps | calls | tok in | tok out | s | gpu s |\n|---|---|---|---|---|---|---|---|\n"
    for n, r in rep.get("resources", {}).items():
        t += f"| {n} | {r['questions']:.2f} | {r['eps_spent']:.2f} | {r['llm_calls']:.2f} | {r['tokens_in']:.0f} | {r['tokens_out']:.0f} | {r['seconds']:.1f} | {r['gpu_seconds']:.1f} |\n"
    return t


def table(res: dict) -> str:
    head = "| model | Brier | vs EIG (95% CI) | vs EIG | valid-DSL | median s | active B | useful | equivalent |\n|---|---|---|---|---|---|---|---|---|\n"
    f = lambda x, k=3: "-" if x is None else f"{x:.{k}f}"  # noqa: E731
    lines = [f"| EIG reference (non-LLM) | {res['eig_brier']:.4f} | - | - | - | - | - | - | - |"]
    for n, r in sorted(res["models"].items(), key=lambda kv: kv[1]["brier"]):
        d = r["vs_eig"]
        lines.append(f"| {n}{' (PICK)' if n == res['pick'] else ''} | {r['brier']:.4f} | {d['diff']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}] | "
                     f"{r['vs_eig_verdict']} | {f(r['valid_dsl'])} | {f(r['median_s'], 1)} | {f(r['active_b'], 1)} | "
                     f"{'yes' if r['useful'] else 'no'} | {'yes' if n in res['equivalent'] else 'no'} |")  # fmt: skip
    pick = res["pick"] or "none (no LLM backbone is useful over the non-LLM EIG policy)"
    return head + "\n".join(lines) + f"\n\nBest by Brier: {res['best']}.  Pick: {pick}.\n"


def build(root: Path, eig_path: Path, clusters_path: Path, only: list[str] | None, n_boot: int, full: bool = True) -> dict:
    arms = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        if only and d.name not in only:
            continue
        a = load_arm(root, d.name)
        if a is not None:
            arms[d.name] = a
    eig = eig_arm(eig_path)
    mapping = e18.load_clusters(clusters_path)
    done = {n: a for n, a in arms.items() if a.ids == eig.ids}
    incomplete = {n: len(a.ids) for n, a in arms.items() if a.ids != eig.ids}
    clusters = e18.clusters_for(eig.ids, mapping)[0]
    res = select(done, eig, clusters, n_boot=n_boot)
    res["incomplete"] = incomplete  # candidates with fewer/other alerts than the reference: not ranked
    if done and full:
        thr = decision_threshold("configs/decision/headroom_v1.yaml")
        res["full"] = full_report(done, eig, clusters, thr, n_boot)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("profile").add_argument("name")
    e = sub.add_parser("eig-ref")
    e.add_argument("--out", type=Path, default=ROOT / "outputs" / "e18_refresh" / "eig_ref.json")
    r = sub.add_parser("report")
    r.add_argument("--root", type=Path, default=ROOT / "outputs" / "e18_refresh")
    r.add_argument("--eig", type=Path, default=None)
    r.add_argument("--clusters", type=Path, default=CLUSTERS)
    r.add_argument("--only", nargs="*", default=None)
    r.add_argument("--n-boot", type=int, default=N_BOOT)
    r.add_argument("--best", action="store_true", help="print only the name of the current lowest-Brier useful model (empty if none)")
    a = ap.parse_args(argv)
    if a.cmd == "profile":
        print("\t".join(profile(a.name)))
    elif a.cmd == "eig-ref":
        res = eig_reference(a.out)
        print(f"EIG reference: {len(res['p'])} alerts, Brier {res['brier']:.4f} (p0 {res['brier_p0']:.4f}) -> {a.out}")
    else:
        res = build(a.root, a.eig or a.root / "eig_ref.json", a.clusters, a.only, 300 if a.best else a.n_boot, full=not a.best)
        if a.best:
            print(res["best"] or "")
            return 0
        (a.root / "report.json").write_text(json.dumps(res, indent=1))
        md = table(res) + ("\n## Full metric set\n\n" + full_table(res["full"]) if "full" in res else "")
        (a.root / "report.md").write_text(md)
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
