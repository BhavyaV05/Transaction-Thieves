"""Arm registry and report of the GPU RAG matrix (scripts/t3_rag_bench.sh, docs/t3_rag_bench.md). CPU only.

    python scripts/t3_rag_bench_report.py arms [--tsv]                 # the arm list (the shell script reads --tsv), blocked arms marked
    python scripts/t3_rag_bench_report.py plan                         # markdown arm table with GPU-hour estimates
    python scripts/t3_rag_bench_report.py report [--root outputs/rag_bench] [--n-boot 4000]   # fit+cal metrics, paired CIs, Holm, selection
    python scripts/t3_rag_bench_report.py report --eval --root outputs/rag_bench_eval         # the once-only eval run, descriptive

The selection rule is fixed in docs/t3_rag_bench.md (written before any GPU run) and implemented in `select_component`. The data
that CHOOSE are val `fit` + `cal` (300 askable alerts). The `--eval` report scores val `eval` for the frozen winners only and never
selects. Labels are joined from the run logs (per_alert.jsonl, val alerts only). Test is never read.

Every arm differs from its group's default in ONE factor (a flag of scripts/run_llm_agent.py on top of `--profile tuned`).
Owner: T3 (Akshat).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ftm.eval import metrics as mt  # noqa: E402
from ftm.llm.profile import decision_threshold  # noqa: E402
from ftm.planners import e18  # noqa: E402

RUN_LLM_AGENT = ROOT / "scripts" / "run_llm_agent.py"
CLUSTERS = ROOT / "outputs" / "e18" / "clusters.json"
DECISION = ROOT / "configs" / "decision" / "headroom_v1.yaml"
N_ALERTS = 300  # val fit 191 + cal 109 askable alerts
CONCURRENCY_SPEEDUP = 2.5  # wall-clock speed-up of --concurrency 4 over one-by-one (conservative; to be replaced by the first measured run)
NON_INFERIORITY_MARGIN = 0.005  # Brier; fixed before any run (docs/t3_rag_bench.md)
MIN_TOKEN_SAVING = 0.10  # an arm must cut mean prompt tokens by this share to win on cost when it is not better
MIN_VERDICT_RATE = 0.95
MIN_VAR_RATIO = 0.5  # var(p_arm) / var(p0) below this = shrinking towards a constant
AUC_DROP = 0.01
# Libraries.  The part decides the file, exactly as in scripts/t3_rerun.sh: val fit and cal use the TEACHER-ONLY libraries (built from the 1,983 train alerts' teacher runs;
# no val label can be in them), val eval uses the full ones (train teacher + the fit/cal seeding runs).  The old outputs/t3_library/frozen/*.json were built from the train teacher runs
# PLUS the val tuning runs, so a fit/cal arm that used them read fit/cal labels (fixed 7 Nov, before any GPU run of this bench).
V3 = "outputs/t3_library/frozen_v3"
BT_LIB = "outputs/rag_bench/libs/shape_backtest_train.json"  # scripts/rag_offline_bench.py --write-library: no LLM, TRAIN back-test rows only (build_backtest_library(contexts["train"]))
LIB_FILES = {"shape": {"fitcal": f"{V3}/shape_gemma_train.json", "eval": f"{V3}/shape_gemma.json"},
             "account": {"fitcal": f"{V3}/account_gemma_train.json", "eval": f"{V3}/account_gemma.json"},
             "shape_bt": {"fitcal": BT_LIB, "eval": BT_LIB}}  # fmt: skip


def lib_path(kind: str, part: str) -> str | None:
    """The library file of an arm's library kind for a val part ("fitcal" or "eval"; fit and cal share the teacher-only file)."""
    if kind == "none":
        return None
    if part not in ("fitcal", "eval"):
        raise ValueError(f"part must be fitcal or eval, got {part!r}")
    return LIB_FILES[kind][part]


# s per alert, one by one, from docs/t3_val_runs.md (old stack, val eval): the planning figure behind the GPU-hour estimates
SERIAL_S = {"planner": 6.3, "sharma_native": 4.2, "sharma_ported": 6.2, "torres_native": 13.9, "torres_ported": 17.6,
            "co_investigator_native": 21.6, "co_investigator_ported": 23.3, "finfre_rag_native": 1.5, "finfre_rag_ported": 4.0}  # fmt: skip
PROMPT_FACTOR = {"ev_none": 1.01, "rules_whole": 1.01, "rules_none": 0.99}  # prompt tokens vs the default, 3 val fit alerts, mock LLM, tuned profile (docs/t3_rag_bench.md 4a): the 6000-token cap binds in every arm, so the prompts are within 1% of each other; planning only

# capability = a regex over scripts/run_llm_agent.py (its argparse help); an arm whose flag is missing is BLOCKED, not silently dropped
CAPS = {"menu_top_k": r"--menu-top-k\b", "evidence_min": r"--evidence-min\b", "evidence_none": r"--evidence-view[\s\S]{0,700}?\bnone\b", "rules": r"--rules\b", "no_exemplars": r"--no-exemplars\b"}
GROUP_C = ("sharma", "torres", "co_investigator")


@dataclass(frozen=True)
class ArmSpec:
    name: str
    group: str  # the comparison family: "planner", or the system for Group C
    component: str  # evidence | rules | library | memory | finfre_pool | combined
    system: str
    flags: tuple[str, ...] = ()
    library: str = "none"  # none | shape | account | shape_bt (planner only); the file depends on the part: lib_path()
    seed: int = 0
    is_default: bool = False
    requires: str | None = None
    prompt_factor: float = 1.0
    note: str = ""

    def est_hours(self, n: int = N_ALERTS) -> float:
        return SERIAL_S[self.system] * self.prompt_factor * n / CONCURRENCY_SPEEDUP / 3600


def build_arms() -> list[ArmSpec]:
    P = "planner"
    arms = [ArmSpec("planner_default", P, "default", P, library="shape", is_default=True,
                    note="production: --profile tuned (p0 v2, scorer_pcst + edge scorer v2, headroom_v1 ConditionalLR, alert-aware rules) + the frozen shape library")]
    arms.append(ArmSpec("planner_default_seed1", P, "seed", P, library="shape", seed=1, note="harness DP-noise seed 1: the variance of the default itself (temperature is 0 everywhere)"))
    for name, flags, req, note in (("ev_none", ("--evidence-view", "none"), "evidence_none", "WITHOUT: alert summary only, no trail"),
                                   ("ev_amount_1hop", ("--evidence-view", "amount_1hop"), None, "top-K 1-hop edges by amount"),
                                   ("ev_amount", ("--evidence-view", "amount"), None, "top-40 by amount (the old stub's selection)"),
                                   ("ev_scorer_v2", ("--evidence-view", "scorer"), None, "top-40 by edge score (v2), not connected"),
                                   ("ev_pcst_v1", ("--edge-scorer", "v1"), None, "default view with the v1 edge scorer")):
        arms.append(ArmSpec(name, P, "evidence", P, flags, "shape", requires=req, prompt_factor=PROMPT_FACTOR.get(name, 1.0), note=note))
    for name, mode, note in (("rules_none", "none", "WITHOUT: no rules in the prompt"), ("rules_fixed", "fixed", "retrieved with the one fixed sentence (audit finding 15)"),
                             ("rules_whole", "whole", "the whole corpus in the prompt")):
        arms.append(ArmSpec(name, P, "rules", P, ("--rules", mode), "shape", requires="rules", prompt_factor=PROMPT_FACTOR.get(name, 1.0), note=note))
    # prompt budget (docs/t3_prompt_budget.md): the tuned default is K=8, N=40 at cap 6,750; menu_topk_8 repeats the default's flags (a determinism check)
    for k in ("8", "16", "all"):
        arms.append(ArmSpec(f"menu_topk_{k}", P, "menu", P, ("--menu-top-k", k), "shape", requires="menu_top_k",
                            note="top-%s questions by expected Brier reduction%s" % (k, " (= the profile default)" if k == "8" else "")))
    for n in ("0", "10", "25"):
        arms.append(ArmSpec(f"evidence_min_{n}", P, "evidence_floor", P, ("--evidence-min", n), "shape", requires="evidence_min",
                            note="newest %s trail rows protected from the trimmer%s" % (n, " (WITHOUT: legacy order, the oldest rows go first)" if n == "0" else "")))
    arms.append(ArmSpec("lib_none", P, "library", P, library="none", note="WITHOUT: no strategy library"))
    arms.append(ArmSpec("lib_account", P, "library", P, library="account", note="account-keyed (E11 memorisation ablation) from the teacher-only account library"))
    arms.append(ArmSpec("lib_shape_bt", P, "library", P, library="shape_bt", note="shape library built from TRAIN back-test rows without any LLM run (offline bench modification)"))
    for s in GROUP_C:
        for mode in ("native", "ported"):
            sysname = f"{s}_{mode}"
            arms.append(ArmSpec(f"{sysname}_mem_on", sysname, "memory", sysname, is_default=True, note="case memory, 500 train alerts, pos frac 0.5"))
            arms.append(ArmSpec(f"{sysname}_mem_off", sysname, "memory", sysname, ("--no-memory",), note="WITHOUT: empty case memory"))
    for mode in ("native", "ported"):
        sysname = f"finfre_rag_{mode}"
        arms.append(ArmSpec(f"{sysname}_pool_bal", sysname, "finfre_pool", sysname, is_default=True, note="pool 500, positive fraction 0.5 (current default)"))
        arms.append(ArmSpec(f"{sysname}_pool_nat", sysname, "finfre_pool", sysname, ("--pool-pos-frac", "-1"), note="natural-rate pool (~13% positives)"))
        arms.append(ArmSpec(f"{sysname}_pool_033", sysname, "finfre_pool", sysname, ("--pool-pos-frac", "0.33"), note="pool with one third positives"))
        arms.append(ArmSpec(f"{sysname}_zero_shot", sysname, "finfre_pool", sysname, ("--no-exemplars",), requires="no_exemplars", note="WITHOUT: zero-shot, no exemplars"))
    return arms


def blocked(arm: ArmSpec, src: str | None = None) -> str | None:
    path = lib_path(arm.library, "fitcal")
    if path and not (ROOT / path).exists():
        return f"library file missing ({path}); build it first"
    if arm.requires is None:
        return None
    src = src if src is not None else RUN_LLM_AGENT.read_text()
    return None if re.search(CAPS[arm.requires], src) else f"missing flag ({arm.requires}) in scripts/run_llm_agent.py"


def combined_arm(winner_flags: tuple[str, ...], library: str) -> ArmSpec:
    return ArmSpec("planner_combined", "planner", "combined", "planner", winner_flags, library, note="all chosen winners combined (interaction check)")


# -- loading ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Loaded:
    name: str
    ids: tuple
    y: np.ndarray
    p: np.ndarray
    p0: np.ndarray
    extras: dict = field(default_factory=dict)


def expected_settings(arm: ArmSpec) -> dict:
    """What summary.json must record for this arm's flags (the arm is verified, not trusted by name): evidence view, edge scorer, rules mode, exemplars."""
    from ftm.llm.profile import load_profile

    prof = load_profile("tuned")
    f = list(arm.flags)

    def flag(name: str, default):
        return f[f.index(name) + 1] if name in f else default

    out = {"evidence_view": flag("--evidence-view", prof["evidence_view"]), "edge_scorer": flag("--edge-scorer", prof["edge_scorer"]),
           "rules_mode": flag("--rules", "alert"),
           "exemplars": ("none" if "--no-exemplars" in f else "retrieved") if arm.system.startswith("finfre_rag") else None}
    if "--menu-top-k" in f:  # the prompt-budget arms are verified on what the run recorded (run_llm_agent summary: menu_top_k, evidence_min)
        out["menu_top_k"] = flag("--menu-top-k", str(prof["menu_top_k"]))
    if "--evidence-min" in f:
        out["evidence_min"] = int(flag("--evidence-min", prof["min_evidence_rows"]))
    return out


def settings_mismatch(arm: ArmSpec, loaded: Loaded) -> list[str]:
    """The settings the finished chunks recorded against what the arm's flags demand; [] = verified."""
    seen = loaded.extras.get("settings", {})
    return [f"{k}: expected {v!r}, chunks recorded {sorted(map(str, seen.get(k, [])))}" for k, v in expected_settings(arm).items()
            if set(seen.get(k, [])) != {v}]  # fmt: skip


def load_arm(root: Path, name: str, parts=("fit", "cal")) -> Loaded | None:
    """Pool the finished chunks of fit and cal (or eval): labels from per_alert.jsonl (val alerts only), no verdict = p0."""
    rows, steps, n_chunks = [], [], 0
    settings: dict[str, set] = {k: set() for k in ("evidence_view", "edge_scorer", "rules_mode", "exemplars", "menu_top_k", "evidence_min")}
    import polars as pl

    for part in parts:
        for c in sorted((root / name / part).glob("c[0-9][0-9][0-9]")):
            if (c / "summary.json").exists():
                rows += e18.read_per_alert(c)
                summ = json.loads((c / "summary.json").read_text(encoding="utf-8"))
                for k in settings:
                    settings[k].add(summ.get(k))
                n_chunks += 1
                steps.append(pl.read_parquet(c / "runlog.parquet").filter(pl.col("event") == "step"))  # time-guard: not transactions
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r["alert_id"])
    p0 = np.array([r["p0"] for r in rows], "f8")
    p = np.array([r["p0"] if r["probability"] is None else r["probability"] for r in rows], "f8")
    n = len(rows)
    st = pl.concat(steps) if steps else None
    nq = st.height if st is not None else 0
    used = int(st["log_lr"].is_not_null().sum()) if st is not None and st.height else 0
    secs = np.array([r["seconds"] for r in rows])
    ex = {"n": n, "no_verdict_rate": float(np.mean([r["probability"] is None for r in rows])),
          "questions_per_alert": nq / n, "accepted_question_share": used / nq if nq else float("nan"),
          "eps_per_alert": float(np.mean([r["eps_spent"] or 0.0 for r in rows])), "llm_calls": float(np.mean([r["llm_calls"] for r in rows])),
          "tokens_in": float(np.mean([r["tokens_in"] for r in rows])), "tokens_out": float(np.mean([r["tokens_out"] for r in rows])),
          "prompt_tokens_per_call": float(sum(r["tokens_in"] for r in rows) / max(sum(r["llm_calls"] for r in rows), 1)),
          "median_s": float(np.median(secs)), "p90_s": float(np.quantile(secs, 0.9)),
          "gpu_hours": float(sum(r["gpu_seconds"] for r in rows) / 3600), "chunks": n_chunks,
          "settings": {k: sorted(v, key=str) for k, v in settings.items()}}  # fmt: skip
    return Loaded(name, tuple(r["alert_id"] for r in rows), np.array([r["label"] for r in rows], "f8"), p, p0, ex)


# -- metrics ----------------------------------------------------------------------------------------------------------------------
def collapse_checks(arm: Loaded, ref: Loaded, ref_auc: float | None = None) -> dict:
    """Three checks against shrinking to a constant: variance of the posterior vs p0's, AUC not below p0's, and no Brier gain bought with an
    AUC loss against the group's default (a gain that comes from the base-rate shrink leaves AUC unchanged or lower)."""
    from rag_offline_bench import auc

    from ftm.eval.metrics import brier

    a_auc, p0_auc = auc(arm.y, arm.p), auc(arm.y, arm.p0)
    var_ratio = float(np.var(arm.p) / max(np.var(arm.p0), 1e-12))
    ref_auc = ref_auc if ref_auc is not None else auc(ref.y, ref.p)
    out = {"var_ratio_vs_p0": var_ratio, "auc": a_auc, "auc_p0": p0_auc, "auc_default": ref_auc,
           "auc_below_p0": bool(a_auc < p0_auc), "shrunk": bool(var_ratio < MIN_VAR_RATIO),
           "gain_from_shrink": bool(brier(arm.y, arm.p) < brier(ref.y, ref.p) and a_auc < ref_auc - AUC_DROP)}  # fmt: skip
    out["collapsed"] = bool(out["shrunk"] or out["gain_from_shrink"] or out["auc_below_p0"])
    return out


def group_report(arms: dict[str, Loaded], default: str, clusters: np.ndarray, thr: float, n_boot: int) -> dict:
    """Full metric set per arm, paired differences vs the default and vs p0 with cluster-bootstrap CIs, Holm within the group's arms."""
    import t3_e18_refresh_report as e18r

    from ftm.eval.stats import holm

    ref = arms[default]
    preds = {n: a.p for n, a in arms.items()} | {"p0": ref.p0}
    point = {n: e18r.metric_set(ref.y, p, thr) for n, p in preds.items()}
    boot = e18r.boot_metrics(ref.y, preds, clusters, thr, n_boot)
    out = {"threshold": thr, "n": len(ref.y), "n_boot": n_boot, "metrics": point, "vs_default": {}, "vs_p0": {}, "flips_vs_p0": {}, "resources": {},
           "collapse": {}}  # fmt: skip
    for n, a in arms.items():
        out["vs_p0"][n] = e18r.paired(point, boot, n, "p0")
        if n != default:
            out["vs_default"][n] = e18r.paired(point, boot, n, default)
        out["flips_vs_p0"][n] = mt.flips(ref.y, ref.p0, a.p, thr)
        out["resources"][n] = a.extras
        out["collapse"][n] = collapse_checks(a, ref)
    for fam in ("vs_default", "vs_p0"):
        src = out[fam]
        for m in e18r.METRICS:
            ps = {n: src[n][m]["p"] for n in src if src[n][m]["p"] is not None}
            for n, h in (holm(ps).items() if ps else ()):
                src[n][m]["p_holm"], src[n][m]["holm_reject"] = h["p_adj"], bool(h["reject"])
    return out


def select_component(arm_names: list[str], default: str, rep: dict, components: dict[str, str]) -> dict:
    """THE SELECTION RULE (docs/t3_rag_bench.md, fixed before any GPU run), per component, on val fit+cal:
    eligible = verdict rate >= 0.95, not collapsed. Tier 1: an eligible arm whose paired cluster-bootstrap 95% CI of
    Brier(arm) - Brier(default) lies below 0: the one with the lowest Brier. Tier 2 (no tier-1 arm): an eligible arm with CI upper bound
    < +0.005 (non-inferior) that cuts mean prompt tokens by >= 10%: the cheapest. Else the default stays."""
    cands = []
    for n in arm_names:
        if n == default:
            continue
        d = rep["vs_default"][n]["brier"]
        res = rep["resources"][n]
        ok = res["no_verdict_rate"] <= 1 - MIN_VERDICT_RATE and not rep["collapse"][n]["collapsed"]
        saving = 1 - res["tokens_in"] / max(rep["resources"][default]["tokens_in"], 1)
        cands.append({"arm": n, "diff": d["diff"], "ci": d["ci"], "eligible": ok, "token_saving": saving,
                      "tier": 1 if ok and d["ci"][1] < 0 else 2 if ok and d["ci"][1] < NON_INFERIORITY_MARGIN and saving >= MIN_TOKEN_SAVING else 0})  # fmt: skip
    t1 = sorted((c for c in cands if c["tier"] == 1), key=lambda c: c["diff"])
    t2 = sorted((c for c in cands if c["tier"] == 2), key=lambda c: -c["token_saving"])
    win = (t1 or t2 or [None])[0]
    return {"default": default, "winner": win["arm"] if win else default, "reason": ("tier 1: CI excludes 0, better" if t1 else "tier 2: non-inferior and cheaper" if t2 else "default kept"),
            "candidates": cands}  # fmt: skip


# -- report -----------------------------------------------------------------------------------------------------------------------
def fmt(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and x != x) else f"{x:.{nd}f}"


def group_table(rep: dict, default: str) -> str:
    cols = ["brier", "log_loss", "auc", "pr_auc", "ece", "cal_slope", "precision", "recall", "f1", "escalation_rate", "catch_top10"]
    t = "| arm | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n"
    for n, m in rep["metrics"].items():
        t += f"| {n}{' (default)' if n == default else ''} | " + " | ".join(fmt(m[c]) for c in cols) + " |\n"
    for fam, title in (("vs_default", f"the default ({default})"), ("vs_p0", "p0")):
        if not rep[fam]:
            continue
        t += f"\nPaired difference (arm - {title}), 95% cluster-bootstrap CI, Holm-adjusted p within this group:\n\n| arm | brier | log_loss | auc | pr_auc | ece | f1 |\n|---|---|---|---|---|---|---|\n"
        for n, d in rep[fam].items():
            cells = [("-" if d[c]["diff"] is None else f"{d[c]['diff']:+.4f} [{d[c]['ci'][0]:+.4f}, {d[c]['ci'][1]:+.4f}] p_holm {fmt(d[c].get('p_holm'))}")
                     for c in ("brier", "log_loss", "auc", "pr_auc", "ece", "f1")]  # fmt: skip
            t += f"| {n} | " + " | ".join(cells) + " |\n"
    t += "\nFlipped calls vs p0 at the threshold (right-to-wrong / wrong-to-right), no-verdict rate, collapse checks:\n\n| arm | flips | no verdict | var(p)/var(p0) | AUC | AUC p0 | collapsed |\n|---|---|---|---|---|---|---|\n"
    for n in rep["metrics"]:
        if n == "p0":
            continue
        f, c, r = rep["flips_vs_p0"][n], rep["collapse"][n], rep["resources"][n]
        t += f"| {n} | {f['right_to_wrong']} / {f['wrong_to_right']} | {fmt(r['no_verdict_rate'])} | {fmt(c['var_ratio_vs_p0'])} | {fmt(c['auc'])} | {fmt(c['auc_p0'])} | {c['collapsed']} |\n"
    t += "\nPer alert (means): questions, accepted-question share, eps, LLM calls, prompt tokens in / per call, tokens out, median / p90 s, GPU h.\n\n| arm | questions | accepted | eps | calls | tok in | tok/call | tok out | median s | p90 s | GPU h |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
    for n, r in rep["resources"].items():
        t += (f"| {n} | {r['questions_per_alert']:.2f} | {fmt(r['accepted_question_share'], 2)} | {r['eps_per_alert']:.2f} | {r['llm_calls']:.2f} | {r['tokens_in']:.0f} | "
              f"{r['prompt_tokens_per_call']:.0f} | {r['tokens_out']:.0f} | {r['median_s']:.1f} | {r['p90_s']:.1f} | {r['gpu_hours']:.2f} |\n")
    return t


def build_report(root: Path, n_boot: int, parts=("fit", "cal"), select: bool = True, clusters_path: Path = CLUSTERS) -> dict:
    arms = build_arms()
    by_group: dict[str, dict[str, Loaded]] = {}
    comps = {a.name: a.component for a in arms}
    comb = root / "planner_combined"
    names = [a.name for a in arms] + (["planner_combined"] if comb.exists() else [])
    comps["planner_combined"] = "combined"
    group_of = {a.name: a.group for a in arms} | {"planner_combined": "planner"}
    spec_of, out_bad = {a.name: a for a in arms}, {}
    defaults = {a.group: a.name for a in arms if a.is_default}
    for n in names:
        la = load_arm(root, n, parts)
        if la is not None and n in spec_of and (bad := settings_mismatch(spec_of[n], la)):
            out_bad[n] = bad  # an arm whose recorded settings differ from its flags is never compared
            continue
        if la is not None:
            by_group.setdefault(group_of[n], {})[n] = la
    mapping = e18.load_clusters(clusters_path) if clusters_path.exists() else None
    thr = decision_threshold(str(DECISION))
    out: dict = {"parts": list(parts), "root": str(root), "n_boot": n_boot, "groups": {}, "selection": {}, "incomplete": {},
           "settings_mismatch": out_bad}  # fmt: skip
    for g, arms_g in by_group.items():
        d = defaults[g]
        if d not in arms_g:
            out["incomplete"][g] = f"default arm {d} has no finished chunk"
            continue
        ref = arms_g[d]
        done = {n: a for n, a in arms_g.items() if a.ids == ref.ids}
        out["incomplete"].update({n: len(a.ids) for n, a in arms_g.items() if a.ids != ref.ids})
        cl = e18.clusters_for(ref.ids, mapping)[0]
        rep = group_report(done, d, cl, thr, n_boot)
        out["groups"][g] = rep
        if select:
            for comp in sorted({comps[n] for n in done if comps[n] not in ("default", "seed", "combined")}):
                members = [n for n in done if comps[n] == comp or n == d]
                if len(members) > 1:
                    out["selection"][f"{g}/{comp}"] = select_component(members, d, rep, comps)
    return out


def winner_flags(selection: dict, arms: list[ArmSpec] | None = None) -> tuple[tuple[str, ...], str]:
    """(flags, library) of the planner arm that combines the winning planner components (the default's flags where it kept the default)."""
    arms = {a.name: a for a in (arms or build_arms())}
    flags: list[str] = []
    lib = "shape"
    for comp in ("evidence", "rules", "library", "menu", "evidence_floor"):
        s = selection.get(f"planner/{comp}")
        if s and s["winner"] != s["default"]:
            a = arms[s["winner"]]
            flags += list(a.flags)
            lib = a.library if comp == "library" else lib
    return tuple(flags), lib


def markdown(rep: dict) -> str:
    L = [f"# GPU RAG matrix report ({'+'.join(rep['parts'])}, {rep['n_boot']} bootstrap resamples)", ""]
    if rep["parts"] == ["eval"]:
        L += ["**val `eval`, scored once for the frozen winners: descriptive, nothing is selected from it.**", ""]
    for g, r in rep["groups"].items():
        L += [f"## Group {g} (n={r['n']}, threshold {r['threshold']})", "", group_table(r, _default_of(g)), ""]
    if rep["selection"]:
        L += ["## Selection (rule of docs/t3_rag_bench.md)", "", "| component | default | winner | reason |", "|---|---|---|---|"]
        for k, s in rep["selection"].items():
            L.append(f"| {k} | {s['default']} | **{s['winner']}** | {s['reason']} |")
        L.append("")
    if rep.get("settings_mismatch"):
        L += ["Arms excluded because summary.json does not record the settings their flags demand:", "", "```json",
              json.dumps(rep["settings_mismatch"], indent=1), "```", ""]
    if rep["incomplete"]:
        L += ["Incomplete / not comparable:", "", "```json", json.dumps(rep["incomplete"], indent=1), "```"]
    return "\n".join(L)


def _default_of(group: str) -> str:
    return next(a.name for a in build_arms() if a.group == group and a.is_default)


def plan_table(src: str | None = None) -> str:
    arms = build_arms()
    t = "| arm | group | component | one-factor change vs the group default | est. GPU h (300 alerts) | status |\n|---|---|---|---|---|---|\n"
    tot = every = 0.0
    for a in arms:
        b = blocked(a, src)
        h = a.est_hours()
        tot += 0 if b else h
        every += h
        ch = " ".join(a.flags) or "-"
        if a.library != "shape" and a.group == "planner":
            ch = f"library={a.library} " + ch
        if a.seed:
            ch += f" seed={a.seed}"
        t += f"| {a.name}{' (default)' if a.is_default else ''} | {a.group} | {a.component} | {ch} | {h:.2f} | {b or 'ready'} |\n"
    return t + (f"\nTotal: {tot:.1f} GPU-h for the arms runnable today, {every:.1f} GPU-h for all {len(arms)} arms once their library files exist "
                f"(fit+cal, 300 alerts each; planning figure: old-stack seconds per alert / {CONCURRENCY_SPEEDUP} for concurrency 4). Not included: the combined arm with two seeds, the once-only eval.\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("arms")
    s.add_argument("--tsv", action="store_true")
    sub.add_parser("plan")
    r = sub.add_parser("report")
    r.add_argument("--root", type=Path, default=ROOT / "outputs" / "rag_bench")
    r.add_argument("--eval", action="store_true", help="score the eval part (frozen winners only; descriptive)")
    r.add_argument("--n-boot", type=int, default=4000)
    r.add_argument("--clusters", type=Path, default=CLUSTERS)
    w = sub.add_parser("winner-flags")
    w.add_argument("--selection", type=Path, default=ROOT / "outputs" / "rag_bench" / "selection.json")
    a = ap.parse_args(argv)
    if a.cmd == "arms":
        src = RUN_LLM_AGENT.read_text()
        for arm in build_arms():
            b = blocked(arm, src)
            if a.tsv:
                print("\t".join([arm.name, arm.system, str(arm.seed), lib_path(arm.library, "fitcal") or "none", lib_path(arm.library, "eval") or "none",
                                f"{arm.est_hours():.3f}", b or "ok", " ".join(arm.flags)]))
            else:
                print(arm.name, arm.group, arm.component, arm.flags, b or "ok")
        return 0
    if a.cmd == "plan":
        print(plan_table())
        return 0
    if a.cmd == "winner-flags":
        sel = json.loads(a.selection.read_text())["selection"]
        flags, lib = winner_flags(sel)
        print(lib, " ".join(flags))
        return 0
    rep = build_report(a.root, a.n_boot, parts=("eval",) if a.eval else ("fit", "cal"), select=not a.eval, clusters_path=a.clusters)
    a.root.mkdir(parents=True, exist_ok=True)
    (a.root / ("report_eval.json" if a.eval else "selection.json")).write_text(json.dumps(rep, indent=1, default=float))
    (a.root / ("report_eval.md" if a.eval else "report.md")).write_text(markdown(rep))
    print(markdown(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
