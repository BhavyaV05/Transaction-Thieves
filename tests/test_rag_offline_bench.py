"""Offline RAG bench helpers and the GPU matrix report (scripts/rag_offline_bench.py, scripts/t3_rag_bench_report.py, scripts/t3_rag_bench.sh)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


ob = _load("rag_offline_bench")
rb = _load("t3_rag_bench_report")


# -- offline helpers ---------------------------------------------------------------------------------------------------
def test_auc_and_logistic():
    y = np.array([0, 0, 1, 1, 0, 1])
    assert ob.auc(y, np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])) == 1.0
    assert ob.auc(y, np.zeros(6)) == 0.5
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    yy = (rng.random(2000) < 1 / (1 + np.exp(-(0.5 + 2 * x)))).astype(float)
    b = ob.fit_logistic(x[:, None], yy)
    assert abs(b[1] - 2) < 0.4 and abs(b[0] - 0.5) < 0.3
    assert ob.predict_logistic(b, x[:, None]).shape == (2000,)


def test_time_split_and_balanced_sample():
    past, later = ob.time_split(["2022-09-03", "2022-09-01", "2022-09-02", "2022-09-05", "2022-09-04"], 0.6)
    assert sorted(past.tolist()) == [0, 1, 2] and sorted(later.tolist()) == [3, 4]  # earliest three are the past
    y = np.array([1] * 20 + [0] * 180)
    idx = ob.balanced_sample(y, 20, 0.5)
    assert len(idx) == 20 and y[idx].sum() == 10
    assert len(ob.balanced_sample(y, 20, None)) == 200 and len(ob.balanced_sample(y, 20, -1)) == 200


def test_knn_scores_separate_classes_and_loo():
    rng = np.random.default_rng(1)
    pos = rng.normal(2, 0.3, (30, 3))
    neg = rng.normal(-2, 0.3, (60, 3))
    X, y = np.vstack([pos, neg]), np.array([1] * 30 + [0] * 60)
    q = np.array([[2, 2, 2], [-2, -2, -2]], float)
    s = ob.knn_scores(X, y, q, k=5)
    assert s["vote"][0] == 1.0 and s["vote"][1] == 0.0 and s["margin"][0] > 0 > s["margin"][1]
    loo = ob.knn_scores(X, y, X, k=5, exclude_self=True)
    assert ob.auc(y, loo["vote"]) > 0.99


def test_beyond_p0_detects_added_information():
    rng = np.random.default_rng(2)
    n = 1500

    def make():
        z = rng.normal(size=n)
        s = rng.normal(size=n)
        y = (rng.random(n) < 1 / (1 + np.exp(-(z + 1.5 * s)))).astype(float)
        return y, 1 / (1 + np.exp(-z)), s

    yf, p0f, sf = make()
    yt, p0t, st = make()
    r = ob.beyond_p0(yf, p0f, sf, yt, p0t, st, np.arange(n), 200)
    assert r["auc_gain"] > 0.05 and r["auc_gain_ci"][0] > 0 and r["coef_s"] > 0.8
    noise = ob.beyond_p0(yf, p0f, rng.normal(size=n), yt, p0t, rng.normal(size=n), np.arange(n), 200)
    assert noise["auc_gain_ci"][0] < 0.01 < 0.05


def test_auc_diff_with_ci_is_paired():
    rng = np.random.default_rng(3)
    y = (rng.random(400) < 0.3).astype(float)
    a = y + rng.normal(0, 0.5, 400)
    b = rng.normal(size=400)
    d = ob.auc_diff_with_ci(y, a, b, np.arange(400) // 2, 200)
    assert d["diff"] > 0.2 and d["ci"][0] > 0


def test_realised_gain_formula():
    table = {("A1", "q", 2.0, "b1"): (4.0, 1), ("A2", "q", 2.0, "b1"): (4.0, 0), ("A3", "q", 2.0, "b9"): (None, 1)}
    g = ob.realised_gains(table, {"A1": 0.2, "A2": 0.2, "A3": 0.2})
    post = 0.2 * 4 / (0.2 * 4 + 0.8)
    assert g[("A1", "q", 2.0)] == pytest.approx(0.8**2 - (1 - post) ** 2)
    assert g[("A2", "q", 2.0)] < 0 and ("A3", "q", 2.0) not in g


def test_shape_typology_cue():
    assert ob.shape_typology_cue({"in_deg": "8+", "out_deg": "1"}) == "fan-in"
    assert ob.shape_typology_cue({"in_deg": "0", "out_deg": "4-7"}) == "fan-out"
    assert ob.shape_typology_cue({"in_deg": "4-7", "out_deg": "8+"}) == "scatter-gather"
    assert ob.shape_typology_cue({"in_deg": "1", "out_deg": "0"}) == ""


def test_library_eval_hit_rate_by_cutoff_and_gain():
    from types import SimpleNamespace

    from ftm.library.shape import SituationShape
    from ftm.library.store import Strategy, StrategyLibrary

    sh = SituationShape("any", "1", "0", "2", "1k-10k flat", "<6h")
    far = SituationShape("any", "8+", "8+", "5+", "100k+ varied", ">3d")
    lib = StrategyLibrary("shape")
    lib.add(Strategy("S0", "success", "t", "q1", None, 2.0, 30, 0.01, 0.001, True, shape=sh.as_dict()))
    rows = [{"ctx": SimpleNamespace(alert_id=f"A{i}", summary={"shape": (sh if i < 6 else far).as_dict() | {"receiver": f"R{i}"}, "sender": f"S{i}", "receiver": f"R{i}"}),
             "attempt": None} for i in range(8)]
    gains = {(f"A{i}", "q1", 2.0): 0.02 for i in range(6)}
    r = ob.library_eval(lib, rows, gains, cutoffs=(0.0, 2.0, float("inf")), n_boot=50)
    assert r["by_cutoff"]["0.0"]["hit_rate"] == 6 / 8  # exact key only
    assert r["by_cutoff"]["2.0"]["hit_rate"] == 6 / 8  # the far shape is beyond two buckets
    assert r["by_cutoff"]["inf"]["hit_rate"] == 1.0  # no cutoff: the least bad note is returned for everyone
    assert r["by_cutoff"]["0.0"]["success_mean_realised_gain"] == pytest.approx(0.02)


# -- matrix and report -----------------------------------------------------------------------------------------------------
def test_arm_registry_one_factor_and_unique():
    arms = rb.build_arms()
    names = [a.name for a in arms]
    assert len(set(names)) == len(names)
    for g in {a.group for a in arms}:
        assert sum(a.is_default for a in arms if a.group == g) == 1  # every comparison group has exactly one default
    comps = {a.component for a in arms}
    assert {"evidence", "rules", "library", "memory", "finfre_pool"} <= comps
    withouts = {"ev_none", "rules_none", "lib_none", "sharma_native_mem_off", "torres_native_mem_off", "co_investigator_native_mem_off",
                "finfre_rag_native_zero_shot", "finfre_rag_ported_zero_shot"}
    assert withouts <= set(names)  # a WITHOUT arm for every component
    for a in arms:  # one factor: at most one flag family changes per arm
        assert len([f for f in a.flags if f.startswith("--")]) <= 1


def test_blocked_by_missing_flags_and_library_files(monkeypatch):
    monkeypatch.setattr(rb, "lib_path", lambda kind, part: "outputs/_no_such_library.json" if kind == "account" else None)
    ev_none = next(a for a in rb.build_arms() if a.name == "ev_none")
    assert rb.blocked(ev_none, "ap.add_argument('--evidence-view', help='amount | scorer')") is not None
    ok_src = "ap.add_argument('--evidence-view', help='none | amount | scorer')"
    assert rb.blocked(ev_none, ok_src) is None
    rules = next(a for a in rb.build_arms() if a.name == "rules_whole")
    assert rb.blocked(rules, "nothing") and rb.blocked(rules, "ap.add_argument('--rules', choices=[...])") is None
    acct = next(a for a in rb.build_arms() if a.name == "lib_account")
    assert "library file missing" in (rb.blocked(acct, "") or "")


def test_estimates_and_plan_table():
    t = rb.plan_table("")
    assert "planner_default" in t and "GPU-h" in t
    assert 0 < sum(a.est_hours() for a in rb.build_arms()) < 40


def _fake_arm(name, y, p, p0, tokens_in=1000.0, nv=0.0):
    ex = {"n": len(y), "no_verdict_rate": nv, "questions_per_alert": 1.5, "accepted_question_share": 0.5, "eps_per_alert": 2.0, "llm_calls": 2.0,
          "tokens_in": tokens_in, "tokens_out": 200.0, "prompt_tokens_per_call": 500.0, "median_s": 5.0, "p90_s": 9.0, "gpu_hours": 0.1, "chunks": 1}
    return rb.Loaded(name, tuple(f"V{i:06d}" for i in range(len(y))), y, p, p0, ex)


def _synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(z - 0.5)))).astype(float)
    p0 = np.clip(1 / (1 + np.exp(-(0.5 * z))), 0.01, 0.99)
    return y, p0, z


def test_group_report_selects_a_better_arm_and_keeps_default_otherwise():
    y, p0, z = _synthetic()
    better = np.clip(1 / (1 + np.exp(-(1.0 * z - 0.5))), 0.01, 0.99)  # closer to the truth than p0
    arms = {"d": _fake_arm("d", y, p0 * 0.98 + 0.01, p0), "better": _fake_arm("better", y, better, p0, tokens_in=1200),
            "same": _fake_arm("same", y, np.clip(p0 * 0.98 + 0.0101, 0, 1), p0, tokens_in=1100)}
    cl = np.arange(len(y)) // 3
    rep = rb.group_report(arms, "d", cl, 0.3227, 200)
    assert set(rep["vs_default"]) == {"better", "same"} and "p_holm" in rep["vs_default"]["better"]["brier"]
    sel = rb.select_component(list(arms), "d", rep, {})
    assert sel["winner"] == "better" and sel["reason"].startswith("tier 1")
    only_same = {k: arms[k] for k in ("d", "same")}
    rep2 = rb.group_report(only_same, "d", cl, 0.3227, 200)
    assert rb.select_component(list(only_same), "d", rep2, {})["winner"] == "d"


def test_tier2_cheaper_noninferior_and_collapse_and_no_verdict_block():
    y, p0, z = _synthetic(seed=1)
    d = np.clip(p0, 0.01, 0.99)
    cheap = np.clip(p0 + 0.0005, 0.01, 0.99)
    arms = {"d": _fake_arm("d", y, d, p0, 1000.0), "cheap": _fake_arm("cheap", y, cheap, p0, 600.0),
            "const": _fake_arm("const", y, np.full_like(y, y.mean()) + 0 * z, p0, 400.0), "noverdict": _fake_arm("noverdict", y, cheap, p0, 300.0, nv=0.2)}
    rep = rb.group_report(arms, "d", np.arange(len(y)) // 3, 0.3227, 200)
    assert rep["collapse"]["const"]["shrunk"] and rep["collapse"]["const"]["collapsed"]
    sel = rb.select_component(list(arms), "d", rep, {})
    assert sel["winner"] == "cheap" and sel["reason"].startswith("tier 2")
    assert not next(c for c in sel["candidates"] if c["arm"] == "noverdict")["eligible"]


def test_winner_flags():
    sel = {"planner/evidence": {"winner": "ev_amount_1hop", "default": "planner_default"},
           "planner/rules": {"winner": "planner_default", "default": "planner_default"},
           "planner/library": {"winner": "lib_none", "default": "planner_default"}}
    flags, lib = rb.winner_flags(sel)
    assert flags == ("--evidence-view", "amount_1hop") and lib == "none"


def test_load_arm_from_run_logs(tmp_path):
    for part, ids in (("fit", ["V000001", "V000002"]), ("cal", ["V000003"])):
        c = tmp_path / "a" / part / "c000"
        c.mkdir(parents=True)
        rows = [{"alert_id": i, "label": k % 2, "p0": 0.2, "probability": None if i == "V000003" else 0.4, "seconds": 3.0, "gpu_seconds": 2.0, "llm_calls": 2,
                 "tokens_in": 1000, "tokens_out": 100, "eps_spent": 2.0} for k, i in enumerate(ids)]
        (c / "per_alert.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        (c / "summary.json").write_text("{}")
        pl.DataFrame({"event": ["step", "step"], "alert_id": [ids[0], ids[0]], "log_lr": [0.1, None]}).write_parquet(c / "runlog.parquet")
    a = rb.load_arm(tmp_path, "a")
    assert a.ids == ("V000001", "V000002", "V000003") and a.extras["accepted_question_share"] == 0.5
    assert a.extras["questions_per_alert"] == pytest.approx(4 / 3) and a.p[2] == 0.2  # no verdict = p0
    assert rb.load_arm(tmp_path, "missing") is None


def test_shell_script_syntax_and_dry_run(tmp_path):
    sh = ROOT / "scripts" / "t3_rag_bench.sh"
    assert subprocess.run(["bash", "-n", str(sh)]).returncode == 0
    r = subprocess.run(["bash", str(sh), "arms"], env={"DRY_RUN": "1", "ARMS": "lib_none", "OUT": str(tmp_path / "o"), "PY": sys.executable,
                                                        "PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"}, capture_output=True, text=True, cwd=ROOT)
    assert "--profile tuned" in r.stdout and "--library" not in r.stdout.split("DRY: " + sys.executable)[1].split("\n")[0]
    assert not (tmp_path / "o").exists()  # a dry run creates nothing


# -- no val label may reach a fit/cal arm through its library (fixed 7 Nov, before any GPU run of the bench) ----------------------------------------
TRAIN_ONLY = ("outputs/t3_library/frozen_v3/shape_gemma_train.json", "outputs/t3_library/frozen_v3/account_gemma_train.json", "outputs/rag_bench/libs/shape_backtest_train.json")
BUILT_WITH_VAL_RUNS = ("outputs/t3_library/frozen/", "outputs/t3_library/frozen_v2/", "outputs/rag_bench/libs/shape_tuned_teacher", "outputs/rag_bench/libs/account_tuned_teacher")


def test_fit_and_cal_arms_only_get_train_built_libraries_and_eval_gets_the_full_ones():
    for arm in rb.build_arms():
        fitcal, ev = rb.lib_path(arm.library, "fitcal"), rb.lib_path(arm.library, "eval")
        assert (fitcal is None) == (arm.library == "none") and (ev is None) == (arm.library == "none")
        if fitcal is None:
            continue
        assert fitcal in TRAIN_ONLY, f"{arm.name}: fit/cal would read {fitcal}"
        assert not fitcal.startswith(BUILT_WITH_VAL_RUNS) and not (ev or "").startswith(BUILT_WITH_VAL_RUNS)
        if arm.library in ("shape", "account"):
            assert ev == fitcal.replace("_train.json", ".json") and ev != fitcal  # eval takes the full library, never the train one
    with pytest.raises(ValueError):
        rb.lib_path("shape", "fit")  # the part must be said explicitly


def test_the_scripts_on_the_val_path_name_no_library_built_with_val_runs():
    """Every script and config except the listed ones is free of the old libraries.  Listed, with the reason each is not a fit/cal path:
    configs/final/jobs.yaml (T4's final-run plan for the TEST split, where the full library is the right one; reported, not changed), scripts/rag_offline_bench.py (descriptive offline
    comparison of the old and the teacher-only libraries on val; it is what showed the difference), scripts/t3_qwen_session.sh (E16 on val `eval` only) and scripts/t3_val_session.sh
    (the old session that BUILT those libraries; superseded by scripts/t3_rerun.sh)."""
    allowed = {"configs/final/jobs.yaml", "scripts/rag_offline_bench.py", "scripts/t3_qwen_session.sh", "scripts/t3_val_session.sh", "scripts/t3_rag_bench_report.py", "tests/test_rag_offline_bench.py"}
    hits = []
    for base in ("scripts", "configs", "ftm"):
        for f in (ROOT / base).rglob("*"):
            if f.suffix not in (".py", ".sh", ".yaml", ".json") or "__pycache__" in f.parts or f.relative_to(ROOT).as_posix() in allowed:
                continue
            text = f.read_text(errors="ignore")
            for bad in ("t3_library/frozen/", "t3_library/frozen_v2", "shape_tuned_teacher", "account_tuned_teacher"):
                if bad in text:
                    hits.append(f"{f.relative_to(ROOT)}: {bad}")
    assert not hits, hits
