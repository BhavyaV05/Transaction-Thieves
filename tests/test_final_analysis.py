"""The final analysis pipeline (scripts/final_analysis.py, ftm/eval/final.py, ftm/eval/final_tables.py) on synthetic run logs.

Owner: T4 (Shreyas).  No data, no LLM: a synthetic run directory with systems of known skill (a probability is the one-bank
log-odds shifted toward the truth by `skill`), so every verdict has a reason that can be read off the planted skills.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from ftm import guards
from ftm.eval import final as fn
from ftm.eval import final_tables as ft
from ftm.eval import stats as st
from ftm.fusion.fuse import sigmoid
from ftm.harness.runlog_rows import to_frame

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("final_analysis", ROOT / "scripts" / "final_analysis.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

N = 900
N_BOOT = 1000

# planted skills: log-odds shift toward the truth (0 = no better than one bank)
SKILL = {
    "ours_compose": 1.6, "ours_fixed_list": 0.5, "ours_no_backtest": -0.1, "ours_llm_decides": 0.8, "ours_act_only": 1.2,
    "random": 0.05, "fixed": 0.35, "eig": 0.5, "learned": 0.8, "enumerator": 0.6, "raw_share": 1.8,
    "eig_batched": 0.45,
}
EXTERNAL = {  # (system, mode) -> skill
    ("react", "ported"): 2.6, ("torres_ported", "ported"): 1.0, ("torres_native", "native"): 0.5, ("sharma_ported", "ported"): 0.3,
    ("sharma_native", "native"): 0.0, ("rewoo", "ported"): 1.0,
}
UPPER = {"gfp_see_everything": 3.0, "best_sota_see_everything": 3.4, "dp_fedavg": 1.2}


def _logit(p):
    return np.log(p / (1 - p))


class World:
    def __init__(self, split: str = "val", seed: int = 0, ours_skill: float | None = None, useless_frac: float = 0.0):
        rng = np.random.default_rng(seed)
        self.split = split
        self.rng = rng
        self.ids = [f"{'V' if split == 'val' else 'T'}{i:05d}" for i in range(N)]
        self.y = (rng.random(N) < 0.3).astype(int)
        self.z0 = _logit(0.3) + 0.4 * (2 * self.y - 1) + rng.normal(0, 0.8, N)
        self.p0 = sigmoid(self.z0)
        grp = np.arange(N) // 3  # clusters of three alerts share an attempt
        self.alerts = pl.DataFrame({
            "alert_id": self.ids, "label": self.y, "attempt_id": [f"A{g}" for g in grp], "src_id": 2 * np.arange(N), "dst_id": 2 * np.arange(N) + 1,
            "typology": np.where(np.arange(N) % 2 == 0, "cycle", "fan-out"), "home_bank": np.where(np.arange(N) % 3 == 0, "1", "2"),
            "p0": self.p0, "val_part": "eval", "test_part": "bulk", "spans_split": np.arange(N) % 10 == 0})  # fmt: skip
        self.subset = sorted(rng.choice(self.ids, 300, replace=False).tolist())
        self.skill = dict(SKILL)
        if ours_skill is not None:
            self.skill["ours_compose"] = ours_skill
        self.useless_frac = useless_frac

    def prob(self, skill: float, noise: float = 0.6) -> np.ndarray:
        return np.clip(sigmoid(self.z0 + skill * (2 * self.y - 1) + self.rng.normal(0, noise, N)), 1e-4, 1 - 1e-4)


def _verdicts(w: World, system, variant, mode, ids, probs, seeds=(0,), cost=None, no_verdict=0.0, stop=("budget", "finish"), base=None):
    rows = []
    idx = {a: i for i, a in enumerate(w.ids)}
    cost = cost or {}
    for seed in seeds:
        for a in ids:
            i = idx[a]
            p = float(probs[seed][i]) if isinstance(probs, dict) else float(probs[i])
            if no_verdict and w.rng.random() < no_verdict:
                p = None
            rows.append({"run_id": "r", "system": system, "variant": variant, "seed": seed, "split": w.split, "alert_id": a, "event": "verdict",
                         "probability": p, "p0": float(w.p0[i]), "mode": mode, "stop_reason": stop[(i + seed) % len(stop)], "model": "m",
                         "machine": "t3-5070ti", "label": 1 - int(w.y[i]), **cost})  # the label written here is wrong on purpose
    return rows


def _steps(w: World, system, variant, mode, ids, seed=0, n_programs=40, per_alert=2, useless_frac=0.0):
    """Answered, back-tested question rows. Good programs: P(true | launder) .8 vs .3; useless programs: independent of the label."""
    rng = np.random.default_rng(100 + seed)
    idx = {a: i for i, a in enumerate(w.ids)}
    n_useless = int(round(useless_frac * n_programs))
    rows = []
    for a in ids:
        i = idx[a]
        for s in range(per_alert):
            q = int(rng.integers(0, n_programs))
            good = q >= n_useless
            p_true = (0.8 if w.y[i] else 0.3) if good else 0.5
            ans = bool(rng.random() < p_true)
            llr = float(np.log(0.8 / 0.3) if ans else np.log(0.2 / 0.7))
            rows.append({"run_id": "r", "system": system, "variant": variant, "seed": seed, "split": w.split, "alert_id": a, "event": "step", "step": s,
                         "bank": "7", "program_canonical": f"q{q}()", "eps": 1.0, "gate": "allowed", "response_kind": "reply",
                         "answer_value": json.dumps(ans), "family": f"f{q % 4}", "kind": "bool", "log_lr": llr, "mode": mode})  # fmt: skip
    return rows


def _write(rows, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = to_frame(rows)
    extra = {"mode": [r.get("mode") for r in rows], "label": [r.get("label") for r in rows]}
    df = df.with_columns(pl.Series("mode", extra["mode"], dtype=pl.Utf8), pl.Series("label", extra["label"], dtype=pl.Int64))
    df.write_parquet(path)
    return path


def _injection_rows(w: World, rng, checker: bool) -> list[dict]:
    rows = []
    kinds = ["direct", "authority", "smuggling", "exfil", "benign"]
    for i in range(50):
        kind = kinds[i % 5]
        item = f"{kind}-{i:03d}"
        alert = f"inj{'c' if checker else 'n'}{i:03d}"
        success = kind != "benign" and rng.random() < (0.1 if checker else 0.8)
        base = {"run_id": "r", "system": "ours_compose", "variant": "harness_inj_checker" if checker else "harness_inj_nochecker", "seed": 0,
                "split": w.split, "alert_id": alert, "mode": "ported"}  # fmt: skip
        rows.append({**base, "event": "step", "step": 0, "bank": "7", "program_canonical": "q1()", "eps": 1.0, "gate": "allowed", "response_kind": "reply",
                     "injection_id": item})  # fmt: skip
        goal_exfil = kind == "exfil"
        if not success:
            rows.append({**base, "event": "step", "step": 1, "bank": "7", "program_canonical": "q2()", "eps": 1.0, "gate": "allowed", "response_kind": "reply"})
        elif goal_exfil:
            rows.append({**base, "event": "step", "step": 1, "bank": "7", "program_canonical": "ALERT.receiver.has_account(99999)", "eps": 1.0,
                         "gate": "blocked" if checker else "allowed", "response_kind": None})  # fmt: skip
        rows.append({**base, "event": "verdict", "probability": 0.01 if (success and not goal_exfil) else 0.5, "p0": 0.3,
                     "stop_reason": "finish" if success else "budget"})  # fmt: skip
    return rows


def build_run(root: Path, w: World | None = None, split: str = "val", drop_jobs: tuple[str, ...] = (), with_injection: bool = True,
              seeds=(0, 1), label_noise: bool = True) -> dict:  # fmt: skip
    w = w or World(split)
    rng = np.random.default_rng(7)
    run_dir = root / "final" / "run1"
    jobs, files = [], {}

    def add(name, system, variant, mode, alert_set, rows):
        if name in drop_jobs:
            return
        p = _write(rows, run_dir / f"{name}.parquet")
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        jobs.append({"system": system, "variant": variant, "mode": mode, "alert_set": alert_set, "seeds": list(seeds), "runlog": p.name, "sha256": sha})
        files[name] = p

    ours_cost = {"llm_calls": 6, "tokens_in": 3000, "tokens_out": 400, "gpu_seconds": 7.5, "eps_spent": 1.8}
    pr = {s: w.prob(w.skill["ours_compose"]) for s in seeds}
    rows = _verdicts(w, "ours_compose", "harness", "ported", w.ids, pr, seeds, ours_cost) + _steps(w, "ours_compose", "harness", "ported", w.ids,
                                                                                                    useless_frac=w.useless_frac)  # fmt: skip
    add("ours_full", "ours_compose", "harness", "ported", "full_test", rows)
    pr = {s: w.prob(w.skill["ours_compose"]) for s in seeds}
    add("ours_subset", "ours_compose", "harness", "ported", "subset", _verdicts(w, "ours_compose", "harness", "ported", w.subset, pr, seeds, ours_cost))
    for name, system, variant in (("llm_fixed", "ours_fixed_list", "harness"), ("llm_nobt", "ours_no_backtest", "harness"),
                                  ("llm_dec", "ours_llm_decides", "harness"), ("llm_act", "ours_act_only", "harness")):
        add(name, system, variant, "ported", "full_test", _verdicts(w, system, variant, "ported", w.ids, {s: w.prob(w.skill[system]) for s in seeds}, seeds,
                                                                      {"llm_calls": 5, "gpu_seconds": 6.0, "eps_spent": 1.9}))  # fmt: skip
    for name, system, variant in (("random", "random", "random"), ("fixed", "fixed", "fixed_order"), ("eig", "eig", "eig"), ("learned", "learned", "learned_policy"),
                                  ("enum", "enumerator", "enumerator"), ("raw", "raw_share", "raw_share")):
        add(name, system, variant, "", "full_test", _verdicts(w, system, variant, "", w.ids, {s: w.prob(w.skill[system]) for s in seeds}, seeds,
                                                               {"eps_spent": 2.0, "sim_days": 2.0}))  # fmt: skip
    add("eig_batched", "eig", "eig_batched_2_1", "", "full_test", _verdicts(w, "eig", "eig_batched_2_1", "", w.ids, {s: w.prob(0.45) for s in seeds}, seeds,
                                                                              {"eps_spent": 2.0, "sim_days": 1.0}))  # fmt: skip
    add("view_pcst", "ours_compose", "harness_view_scorer_pcst", "ported", "full_test",
        _verdicts(w, "ours_compose", "harness_view_scorer_pcst", "ported", w.ids, {s: w.prob(2.0) for s in seeds}, seeds))  # fmt: skip
    add("view_top40", "ours_compose", "harness_view_top40", "ported", "full_test",
        _verdicts(w, "ours_compose", "harness_view_top40", "ported", w.ids, {s: w.prob(1.0) for s in seeds}, seeds))  # fmt: skip
    for (system, mode), skill in EXTERNAL.items():
        nov = 0.4 if system == "rewoo" else 0.0
        rows = _verdicts(w, system, "harness", mode, w.subset, {s: w.prob(skill) for s in seeds}, seeds, {"llm_calls": 9, "gpu_seconds": 12.0, "eps_spent": 1.5},
                         no_verdict=nov) + _steps(w, system, "harness", mode, w.subset, per_alert=1)  # fmt: skip
        add(f"ext_{system}", system, "harness", mode, "subset", rows)
    if with_injection:
        add("inj_nochecker", "ours_compose", "harness_inj_nochecker", "ported", "full_test", _injection_rows(w, rng, False))
        add("inj_checker", "ours_compose", "harness_inj_checker", "ported", "full_test", _injection_rows(w, rng, True))
    manifest = {"run_id": "run1", "split": split, "subset_sha256": fn.ids_sha256(w.subset), "jobs": jobs}
    (run_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    # inputs
    inp = root / "inputs"
    inp.mkdir(exist_ok=True)
    w.alerts.write_parquet(inp / "alerts.parquet")
    paths = {}
    for name, skill in UPPER.items():
        pl.DataFrame({"alert_id": w.ids, "p0": w.prob(skill)}).write_parquet(inp / f"{name}.parquet")
        paths[name] = str(inp / f"{name}.parquet")
    pl.DataFrame({"view": ["scorer_pcst"] * N + ["top40_by_amount"] * N, "alert_id": w.ids * 2,
                  "pattern_edge_share": np.r_[np.clip(rng.normal(0.5, 0.1, N), 0, 1), np.clip(rng.normal(0.2, 0.1, N), 0, 1)]}).write_parquet(inp / "edges.parquet")
    rows_df = pl.concat([
        w.alerts.select("alert_id", "label", "p0", "attempt_id", "src_id", "dst_id", "spans_split"),
        pl.DataFrame({"alert_id": [None] * 1500, "label": (rng.random(1500) < 0.03).astype(int), "p0": rng.uniform(0.001, 0.05, 1500),
                      "attempt_id": [None] * 1500, "src_id": 10_000 + np.arange(1500), "dst_id": 20_000 + np.arange(1500), "spans_split": [False] * 1500}),
    ], how="vertical_relaxed")
    rows_df = rows_df.with_columns(pl.col("p0").alias("p_gfp_see_everything"))
    rows_df.write_parquet(inp / "rows.parquet")
    (inp / "fast.json").write_text(json.dumps({"decision_types": [{"name": "stop", "alpha": 0.05, "error_marginal": 0.04, "kept": 0.6, "speedup": 12.0}]}))
    inputs = {
        "alerts": {"path": str(inp / "alerts.parquet"), "account_cols": ["src_id", "dst_id"], "headline": {}, "tail": {}},
        "scores": {"one_bank": {"path": None}, **{n: {"path": p, "key": ["alert_id"], "value": "p0", "filter": {}} for n, p in paths.items()}},
        "rows": {"path": str(inp / "rows.parquet")}, "fast_model": {"path": str(inp / "fast.json")},
        "evidence_edges": {"path": str(inp / "edges.parquet")}, "fixed_on_val": None,
    }
    (inp / "inputs.yaml").write_text(yaml.safe_dump(inputs))
    return {"world": w, "run_dir": run_dir, "inputs": inp / "inputs.yaml", "files": files, "root": root, "manifest": run_dir / "MANIFEST.json"}


@pytest.fixture(scope="module")
def val_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("val")
    env = build_run(root)
    out = root / "out"
    rc = script.main(["--split", "val", "--run", str(env["run_dir"]), "--inputs", str(env["inputs"]), "--outdir", str(out), "--n-boot", str(N_BOOT),
                      "--no-figures"])  # fmt: skip
    assert rc == 0
    env["out"] = out
    env["hyp"] = json.loads((out / "hypotheses.json").read_text())
    return env


# ---- verdicts follow the planted skills -----------------------------------------------------------------------------
def test_verdicts_follow_planted_skills(val_run):
    h = val_run["hyp"]
    for hid in ("H1", "H2", "H3", "H4", "H7", "H10", "H6"):
        assert h[hid]["verdict"] == "supported", (hid, h[hid]["verdict_text"])
    assert h["H9"]["verdict"] == "not_supported", h["H9"]["verdict_text"]  # react (skill 2.6) is clearly better than ours (1.6)
    assert h["H9"]["best_ported_external"] == "react"
    assert h["H1"]["estimate"] >= 0.30 and h["H1"]["ci"][0] > 0
    assert h["H2"]["ci"][0] >= 0.70
    assert h["H3"]["best_simple_planner"] == "learned"  # the best simple planner is derived on validation
    assert h["H4"]["fdr_audit"]["k_useless"] == 0 and h["H4"]["fdr_audit"]["upper95"] < 0.10
    assert h["H8"]["verdict"].startswith("met") and h["H8"]["label"] == "exploratory, uncorrected"
    assert h["H5"]["verdict"].startswith("met")
    assert all(h[x]["n_alerts"] == N for x in ("H1", "H2", "H3", "H4", "H7"))
    assert h["H9"]["n_alerts"] == 300  # the external agents run on the subset only


def test_h3_fails_when_the_llm_is_no_better_than_the_simple_planners(tmp_path):
    env = build_run(tmp_path, World("val", ours_skill=0.05))
    out = tmp_path / "out"
    assert script.main(["--split", "val", "--run", str(env["run_dir"]), "--inputs", str(env["inputs"]), "--outdir", str(out), "--n-boot", str(N_BOOT),
                        "--no-figures"]) == 0  # fmt: skip
    h = json.loads((out / "hypotheses.json").read_text())
    assert h["H1"]["verdict"] == "not_supported" and h["H3"]["verdict"] == "not_supported" and h["H2"]["verdict"] == "not_supported"
    assert h["H7"]["verdict"] == "not_supported"  # ours no longer beats the LLM that decides


def test_h4_fails_when_many_accepted_questions_are_useless(tmp_path):
    env = build_run(tmp_path, World("val", useless_frac=0.5))
    out = tmp_path / "out"
    script.main(["--split", "val", "--run", str(env["run_dir"]), "--inputs", str(env["inputs"]), "--outdir", str(out), "--n-boot", str(N_BOOT), "--no-figures"])
    h = json.loads((out / "hypotheses.json").read_text())["H4"]
    assert h["components"][0]["rule_met"] is True  # back-testing still matters for the Brier
    assert h["fdr_audit"]["share"] > 0.3 and h["verdict"] == "not_supported"


# ---- Holm within families --------------------------------------------------------------------------------------------
def test_holm_is_applied_within_families_only(val_run):
    h = val_run["hyp"]
    prim = st.holm({k: h[k]["p"] for k in ("H1", "H2", "H3", "H4", "H9")})
    sec = st.holm({k: h[k]["p"] for k in ("H6", "H7", "H10")})
    for k, v in prim.items():
        assert h[k]["p_adj"] == pytest.approx(v["p_adj"]) and h[k]["family"] == "primary" and h[k]["family_size"] == 5
    for k, v in sec.items():
        assert h[k]["p_adj"] == pytest.approx(v["p_adj"]) and h[k]["family"] == "secondary" and h[k]["family_size"] == 3
    allp = st.holm({k: h[k]["p"] for k in ("H1", "H2", "H3", "H4", "H9", "H6", "H7", "H10")})
    assert any(abs(allp[k]["p_adj"] - h[k]["p_adj"]) > 1e-9 for k in allp)  # pooling all eight would give other adjusted p-values
    for k in ("H5", "H8"):
        assert h[k]["p_adj"] is None and h[k]["family"] == "exploratory"


def test_not_run_member_keeps_its_slot_in_the_family(tmp_path):
    env = build_run(tmp_path, drop_jobs=("raw",))
    out = tmp_path / "out"
    script.main(["--split", "val", "--run", str(env["run_dir"]), "--inputs", str(env["inputs"]), "--outdir", str(out), "--n-boot", str(N_BOOT), "--no-figures"])
    h = json.loads((out / "hypotheses.json").read_text())
    assert h["H2"]["status"] == "not_run" and "raw_share" in h["H2"]["reason"] and h["H2"]["p_adj"] == 1.0
    p = {k: (h[k]["p"] if h[k]["status"] == "tested" else 1.0) for k in ("H1", "H2", "H3", "H4", "H9")}
    assert h["H1"]["p_adj"] == pytest.approx(st.holm(p)["H1"]["p_adj"])  # the family size stays 5
    assert h["H1"]["verdict"] == "supported"


# ---- missing inputs are reported, never skipped -----------------------------------------------------------------------
def test_missing_inputs_are_not_run_with_a_reason(tmp_path):
    env = build_run(tmp_path)
    inp = yaml.safe_load(env["inputs"].read_text())
    inp["scores"]["dp_fedavg"]["path"] = None
    inp["scores"]["best_sota_see_everything"]["path"] = None
    inp["evidence_edges"]["path"] = None
    inp["fast_model"]["path"] = None
    inp["rows"]["path"] = None
    env["inputs"].write_text(yaml.safe_dump(inp))
    out = tmp_path / "out"
    script.main(["--split", "val", "--run", str(env["run_dir"]), "--inputs", str(env["inputs"]), "--outdir", str(out), "--n-boot", str(N_BOOT), "--no-figures"])
    h = json.loads((out / "hypotheses.json").read_text())
    assert h["H10"]["status"] == "not_run" and "dp_fedavg" in h["H10"]["reason"]
    assert h["H6"]["status"] == "not_run" and "pattern-edge" in h["H6"]["reason"]
    assert h["H5"]["status"] == "not_run"
    assert "only" in " ".join(h["H1"]["notes"])  # tested against the one upper limit that exists, said so
    assert len(h["H1"]["components"]) == 1
    status = json.loads((out / "status.json").read_text())
    assert status["end_to_end_all_rows"]["status"] == "not_run"
    for hid, r in h.items():
        assert r["verdict"] is not None
    nums = json.loads((out / "numbers.json").read_text())
    assert nums["gap_closed_sota"]["status"] == "not_run"


def test_ambiguous_role_is_reported(tmp_path):
    env = build_run(tmp_path)
    s = yaml.safe_load((ROOT / "configs/analysis/systems.yaml").read_text())
    s["roles"]["no_backtest"] = {"system": "ours_*", "variant": ["harness"]}
    sp = tmp_path / "systems.yaml"
    sp.write_text(yaml.safe_dump(s))
    out = tmp_path / "out"
    script.main(["--split", "val", "--run", str(env["run_dir"]), "--inputs", str(env["inputs"]), "--systems", str(sp), "--outdir", str(out),
                 "--n-boot", str(N_BOOT), "--no-figures"])  # fmt: skip
    h = json.loads((out / "hypotheses.json").read_text())
    assert h["H4"]["status"] == "not_run" and "ambiguous" in h["H4"]["reason"]


# ---- labels ---------------------------------------------------------------------------------------------------------
def test_labels_in_run_logs_are_ignored(val_run, tmp_path):
    env = build_run(tmp_path)  # same seed, but this builder flips... the logs' label column is wrong in both: results must not depend on it
    run = fn.load_run("val", env["run_dir"])
    assert "label" not in run.log.columns
    out = tmp_path / "out"
    script.main(["--split", "val", "--run", str(env["run_dir"]), "--inputs", str(env["inputs"]), "--outdir", str(out), "--n-boot", str(N_BOOT), "--no-figures"])
    h = json.loads((out / "hypotheses.json").read_text())
    assert h["H1"]["estimate"] == pytest.approx(val_run["hyp"]["H1"]["estimate"])
    assert h["H1"]["verdict"] == "supported"  # logs carry the inverted label; had it been used, ours would look worse than one bank


# ---- test split guard and fixed values ----------------------------------------------------------------------------------
def test_test_split_is_refused_without_the_env_var(tmp_path, monkeypatch):
    env = build_run(tmp_path, World("test"), split="test")
    monkeypatch.delenv("FTM_FINAL_RUN", raising=False)
    with pytest.raises(guards.TestSplitLocked):
        script.main(["--split", "test", "--run", str(env["run_dir"]), "--inputs", str(env["inputs"]), "--outdir", str(tmp_path / "o")])
    assert not (tmp_path / "o").exists()
    with pytest.raises(guards.TestSplitLocked):
        fn.load_run("test", env["run_dir"])


def test_script_never_sets_the_env_var():
    assert "FTM_FINAL_RUN" not in (ROOT / "scripts" / "final_analysis.py").read_text().replace("FTM_FINAL_RUN=1", "").replace("FTM_FINAL_RUN\"", "")


def test_val_analysis_refuses_test_rows(tmp_path):
    env = build_run(tmp_path, World("test"), split="test")
    with pytest.raises(fn.AnalysisInputError, match="never reads test rows"):
        fn.load_run("val", env["run_dir"])


def test_test_run_needs_the_values_fixed_on_validation(val_run, tmp_path, monkeypatch):
    env = build_run(tmp_path, World("test", seed=1), split="test")
    monkeypatch.setenv("FTM_FINAL_RUN", "1")
    argv = ["--split", "test", "--run", str(env["run_dir"]), "--inputs", str(env["inputs"]), "--outdir", str(tmp_path / "o"), "--n-boot", str(N_BOOT), "--no-figures"]
    assert script.main(argv) == 2  # no fixed_on_val.json: the margins and the best simple planner are not derived on test
    fixed = json.loads((val_run["out"] / "fixed_on_val.json").read_text())
    assert fixed["best_simple_planner"] == "learned" and fixed["equivalence_margin"] > 0 and fixed["best_ported_external"] == "react"
    assert script.main([*argv, "--fixed", str(val_run["out"] / "fixed_on_val.json")]) == 0
    h = json.loads((tmp_path / "o" / "hypotheses.json").read_text())
    assert h["H3"]["margin"] == pytest.approx(fixed["equivalence_margin"]) and h["H3"]["margin_source"] == "fixed on validation"
    assert h["H3"]["best_simple_planner_source"] == "fixed on validation"
    assert not (tmp_path / "o" / "fixed_on_val.json").exists()  # only a validation run fixes values
    monkeypatch.delenv("FTM_FINAL_RUN")


def test_test_refuses_never_result_variants(tmp_path, monkeypatch):
    env = build_run(tmp_path, World("test"), split="test", with_injection=False)
    monkeypatch.setenv("FTM_FINAL_RUN", "1")
    df = pl.read_parquet(env["files"]["eig"]).with_columns(pl.lit("harness_mock").alias("variant"))
    df.write_parquet(env["files"]["eig"])
    m = json.loads(env["manifest"].read_text())
    for j in m["jobs"]:
        j.pop("sha256")
    env["manifest"].write_text(json.dumps(m))
    with pytest.raises(fn.AnalysisInputError, match="never results"):
        fn.load_run("test", env["run_dir"], never_results=["harness_mock", "devtools"])
    monkeypatch.delenv("FTM_FINAL_RUN")


# ---- validation of the whole run before anything is printed --------------------------------------------------------------
def test_incomplete_run_prints_nothing(tmp_path, capsys):
    env = build_run(tmp_path)
    env["files"]["ext_react"].unlink()
    rc = script.main(["--split", "val", "--run", str(env["run_dir"]), "--inputs", str(env["inputs"]), "--outdir", str(tmp_path / "o")])
    cap = capsys.readouterr()
    assert rc == 2 and cap.out == "" and "ext_react.parquet" in cap.err
    assert not (tmp_path / "o").exists()


def test_checksum_mismatch_is_refused(tmp_path):
    env = build_run(tmp_path)
    p = env["files"]["eig"]
    p.write_bytes(p.read_bytes() + b"x")
    with pytest.raises(fn.AnalysisInputError, match="sha256"):
        fn.load_run("val", env["run_dir"])


def test_subset_hash_must_agree_across_jobs(tmp_path):
    env = build_run(tmp_path)
    m = json.loads(env["manifest"].read_text())
    sub = [j for j in m["jobs"] if j["alert_set"] == "subset"]
    sub[0]["subset_sha256"], sub[1]["subset_sha256"] = "a" * 64, "b" * 64
    env["manifest"].write_text(json.dumps(m))
    with pytest.raises(fn.AnalysisInputError, match="subset_sha256 differs"):
        fn.load_run("val", env["run_dir"])


def test_subset_jobs_on_different_alerts_are_refused(tmp_path):
    env = build_run(tmp_path)
    p = env["files"]["ext_react"]
    df = pl.read_parquet(p)
    drop = df["alert_id"].unique().to_list()[:5]
    df.filter(~pl.col("alert_id").is_in(drop)).write_parquet(p)
    m = json.loads(env["manifest"].read_text())
    for j in m["jobs"]:
        j.pop("sha256", None)
    env["manifest"].write_text(json.dumps(m))
    with pytest.raises(fn.AnalysisInputError, match="different alert sets|different numbers of alerts"):
        fn.load_run("val", env["run_dir"])


def test_logs_without_manifest(tmp_path):
    env = build_run(tmp_path)
    logs = [str(p) for k, p in env["files"].items() if not k.startswith("ext_") and not k.startswith("inj")]
    run = fn.load_run("val", None, logs)
    assert run.run_id == "logs" and run.manifest is None and len(run.jobs) == len(logs)


# ---- tables ----------------------------------------------------------------------------------------------------------
def test_table_5a_has_both_upper_limits_native_vs_ported_and_costs(val_run):
    rows = list(__import__("csv").DictReader((val_run["out"] / "table_5a.csv").open()))
    by = {(r["system"], r["mode"]): r for r in rows}
    assert ("torres", "native") in by and ("torres", "ported") in by and ("sharma", "native") in by
    r = by[("rewoo", "ported")]
    assert 0.25 < float(r["no-verdict rate"]) < 0.55  # the planted no-verdict agent
    assert r["gap e2e vs gfp"] and r["gap e2e vs sota"] and r["gap questions-only vs sota"]
    assert float(r["LLM calls"]) == 9 and float(r["GPU-s"]) == 12.0
    assert float(by[("react", "ported")]["brier e2e"]) < float(by[("sharma", "native")]["brier e2e"])
    assert any(r["system"].startswith("ours") for r in rows)
    assert (val_run["out"] / "table_5a.md").read_text().startswith("## Table 5a")


def test_table_5b_and_privacy_curve(val_run):
    rows = {r["arm"]: r for r in __import__("csv").DictReader((val_run["out"] / "table_5b.csv").open())}
    assert float(rows["Random planner"]["brier"]) > float(rows["LLM, compose (ours, eps 2)"]["brier"])
    assert rows["Fine-tuned planner (E16)"]["note"].startswith("not_run")  # arms without results are listed, not dropped
    curve = list(__import__("csv").DictReader((val_run["out"] / "privacy_curve.csv").open()))
    ceil = {float(r["eps"]): float(r["gap vs gfp"]) for r in curve if r["arm"] == "ceiling"}
    assert ceil[0.5] < ceil[1.0] < ceil[2.0] < ceil[4.0]  # the theoretical ceiling rises with eps
    assert any(r["arm"] == "ours" and r["eps"] == "2.0" for r in curve) and any(r["arm"] == "raw_share" for r in curve)
    assert any("not_run" in (r.get("note") or "") for r in curve)  # the eps variants this run does not have


def test_breakdowns_end_to_end_and_stop_reasons(val_run):
    import csv

    typ = list(csv.DictReader((val_run["out"] / "breakdown_typology.csv").open()))
    assert {r["typology"] for r in typ} == {"cycle", "fan-out"} and sum(int(r["n"]) for r in typ) == N
    bank = list(csv.DictReader((val_run["out"] / "breakdown_bank.csv").open()))
    assert {r["home_bank"] for r in bank} == {"1", "2"}
    e2e = list(csv.DictReader((val_run["out"] / "end_to_end_all_rows.csv").open()))
    allrows = next(r for r in e2e if r["level"].startswith("all rows"))
    assert int(allrows["rows"]) == N + 1500 and 0 < float(allrows["intake miss rate"]) < 1
    assert any(r["level"].startswith("kept only") for r in e2e) and any(r["level"].startswith("on alerts") for r in e2e)
    sr = list(csv.DictReader((val_run["out"] / "stop_reasons.csv").open()))
    assert {"budget", "finish"} <= {r["stop_reason"] for r in sr}


def test_injection_table_checker_lowers_attack_success(val_run):
    import csv

    rows = list(csv.DictReader((val_run["out"] / "injection.csv").open()))
    allatk = {r["checker"]: float(r["bad action executed"]) for r in rows if r["type"] == "all attacks"}
    assert allatk["off"] > 0.5 > allatk["on"]
    assert {"direct", "authority", "smuggling", "exfil", "benign"} <= {r["type"] for r in rows}
    summ = list(csv.DictReader((val_run["out"] / "injection_summary.csv").open()))
    assert summ[0]["system"] == "ours_compose"


def test_error_tags_and_case_studies(val_run):
    import csv

    cs = list(csv.DictReader((val_run["out"] / "case_studies.csv").open()))
    assert len(cs) == 10 and {r["kind"] for r in cs} >= {"error"}
    assert all(r["tag"] in ft.TAGS for r in cs)
    errs = [float(r["brier"]) for r in cs if r["kind"] == "error"]
    assert errs == sorted(errs, reverse=True)
    tags = list(csv.DictReader((val_run["out"] / "error_tags.csv").open()))
    row = next(r for r in tags if r["system"] == "ours_compose" and r["variant"] == "harness" and r["alerts"] == "full_test")
    assert sum(int(row[t]) for t in ft.TAGS) == int(row["wrong calls (p>=0.5 vs label)"])


def test_tag_errors_follows_the_pipeline_order():
    df = pl.DataFrame({"label": [1, 1, 1, 1, 1, 1, 0], "p0": [0.2] * 7, "p_e2e": [0.2, 0.2, 0.2, 0.2, 0.1, 0.2, 0.9], "p_qo": [0.2, 0.2, 0.2, 0.2, 0.3, 0.1, 0.2],
                       "probability": [None, 0.2, 0.2, 0.2, 0.1, 0.2, 0.9], "n_steps": [3, 0, 2, 2, 2, 2, 2], "n_allowed": [3, 0, 0, 2, 2, 2, 2],
                       "n_replies": [3, 0, 0, 0, 2, 2, 2], "n_lr": [3, 0, 0, 0, 2, 2, 2]})  # fmt: skip
    tags = ft.tag_errors(df)["tag"].to_list()
    assert tags[:5] == ["no_verdict", "no_question_asked", "all_blocked", "no_reply", "verdict_override"]
    assert tags[5] == "fusion_moved_wrong" and tags[6] == "verdict_override"


def test_power_statement(val_run):
    txt = (val_run["out"] / "power.txt").read_text()
    assert "0.0111" in txt or "0.0110" in txt  # 0.011 * sqrt(4000 / 900) = 0.0232 ... checked below
    assert "900 alerts" in txt and "300-alert external subset" in txt and "80% power" in txt


def test_compute_and_calibration_and_shuffled_tables(val_run):
    import csv

    comp = list(csv.DictReader((val_run["out"] / "compute.csv").open()))
    assert any(r["system"] == "react" and float(r["GPU-s/alert"]) == 12.0 for r in comp)
    assert (val_run["out"] / "compute_by_machine.csv").exists()
    sh = {r["answers"]: float(r["brier gain over p0 (cross-fitted)"]) for r in csv.DictReader((val_run["out"] / "shuffled_answers.csv").open())}
    assert sh["real"] > 0.01 > sh["shuffled across alerts"]  # the real answers help, permuted answers do not
    cal = list(csv.DictReader((val_run["out"] / "fusion_calibration.csv").open()))
    assert cal[0]["score"].startswith("p0")


# ---- figures: optional matplotlib, CSV always --------------------------------------------------------------------------
def test_figures_always_have_csv_and_png_only_with_matplotlib(tmp_path):
    env = build_run(tmp_path, with_injection=False)
    out = tmp_path / "o1"
    script.main(["--split", "val", "--run", str(env["run_dir"]), "--inputs", str(env["inputs"]), "--outdir", str(out), "--n-boot", "300"])
    figs = out / "figures"
    for name in ("privacy_curve", "pareto", "external_gap", "reliability"):
        assert (figs / f"{name}.csv").exists()
    pytest.importorskip("matplotlib")
    assert (figs / "privacy_curve.png").stat().st_size > 1000 and (figs / "pareto.png").exists()


def test_figures_degrade_without_matplotlib(tmp_path, monkeypatch):
    env = build_run(tmp_path, with_injection=False)
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    out = tmp_path / "o2"
    assert script.main(["--split", "val", "--run", str(env["run_dir"]), "--inputs", str(env["inputs"]), "--outdir", str(out), "--n-boot", "300"]) == 0
    assert (out / "figures" / "privacy_curve.csv").exists() and not (out / "figures" / "privacy_curve.png").exists()
    st_ = json.loads((out / "status.json").read_text())
    assert "matplotlib is not installed" in st_["privacy_curve"]["png_note"]


# ---- numbers.json and the report fill-in -----------------------------------------------------------------------------------
def test_every_outline_id_is_produced_or_marked_elsewhere(val_run):
    nums = json.loads((val_run["out"] / "numbers.json").read_text())
    ids = ft.outline_ids(ROOT / "docs" / "report" / "00_outline.md")
    assert len(ids) >= 30
    for i in ids:
        assert i in nums and nums[i]["status"] != "unmapped", i
    assert {nums[i]["status"] for i in ids} <= {"ok", "not_run", "elsewhere"}
    assert nums["gap_closed_gfp"]["status"] == "ok" and "[" in nums["gap_closed_gfp"]["text"]
    assert nums["abstract"]["status"] == "elsewhere" and nums["three_way"]["status"] == "elsewhere"
    assert nums["tab_5a"]["text"].startswith("| group")
    assert nums["fig_privacy_curve"]["status"] == "ok"
    assert nums["hypothesis_verdicts"]["text"].count("\n") >= 10


def test_fill_replaces_placeholders_in_copies_and_lists_the_rest(val_run, tmp_path):
    rep = tmp_path / "report"
    rep.mkdir()
    (rep / "07_results.md").write_text("Gap {{TBD:gap_closed_gfp}} twice {{TBD:gap_closed_gfp}}.\n\n{{TBD:tab_5a}}\n\n{{TBD:abstract}} {{TBD:made_up_id}} {{TBD:fast_model}}\n\n"
                                       "{{TBD:fig_privacy_curve}}\n")  # fmt: skip
    (rep / "00_outline.md").write_text("{{TBD:gap_closed_gfp}} stays\n")
    before = (rep / "07_results.md").read_text()
    out = tmp_path / "filled"
    nums = json.loads((val_run["out"] / "numbers.json").read_text())
    r = ft.fill_report(rep, nums, out, link_base=val_run["out"])
    assert (rep / "07_results.md").read_text() == before  # never in place
    text = (out / "07_results.md").read_text()
    assert "{{TBD:gap_closed_gfp}}" not in text and text.count(nums["gap_closed_gfp"]["text"]) == 2
    assert "| group" in text and "{{TBD:tab_5a}}" not in text
    unfilled = {u["id"]: u["reason"] for u in r["unfilled"]}
    assert "abstract" in unfilled and "made_up_id" in unfilled and "numbers.json" in unfilled["made_up_id"]
    assert "{{TBD:abstract}}" in text and "{{TBD:made_up_id}}" in text
    assert "{{TBD:gap_closed_gfp}}" in (out / "00_outline.md").read_text()  # the outline documents the placeholders
    with pytest.raises(fn.AnalysisInputError):
        ft.fill_report(rep, nums, rep)


def test_fill_option_of_the_script(val_run, tmp_path, capsys):
    rep = tmp_path / "report"
    rep.mkdir()
    (rep / "a.md").write_text("{{TBD:h2_ratio}} {{TBD:abstract}}\n")
    out = tmp_path / "filled"
    rc = script.main(["--fill", str(rep), "--numbers", str(val_run["out"] / "numbers.json"), "--out", str(out)])
    assert rc == 0
    cap = capsys.readouterr().out
    assert "1 placeholders could not be filled" in cap and "TBD:abstract" in cap
    assert "{{TBD:h2_ratio}}" not in (out / "a.md").read_text()
    assert json.loads((out / "unfilled.json").read_text())[0]["id"] == "abstract"


def test_outputs_exist(val_run):
    for f in ("hypotheses.md", "hypotheses.csv", "hypotheses.json", "numbers.json", "status.json", "fixed_on_val.json", "table_5a.md", "table_5b.csv",
              "privacy_curve.md", "n_alerts.md", "table_baselines.md", "h4_fdr_audit.csv", "power.txt"):
        assert (val_run["out"] / f).exists(), f
    # e18 outputs were not given: reported, not silently skipped
    assert json.loads((val_run["out"] / "status.json").read_text())["e18_sweep"]["status"] == "not_run"


def test_build_alert_table_join():
    alerts = pl.DataFrame({"alert_id": ["a", "b"], "tx_id": [1, 2], "bank": [10, 11], "p0": [0.1, 0.2], "label": [0, 1]})
    tx = pl.DataFrame({"tx_id": [1, 2, 3], "src_id": [5, 6, 7], "dst_id": [8, 9, 4]})
    pat = pl.DataFrame({"tx_id": [2], "attempt_id": ["A1"]})
    df = fn.join_alert_table(alerts, tx, pat)
    assert df["src_id"].to_list() == [5, 6] and df["attempt_id"].to_list() == [None, "A1"] and df["home_bank"].to_list() == [10, 11]


def test_copy_of_run_is_not_needed():
    assert shutil is not None
