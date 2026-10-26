"""A synthetic run directory in the shape of scripts/run_final.py's output, with systems of known skill.

Owner: T4 (Shreyas).  Used by tests/test_final_analysis.py and by `scripts/final_analysis.py --synthetic`, the dry run that checks every
table and figure of the final analysis before a test number exists.  No data, no LLM: a system's probability is the one-bank
log-odds shifted toward the truth by its planted `skill`, so every hypothesis verdict can be read off the skills.  Never a result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from ftm.eval import final as fn
from ftm.fusion.fuse import sigmoid
from ftm.harness.runlog_rows import to_frame

N = 900

# planted skills: log-odds shift toward the truth (0 = no better than one bank)
SKILL = {
    "ours_compose": 1.6, "ours_fixed_list": 0.5, "ours_no_backtest": -0.1, "ours_llm_decides": 0.8, "ours_act_only": 1.2,
    "random": 0.05, "fixed": 0.35, "eig": 0.5, "learned": 0.55, "enumerator": 0.6, "raw_share": 1.8,
    "eig_batched": 0.45,
}
EXTERNAL = {  # (system, mode) -> skill
    ("react", "ported"): 3.6, ("torres_ported", "ported"): 1.0, ("torres_native", "native"): 0.5, ("sharma_ported", "ported"): 0.3,
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
        jobs.append({"system": system, "variant": variant, "mode": mode, "alert_set": alert_set, "seeds": [0] if name.startswith("inj_") else list(seeds), "runlog": p.name, "sha256": sha})
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
    manifest = {"manifest_version": 1, "run_id": "run1", "synthetic": True, "split": split, "subset_sha256": fn.ids_sha256(w.subset), "jobs": jobs}
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
