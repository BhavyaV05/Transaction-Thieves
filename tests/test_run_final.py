"""T4: the G5 final-run script: freeze, preflight, job plan, resumable jobs, subset hash, parity, a mock end-to-end run.

Everything runs on tmp dirs, a throw-away git repo and the synthetic world with the mock LLM. The real prereg files are only
read (the plan test). FTM_FINAL_RUN is never left set: the fixture resets it to a value that is undone after each test.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
import yaml

from ftm import guards
from ftm.final import freeze as fz
from ftm.final import plan as fp
from ftm.final import preflight as pf
from ftm.final import runner as rn

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run_final = load_script("run_final")
freeze_script = load_script("freeze_prereg")


@pytest.fixture(autouse=True)
def lock_closed(monkeypatch):
    """Whatever a test (or main()) sets, the environment is restored afterwards."""
    monkeypatch.setenv(guards.FINAL_RUN_ENV, "0")
    yield
    # monkeypatch undoes the setenv after this point; nothing else may leak


# -- a throw-away repo with a complete, valid freeze ---------------------------------------------------------
EXTERNAL = """
status: draft
systems:
  - {group: A, name: multi_gnn, modes: [one_bank], tier: core, owner: T1, impl: x}
  - {group: C, name: torres, modes: [native, ported], tier: core, owner: T4, impl: x}
  - {group: D, name: react, modes: [ported], tier: core, owner: T3, impl: x}
  - {group: D, name: rewoo, modes: [ported], tier: core, owner: T3, impl: x}
"""
SUBSET_RULE = """# rule: stratified, ftm.intake.alerts.stratified_subset, n=40, seed=0, strata="bank_p0_decile"
# subset_sha256: (filled at G5)
"""
E18_CFG = {"candidates": [{"name": "qwen3-8b", "hf_id": "Qwen/Qwen3-8B-AWQ", "stack": "vllm"},
                          {"name": "gpt-oss-20b", "hf_id": "openai/gpt-oss-20b", "stack": "vllm"}],
           "serving": {"qwen3-8b": {"served_model": "qwen3-8b", "revision": "abc123"}},
           "gpt_oss_thinking_on_effort": "medium", "stage2": {"thinking_arm": {"max_output_tokens_per_step": 1024}}}
LLM_CFG = {"backbone": "Qwen/Qwen3-8B", "thinking": False, "limits": {"max_llm_calls": 20, "max_tokens": 6000}}


def git(repo, *args):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.check_output(["git", *args], cwd=repo, text=True, env=env, stderr=subprocess.STDOUT).strip()


def commit_all(repo, msg="c"):
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def world(tmp_path):
    repo = tmp_path / "repo"
    (repo / "prereg").mkdir(parents=True)
    git(repo, "init", "-q")
    pre = repo / "prereg"
    (pre / "external_list.yaml").write_text(EXTERNAL)
    (pre / "hypotheses.yaml").write_text("status: draft\nstatistics: {equivalence_margin: {value: 0.01}}\n")
    (pre / "e18_decision.yaml").write_text("status: committed\n")
    (pre / "subset_manifest.txt").write_text(SUBSET_RULE)
    (repo / ".gitignore").write_text("outputs/\n")
    commit_all(repo, "prereg")
    fr = fz.write_freeze(repo, pre)
    fz.record_ta_hash(pre, fr["combined_sha256"])
    commit_all(repo, "freeze")
    e18 = tmp_path / "e18"
    (e18 / "stage2").mkdir(parents=True)
    (e18 / "stage2" / "decision.json").write_text(json.dumps({"chosen": "qwen3-8b"}))
    cfgs = tmp_path / "cfg"
    cfgs.mkdir()
    (cfgs / "e18.yaml").write_text(yaml.safe_dump(E18_CFG))
    (cfgs / "llm.yaml").write_text(yaml.safe_dump(LLM_CFG))
    top = tmp_path / "top_external.json"
    top.write_text(json.dumps({"systems": ["react", "torres"], "selected_on": "val"}))
    paths = pf.Paths(repo=repo, prereg=pre, e18_dir=e18, e18_config=cfgs / "e18.yaml", llm_config=cfgs / "llm.yaml", top_external=top)
    return paths, fr["combined_sha256"]


def failures(paths, confirm, check=None):
    rep = pf.run_preflight(paths, confirm, {"react", "rewoo", "torres"})
    return {k: v for k, v in rep.items() if v} if check is None else rep[check]


# -- freeze ----------------------------------------------------------------------------------------------------
def test_combined_hash_is_content_only_and_ignores_the_subset_hash_line(world):
    paths, h = world
    assert fz.combined_hash(fz.hash_files(paths.prereg)) == h
    (paths.prereg / "subset_manifest.txt").write_text(SUBSET_RULE.replace("(filled at G5)", "abc123  # run g5"))
    assert fz.combined_hash(fz.hash_files(paths.prereg)) == h  # filling the line after the run keeps the freeze valid
    (paths.prereg / "subset_manifest.txt").write_text(SUBSET_RULE.replace("n=40", "n=41"))
    assert fz.combined_hash(fz.hash_files(paths.prereg)) != h


def test_write_freeze_refuses_a_dirty_tree_but_not_its_own_file(world):
    paths, _ = world
    fz.write_freeze(paths.repo, paths.prereg)  # FREEZE.json is committed and unchanged: fine, and rewriting it is fine
    (paths.prereg / "hypotheses.yaml").write_text("status: edited\n")
    with pytest.raises(RuntimeError, match="not clean"):
        fz.write_freeze(paths.repo, paths.prereg)


def test_record_ta_hash_must_equal_the_files(world):
    paths, h = world
    with pytest.raises(RuntimeError, match="does not match"):
        fz.record_ta_hash(paths.prereg, "0" * 64)
    assert fz.record_ta_hash(paths.prereg, h)["ta_ack_hash"] == h


def test_freeze_script_prints_the_hash_and_check_mode(world, capsys):
    paths, h = world
    args = ["--repo", str(paths.repo), "--prereg-dir", str(paths.prereg)]
    assert freeze_script.main([*args, "--check"]) == 0
    assert h in capsys.readouterr().out
    assert freeze_script.main(args) == 0
    out = capsys.readouterr().out
    assert "COMBINED HASH" in out and h in out
    (paths.prereg / "external_list.yaml").write_text(EXTERNAL + "# edited\n")
    assert freeze_script.main([*args, "--check"]) == 1


# -- preflight: passes when everything holds, fails for each missing or mismatched condition ----------------------
def test_preflight_passes_when_all_hold(world):
    paths, h = world
    assert failures(paths, h) == {}


def test_preflight_fails_without_a_freeze_manifest(world):
    paths, h = world
    (paths.prereg / fz.FREEZE_FILE).unlink()
    assert "FREEZE.json" in " ".join(pf.check_freeze_manifest(paths))
    assert "freeze manifest" in failures(paths, h)


def test_preflight_fails_when_a_frozen_file_changed(world):
    paths, h = world
    (paths.prereg / "hypotheses.yaml").write_text("status: draft\nstatistics: {equivalence_margin: {value: 0.02}}\n")
    assert any("hypotheses.yaml" in x for x in pf.check_hashes_match(paths))
    assert pf.check_tree_clean(paths)  # and the tree is dirty too
    assert pf.check_confirm(paths, h)  # the old hash no longer equals the files


def test_preflight_fails_when_a_file_was_added_after_the_freeze(world):
    paths, _ = world
    (paths.prereg / "extra.yaml").write_text("a: 1\n")
    assert any("extra.yaml" in x for x in pf.check_freeze_manifest(paths) + pf.check_hashes_match(paths))


def test_preflight_fails_on_a_dirty_tree(world):
    paths, _ = world
    assert pf.check_tree_clean(paths) == []
    (paths.repo / "scratch.txt").write_text("x")
    assert any("not clean" in x for x in pf.check_tree_clean(paths))


def test_preflight_fails_when_the_freeze_commit_is_not_in_history(world):
    paths, _ = world
    base = git(paths.repo, "rev-parse", "HEAD")
    git(paths.repo, "checkout", "-q", "-b", "other")
    (paths.repo / "o.txt").write_text("o")
    other = commit_all(paths.repo, "other")
    git(paths.repo, "checkout", "-q", base)
    fr = fz.read_freeze(paths.prereg)
    fr["git_commit"] = other
    (paths.prereg / fz.FREEZE_FILE).write_text(json.dumps(fr))
    commit_all(paths.repo, "bad freeze")
    assert any("not an ancestor" in x for x in pf.check_commit_ancestor(paths))
    fr["git_commit"] = "f" * 40
    (paths.prereg / fz.FREEZE_FILE).write_text(json.dumps(fr))
    commit_all(paths.repo, "unknown commit")
    assert any("not in this repository" in x for x in pf.check_commit_ancestor(paths))


def test_preflight_fails_when_the_freeze_commit_holds_other_contents(world):
    paths, _ = world
    fr = fz.read_freeze(paths.prereg)
    fr["files"]["hypotheses.yaml"] = "0" * 64  # the commit does not hold what the manifest claims
    (paths.prereg / fz.FREEZE_FILE).write_text(json.dumps(fr))
    commit_all(paths.repo, "tampered")
    assert any("does not match FREEZE.json" in x for x in pf.check_commit_ancestor(paths))


def test_preflight_fails_when_the_ta_hash_is_missing_or_differs(world):
    paths, _ = world
    fr = fz.read_freeze(paths.prereg)
    fr["ta_ack_hash"] = None
    (paths.prereg / fz.FREEZE_FILE).write_text(json.dumps(fr))
    assert any("ta_ack_hash is empty" in x for x in pf.check_ta_ack(paths))
    fr["ta_ack_hash"] = "1" * 64
    (paths.prereg / fz.FREEZE_FILE).write_text(json.dumps(fr))
    assert any("differs" in x for x in pf.check_ta_ack(paths))


def test_preflight_fails_without_the_e18_decision_or_with_another_backbone(world):
    paths, _ = world
    assert pf.check_e18_decision(paths) == []
    LLM = yaml.safe_load(paths.llm_config.read_text())
    paths.llm_config.write_text(yaml.safe_dump({**LLM, "backbone": "openai/gpt-oss-20b"}))
    assert any("not the E18 choice" in x for x in pf.check_e18_decision(paths))
    paths.llm_config.write_text(yaml.safe_dump({**LLM, "backbone": "Qwen/Qwen3-8B-AWQ"}))
    assert pf.check_e18_decision(paths) == []  # same backbone, quantised id
    (paths.e18_dir / "stage2" / "decision.json").write_text(json.dumps({"chosen": None, "reason": "none qualifies"}))
    assert any("no 'chosen'" in x for x in pf.check_e18_decision(paths))
    (paths.e18_dir / "stage2" / "decision.json").unlink()
    assert any("not found" in x for x in pf.check_e18_decision(paths))


def test_preflight_checks_the_thinking_arm_decision(world):
    paths, _ = world
    (paths.e18_dir / "thinking").mkdir()
    (paths.e18_dir / "thinking" / "decision.json").write_text(json.dumps({"adopt": True}))
    assert any("thinking" in x for x in pf.check_e18_decision(paths))  # config says thinking: false
    (paths.e18_dir / "thinking" / "decision.json").write_text(json.dumps({"adopt": False}))
    assert pf.check_e18_decision(paths) == []


def test_preflight_lists_every_unresolved_fix_on_val(world):
    paths, _ = world
    (paths.prereg / "hypotheses.yaml").write_text(
        "statistics:\n  equivalence_margin: {value: fix_on_val}\n  clusters_fallback: {hub_cap: fix_on_val}\n"
        "hypotheses:\n  H9: {margin_gap: fix_on_val, threshold: 0.3}\n  H1: {tier: [primary, fix_on_val]}\n")
    left = pf.unresolved_fields(paths.prereg)
    assert sorted(left) == ["hypotheses.yaml:hypotheses.H1.tier[1]", "hypotheses.yaml:hypotheses.H9.margin_gap",
                            "hypotheses.yaml:statistics.clusters_fallback.hub_cap",
                            "hypotheses.yaml:statistics.equivalence_margin.value"]
    assert len(pf.check_unresolved(paths)) == 4


def test_preflight_fails_without_or_with_a_wrong_confirm_hash(world):
    paths, h = world
    assert pf.check_confirm(paths, None) and pf.check_confirm(paths, h[:-1] + ("0" if h[-1] != "0" else "1"))
    assert pf.check_confirm(paths, h[:12])  # a prefix is not enough
    assert pf.check_confirm(paths, h) == []


def test_top_external_file_is_validated(world, tmp_path):
    paths, _ = world
    known = {"react", "rewoo", "torres"}
    f = paths.top_external
    assert pf.load_top_external(f, known) == (["react", "torres"], [])
    for bad, msg in [({"systems": ["react"], "selected_on": "val"}, "2 or 3"),
                     ({"systems": ["react", "rewoo", "torres", "x"], "selected_on": "val"}, "2 or 3"),
                     ({"systems": ["react", "react"], "selected_on": "val"}, "duplicate"),
                     ({"systems": ["react", "torres"], "selected_on": "test"}, "selected_on"),
                     ({"systems": ["react", "multi_gnn"], "selected_on": "val"}, "not external agents"),
                     ({"systems": "react"}, "list of system names")]:
        f.write_text(json.dumps(bad))
        assert any(msg in x for x in pf.load_top_external(f, known)[1]), bad
    f.write_text("{not json")
    assert any("does not parse" in x for x in pf.load_top_external(f, known)[1])
    f.unlink()
    assert any("not found" in x for x in pf.load_top_external(f, known)[1])


def test_subset_rule_is_read_from_the_manifest(world):
    paths, _ = world
    assert pf.parse_subset_rule(paths.prereg / "subset_manifest.txt")[0] == {"n": 40, "seed": 0, "strata": "bank_p0_decile"}
    (paths.prereg / "subset_manifest.txt").write_text("# only prose\n")
    assert pf.check_subset_rule(paths)
    real = pf.parse_subset_rule(ROOT / "prereg" / "subset_manifest.txt")
    assert real == ({"n": 1000, "seed": 0, "strata": "bank_p0_decile"}, [])


def test_job_commands_and_test_alert_source_checks(tmp_path):
    jobs = [fp.Job("a.b.subset", "a", "b", "external_cmd", "subset", [0], "cpu", 1, owner="T1"),
            fp.Job("a.b.full_test", "a", "b", "external_cmd", "full_test", [0], "cpu", 1, owner="T1"),
            fp.Job("c.d.subset", "c", "d", "external_cmd", "subset", [0], "cpu", 1, owner="T2", cmd="echo"),
            fp.Job("o.compose.subset", "o", "compose", "llm_agent", "subset", [0], "gpu_s3", 1)]
    assert pf.check_job_commands(jobs) == ["T1: no command in configs/final/jobs.yaml for 1 jobs: a.b"]
    assert pf.check_test_alert_source(tmp_path / "missing.parquet", None)
    assert pf.check_test_alert_source(tmp_path / "missing.parquet", "python build.py") == []


# -- the lock -------------------------------------------------------------------------------------------------
def write_test_alerts(path, n=300):
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"alert_id": [f"E{i:05d}" for i in range(n)], "bank": [i % 3 for i in range(n)],
                  "p0": [((i * 37) % n) / n for i in range(n)], "label": [i % 2 for i in range(n)]}).write_parquet(path)


WRITER = """
import argparse, os, sys
import polars as pl
ap = argparse.ArgumentParser()
ap.add_argument("--alert-ids"); ap.add_argument("--out"); ap.add_argument("--system", default="x"); ap.add_argument("--fail", default="")
a = ap.parse_args()
if a.fail and os.path.exists(a.fail):
    sys.exit(3)
assert not os.environ.get("WRITER_EXPECT_LOCK") or os.environ.get("FTM_FINAL_RUN") == "1", "the command must run with the lock open"
ids = [i for i in open(a.alert_ids).read().split() if i]
label = [1] * len(ids) if os.environ.get("WRITER_LEAK") else [None] * len(ids)
pl.DataFrame({"run_id": ["r"] * len(ids), "system": [a.system] * len(ids), "variant": ["v"] * len(ids), "seed": [0] * len(ids),
              "split": ["test"] * len(ids), "alert_id": ids, "event": ["verdict"] * len(ids), "probability": [0.3] * len(ids),
              "p0": [0.2] * len(ids), "label": label}, schema_overrides={"label": pl.Int64}).write_parquet(os.path.join(a.out, "runlog.parquet"))
print("brier=0.123")  # an owner's command may print scores: the runner must not show them
"""


def write_jobs_config(tmp_path, alerts_path, **over):
    writer = tmp_path / "writer.py"
    writer.write_text(WRITER)
    cmd = f"{sys.executable} {writer} --alert-ids {{alert_ids}} --out {{out}} --system {{system}}"
    cfg = {"data": "fake", "test_alerts": {"path": str(alerts_path), "cmd": None},
           "external_cmd": {"templates": {"detection": cmd}, "commands": {}},
           "ours": {"variants": ["compose"]}, **over}
    f = tmp_path / "jobs.yaml"
    f.write_text(yaml.safe_dump(cfg))
    return f


def final_args(world, tmp_path, jobs_cfg, *extra):
    paths, _ = world
    return ["--repo", str(paths.repo), "--prereg-dir", str(paths.prereg), "--e18-dir", str(paths.e18_dir),
            "--e18-config", str(paths.e18_config), "--llm-config", str(paths.llm_config),
            "--top-external", str(paths.top_external), "--jobs-config", str(jobs_cfg), "--out-root", str(tmp_path / "out"), *extra]


def test_lock_is_set_only_after_the_preflight_passes(world, tmp_path, capsys, monkeypatch):
    paths, h = world
    alerts = tmp_path / "test.parquet"
    write_test_alerts(alerts)
    jobs_cfg = write_jobs_config(tmp_path, alerts)
    monkeypatch.setenv("WRITER_EXPECT_LOCK", "1")  # the command asserts that it runs with the lock open
    args = final_args(world, tmp_path, jobs_cfg, "--run-id", "g5", "--only", "multi_gnn.one_bank.subset")
    # wrong hash: refused, lock stays closed, nothing written
    assert run_final.main([*args, "--confirm-final", "0" * 64]) == run_final.EXIT_REFUSED
    assert os.environ[guards.FINAL_RUN_ENV] == "0" and not (tmp_path / "out").exists()
    assert "REFUSED" in capsys.readouterr().out
    # missing run id
    assert run_final.main([*final_args(world, tmp_path, jobs_cfg, "--only", "multi_gnn.one_bank.subset"), "--confirm-final", h]) == run_final.EXIT_REFUSED
    assert os.environ[guards.FINAL_RUN_ENV] == "0"
    # everything holds: the lock opens inside main(), the owner's command sees it, the labels column is not read
    assert run_final.main([*args, "--confirm-final", h]) == run_final.EXIT_OK
    assert os.environ[guards.FINAL_RUN_ENV] == "1"
    run = tmp_path / "out" / "g5"
    info = json.loads((run / "subset.json").read_text())
    assert info["n_subset"] == 40 and info["n_full_test"] == 300 and info["rule"]["strata"] == "bank_p0_decile"
    assert (run / "multi_gnn.one_bank.subset" / "done").exists()
    out = capsys.readouterr().out
    assert "subset_sha256: " + info["subset_sha256"] in out and "brier" not in out.lower()


def test_dry_run_never_sets_the_lock_or_writes_anything(world, tmp_path, capsys):
    paths, h = world
    alerts = tmp_path / "test.parquet"
    write_test_alerts(alerts)
    jobs_cfg = write_jobs_config(tmp_path, alerts)
    for extra in ([], ["--confirm-final", h], ["--confirm-final", "bad"]):
        code = run_final.main([*final_args(world, tmp_path, jobs_cfg, "--dry-run"), *extra])
        assert os.environ[guards.FINAL_RUN_ENV] == "0", "a dry run must never open the lock"
        assert not (tmp_path / "out").exists()
        out = capsys.readouterr().out
        assert "DRY RUN" in out and "job plan" in out and "ours.compose.subset" in out
        assert code == run_final.EXIT_FAILED  # the tmp world has no commands for the other jobs; the report says so
    # a dry run reports problems instead of raising
    (paths.e18_dir / "stage2" / "decision.json").unlink()
    assert run_final.main([*final_args(world, tmp_path, jobs_cfg, "--dry-run")]) == run_final.EXIT_FAILED
    assert "E18 decision" in capsys.readouterr().out and os.environ[guards.FINAL_RUN_ENV] == "0"


def test_dry_run_on_the_real_prereg_lists_the_open_problems(capsys):
    assert run_final.main(["--dry-run"]) in (run_final.EXIT_OK, run_final.EXIT_FAILED)
    out = capsys.readouterr().out
    assert "DRY RUN" in out and os.environ[guards.FINAL_RUN_ENV] == "0"
    assert "ours.compose.full_test" in out


# -- job plan --------------------------------------------------------------------------------------------------
def real_plan(top=None, **cfg_over):
    ext = yaml.safe_load((ROOT / "prereg" / "external_list.yaml").read_text())
    abl = yaml.safe_load(fp.ABLATIONS.read_text())
    cfg = fp.load_jobs_config()
    for k, v in cfg_over.items():
        cfg[k] = v
    return fp.build_plan(ext, cfg, abl, top), ext


def test_plan_has_the_core_systems_ours_on_both_sets_externals_on_the_subset():
    jobs, ext = real_plan()
    by = {j.job_id: j for j in jobs}
    assert len(by) == len(jobs)
    core = {s["name"] for s in ext["systems"] if s["tier"] == "core"}
    assert core <= {j.system for j in jobs}
    for v in ("compose", "fixed_list", "act_only", "no_backtest", "llm_decides"):
        assert {f"ours.{v}.subset", f"ours.{v}.full_test"} <= set(by)
    for s in ext["systems"]:
        if s["group"] in ("C", "D"):
            for m in s["modes"]:
                assert f"{s['name']}.{m}.subset" in by and f"{s['name']}.{m}.full_test" not in by  # no G4 choice yet
        else:
            for m in s["modes"]:
                assert {f"{s['name']}.{m}.subset", f"{s['name']}.{m}.full_test"} <= set(by)  # detection: E6/E7 on the full test
    assert all(j.kind == "external_cmd" for j in jobs if j.group in ("A", "B", "nonllm"))
    arms = [a["name"] for a in yaml.safe_load(fp.ABLATIONS.read_text())["arms"]]
    assert all(f"nonllm.{a}.{s}" in by for a in arms for s in ("subset", "full_test"))
    assert all(j.machine_class in fp.MACHINE_CLASSES for j in jobs) and all(j.est_hours > 0 for j in jobs)


def test_top_external_agents_also_run_on_the_full_test_set():
    jobs, _ = real_plan(top=["torres", "react"])
    ids = {j.job_id for j in jobs}
    assert {"torres.native.full_test", "torres.ported.full_test", "react.ported.full_test"} <= ids
    assert "sharma.native.full_test" not in ids and "reflexion.ported.full_test" not in ids


def test_every_llm_job_names_a_system_run_llm_agent_knows():
    agent = run_final.agent_module()
    known = {"planner", *agent.GROUP_C, *agent.GROUP_D}
    jobs, _ = real_plan(top=["torres", "react"])
    assert {j.llm_system for j in jobs if j.kind == "llm_agent"} <= known


def test_machine_assignment_balances_and_pins():
    jobs, _ = real_plan()
    llm = [j for j in jobs if j.kind == "llm_agent"]
    assert {j.machine_class for j in llm} == {"gpu_s3", "gpu_s1"}
    assert all(j.machine_class == "gpu_s3" for j in jobs if j.job_id.startswith("ours.compose."))  # reference machine
    load = {c: sum(j.est_hours for j in llm if j.machine_class == c) for c in ("gpu_s3", "gpu_s1")}
    assert load["gpu_s3"] / load["gpu_s1"] == pytest.approx(2.0, rel=0.25)  # S1 runs at half speed
    cfg = fp.load_jobs_config()
    cfg["machines"]["gpu_s1"]["enabled"] = False
    ext = yaml.safe_load((ROOT / "prereg" / "external_list.yaml").read_text())
    off = fp.build_plan(ext, cfg, None, None)
    assert {j.machine_class for j in off if j.kind == "llm_agent"} == {"gpu_s3"}


def test_selection_filters_and_shards_cover_every_job_exactly_once():
    jobs, _ = real_plan(top=["react", "torres"])
    s3 = fp.select_jobs(jobs, machine="s3")
    assert s3 and all(j.machine_class == "gpu_s3" for j in s3)
    assert {j.job_id for j in fp.select_jobs(jobs, only=["react"])} == {"react.ported.subset", "react.ported.full_test"}
    assert {j.job_id for j in fp.select_jobs(jobs, only=["ours.compose.*"])} == {"ours.compose.subset", "ours.compose.full_test"}
    assert len(fp.select_jobs(jobs, max_jobs=3)) == 3
    with pytest.raises(ValueError, match="matches no job"):
        fp.select_jobs(jobs, only=["nope"])
    for pool in (jobs, s3, fp.select_jobs(jobs, machine="cpu")):
        for n in (1, 2, 3, 5):
            seen = []
            for i in range(1, n + 1):
                seen += [j.job_id for j in fp.select_jobs(pool, shard=f"{i}/{n}")]
            assert sorted(seen) == sorted(j.job_id for j in pool) and len(seen) == len(set(seen)), (n, len(pool))
    sh = fp.shard_of(s3, 2)
    h = [sum(j.est_hours for j in s3 if sh[j.job_id] == k) for k in (1, 2)]
    assert abs(h[0] - h[1]) <= max(j.est_hours for j in s3)  # balanced to within one job
    for bad in ("0/2", "3/2", "x", "1-2"):
        with pytest.raises(ValueError, match="--shard"):
            fp.parse_shard(bad)


def test_order_jobs_follows_dependencies_and_rejects_cycles():
    a = fp.Job("a", "a", "v", "external_cmd", "subset", [0], "cpu", 1, depends_on=["b"])
    b = fp.Job("b", "b", "v", "external_cmd", "subset", [0], "cpu", 1)
    assert [j.job_id for j in fp.order_jobs([a, b])] == ["b", "a"]
    b.depends_on = ["a"]
    with pytest.raises(ValueError, match="cycle"):
        fp.order_jobs([a, b])
    a.depends_on = ["zzz"]
    b.depends_on = []
    with pytest.raises(ValueError, match="unknown job"):
        fp.order_jobs([a, b])


def test_extra_jobs_and_commands_come_from_the_config():
    cfg = fp.load_jobs_config()
    cfg["extra_jobs"] = [{"system": "gfp", "variant": "one_bank", "alerts": ["subset"], "cmd": "echo hi", "owner": "T1"}]
    cfg["external_cmd"]["commands"] = {"multi_gnn.see_everything": "echo multi"}
    ext = yaml.safe_load((ROOT / "prereg" / "external_list.yaml").read_text())
    by = {j.job_id: j for j in fp.build_plan(ext, cfg, None, None)}
    assert by["gfp.one_bank.subset"].cmd == "echo hi" and "gfp.one_bank.full_test" not in by
    assert by["multi_gnn.see_everything.subset"].cmd == "echo multi" == by["multi_gnn.see_everything.full_test"].cmd


# -- subset, jobs, resume ---------------------------------------------------------------------------------------
def make_ctx(tmp_path, n=60, cmd_job=None, **kw):
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    alerts = pl.DataFrame({"alert_id": [f"E{i:05d}" for i in range(n)], "bank": [i % 3 for i in range(n)],
                           "p0": [((i * 37) % n) / n for i in range(n)]})
    from ftm.intake import alerts as intake
    info = rn.build_subset(run_dir, alerts, {"n": 20, "seed": 0, "strata": "bank_p0_decile"}, intake.stratify, data="fake", commit="c0")
    return rn.Ctx(run_dir=run_dir, repo=ROOT, run_id="t", subset=info, commit="c0", machine_name="m", freeze_sha256="f" * 8,
                  n_alerts={"subset": info["n_subset"], "full_test": info["n_full_test"]}, **kw), alerts


def cmd_job(tmp_path, name="a", alerts="subset", extra=""):
    writer = tmp_path / "writer.py"
    writer.write_text(WRITER)
    return fp.Job(f"{name}.v.{alerts}", name, "v", "external_cmd", alerts, [0], "cpu", 0.1,
                  cmd=f"{sys.executable} {writer} --alert-ids {{alert_ids}} --out {{out}} --system {{system}} {extra}")


def test_subset_is_drawn_by_the_rule_and_hashed(tmp_path):
    ctx, alerts = make_ctx(tmp_path)
    info = ctx.subset
    ids = rn.read_ids(ctx.run_dir / "subset_ids.txt")
    assert info["n_subset"] == len(ids) == 20 and ids == sorted(ids)
    assert info["subset_sha256"] == rn.ids_sha256(ids) and info["n_full_test"] == 60
    assert info["full_test_sha256"] == rn.ids_sha256(alerts["alert_id"].to_list())
    assert "subset_sha256: " + info["subset_sha256"] in rn.manifest_line(info, "g5")
    # the same draw again (a second machine, a resume) is accepted and returns the same record
    from ftm.intake import alerts as intake
    again = rn.build_subset(ctx.run_dir, alerts, {"n": 20, "seed": 0, "strata": "bank_p0_decile"}, intake.stratify, data="fake", commit="c1")
    assert again["subset_sha256"] == info["subset_sha256"]


def test_a_different_draw_or_test_set_is_refused(tmp_path):
    ctx, alerts = make_ctx(tmp_path)
    from ftm.intake import alerts as intake
    with pytest.raises(rn.SubsetMismatch, match="rule"):
        rn.build_subset(ctx.run_dir, alerts, {"n": 20, "seed": 1, "strata": "bank_p0_decile"}, intake.stratify, data="fake", commit="c")
    with pytest.raises(rn.SubsetMismatch, match="full_test_sha256|subset_sha256"):
        rn.build_subset(ctx.run_dir, alerts.head(50), {"n": 20, "seed": 0, "strata": "bank_p0_decile"}, intake.stratify, data="fake", commit="c")


def test_a_job_refuses_to_start_if_the_subset_hash_differs(tmp_path):
    ctx, _ = make_ctx(tmp_path)
    job = cmd_job(tmp_path)
    rn.verify_alert_set(ctx.run_dir, ctx.subset, "subset")
    ids = rn.read_ids(ctx.run_dir / "subset_ids.txt")
    (ctx.run_dir / "subset_ids.txt").write_text("\n".join(ids[:-1]) + "\n")  # one alert lost
    with pytest.raises(rn.SubsetMismatch, match="will not start"):
        rn.run_job(job, ctx)
    assert not (ctx.run_dir / job.job_id).exists()
    (ctx.run_dir / "full_test_ids.txt").write_text("E00001\n")
    with pytest.raises(rn.SubsetMismatch):
        rn.run_job(cmd_job(tmp_path, alerts="full_test"), ctx)


def test_job_runs_once_then_is_skipped_and_writes_the_files(tmp_path, capsys):
    ctx, _ = make_ctx(tmp_path)
    job = cmd_job(tmp_path)
    assert rn.run_job(job, ctx) == "done"
    d = ctx.run_dir / job.job_id
    assert (d / "done").exists() and (d / "runlog.parquet").exists()
    meta = json.loads((d / "job.json").read_text())
    assert meta["status"] == "done" and meta["git_commit"] == "c0" and meta["runlog_sha256"] == rn.file_sha256(d / "runlog.parquet")
    assert meta["counts"]["n_alerts_with_verdict"] == 20 and meta["settings"]["alerts_set_sha256"] == ctx.subset["subset_sha256"]
    out = capsys.readouterr().out
    assert "0.123" not in out and "brier" not in out.lower()  # the command's own output stays in the job dir
    assert "brier=0.123" in (d / "stdout-s0.log").read_text()
    first = (d / "runlog.parquet").stat().st_mtime_ns
    assert rn.run_job(job, ctx) == "skipped"
    assert (d / "runlog.parquet").stat().st_mtime_ns == first


def test_incomplete_job_is_deleted_and_rerun(tmp_path):
    ctx, _ = make_ctx(tmp_path)
    job = cmd_job(tmp_path)
    d = ctx.run_dir / job.job_id
    d.mkdir()
    (d / "stale.txt").write_text("from a crashed run")
    (d / "runlog.parquet").write_text("half written")  # no done marker
    assert rn.job_state(ctx.run_dir, job, rn.settings_of(job, ctx)) == "incomplete"
    assert rn.run_job(job, ctx) == "done"
    assert not (d / "stale.txt").exists() and rn.job_state(ctx.run_dir, job, rn.settings_of(job, ctx)) == "done"


def test_finished_job_with_other_settings_is_archived_not_overwritten(tmp_path):
    ctx, _ = make_ctx(tmp_path)
    job = cmd_job(tmp_path)
    assert rn.run_job(job, ctx) == "done"
    job2 = cmd_job(tmp_path, extra="--fail /nonexistent")
    assert rn.job_state(ctx.run_dir, job2, rn.settings_of(job2, ctx)) == "changed"
    assert rn.run_job(job2, ctx) == "done"
    assert len(list((ctx.run_dir / "_superseded").iterdir())) == 1


def test_failed_job_leaves_no_done_marker_and_a_rerun_succeeds(tmp_path):
    ctx, _ = make_ctx(tmp_path)
    flag = tmp_path / "fail.flag"
    flag.write_text("x")
    job = cmd_job(tmp_path, extra=f"--fail {flag}")
    assert rn.run_job(job, ctx) == "failed"
    d = ctx.run_dir / job.job_id
    assert not (d / "done").exists() and json.loads((d / "job.json").read_text())["status"] == "failed"
    assert "exited with 3" in (ctx.run_dir / "failures.log").read_text()
    flag.unlink()
    assert rn.run_job(job, ctx) == "done"  # the failed directory is incomplete: deleted and rerun


def test_a_job_without_a_command_or_with_bad_output_fails_cleanly(tmp_path, monkeypatch):
    ctx, _ = make_ctx(tmp_path)
    nocmd = fp.Job("n.v.subset", "n", "v", "external_cmd", "subset", [0], "cpu", 1)
    assert rn.run_job(nocmd, ctx) == "failed" and "no command" in (ctx.run_dir / "failures.log").read_text()
    monkeypatch.setenv("WRITER_LEAK", "1")  # the command writes labels into its run log
    leak = cmd_job(tmp_path, name="leak")
    assert rn.run_job(leak, ctx) == "failed"
    assert "never write test labels" in (ctx.run_dir / "failures.log").read_text()


def test_validate_runlog_rejects_stray_ids_and_duplicates(tmp_path):
    base = {"run_id": "r", "system": "s", "variant": "v", "seed": 0, "split": "test", "event": "verdict", "probability": 0.5, "p0": 0.1}
    f = tmp_path / "runlog.parquet"
    pl.DataFrame([{**base, "alert_id": "A"}, {**base, "alert_id": "Z"}]).write_parquet(f)
    with pytest.raises(rn.JobFailed, match="not in the frozen set"):
        rn.validate_runlog(f, {"A", "B"})
    pl.DataFrame([{**base, "alert_id": "A"}, {**base, "alert_id": "A"}]).write_parquet(f)
    with pytest.raises(rn.JobFailed, match="more than one verdict"):
        rn.validate_runlog(f, {"A", "B"})
    pl.DataFrame([{**base, "alert_id": "A", "probability": None}]).write_parquet(f)
    assert rn.validate_runlog(f, {"A", "B"})["no_verdict"] == 1


def test_parity_check_compares_alert_set_limits_backbone_thinking_and_commit():
    def meta(alerts="subset", kind="llm_agent", commit="c", **p):
        parity = {"alerts_set_sha256": "s", "alerts_hash": "h", "seeds": [0], "limits": {"max_tokens": 6000}, "model": "m@r",
                  "thinking": False, "backbone": "b", "serving_stack": "vllm", "max_output_tokens": None, **p}
        return {"job": {"alerts": alerts, "kind": kind}, "parity": parity, "git_commit": commit}

    ok = {"a": meta(), "b": meta(), "c": meta(alerts="full_test", alerts_hash="other", alerts_set_sha256="t")}
    assert rn.check_parity(ok) == ["c"]
    for key, val in (("alerts_hash", "x"), ("alerts_set_sha256", "y"), ("seeds", [1])):
        with pytest.raises(rn.ParityError, match=key):
            rn.check_parity({"a": meta(), "b": meta(**{key: val})})
    for key, val in (("limits", {"max_tokens": 1}), ("model", "m@other"), ("thinking", True), ("backbone", "x")):
        with pytest.raises(rn.ParityError, match=key):
            rn.check_parity({"a": meta(), "b": meta(**{key: val})})
    ext = meta(kind="external_cmd", alerts_hash=None, limits=None, model=None)
    assert rn.check_parity({"a": meta(), "b": ext})  # external jobs do not enter the LLM-only checks
    with pytest.raises(rn.ParityError, match="different commits"):
        rn.check_parity({"a": meta(), "b": meta(commit="d")})
    assert rn.check_parity({"a": meta(), "b": meta(commit="d")}, allow_mixed_commits=True) == ["c", "d"]


# -- the mock end to end run -------------------------------------------------------------------------------------
def test_mock_run_end_to_end_writes_a_manifest(tmp_path, capsys):
    ext_dir = tmp_path / "prereg"
    ext_dir.mkdir()
    (ext_dir / "external_list.yaml").write_text(EXTERNAL)
    jobs_cfg = write_jobs_config(tmp_path, tmp_path / "unused.parquet", extra_jobs=[], external_cmd={"templates": {"non_llm_arm": None}, "commands": {}})
    cfg = yaml.safe_load(jobs_cfg.read_text())
    writer = tmp_path / "writer.py"
    cfg["extra_jobs"] = [{"system": "simple", "variant": "arm", "cmd": f"{sys.executable} {writer} --alert-ids {{alert_ids}} --out {{out}} --system {{system}}"}]
    jobs_cfg.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "out"
    args = ["--synthetic", "24", "--subset-n", "8", "--run-id", "mock", "--out-root", str(out), "--jobs-config", str(jobs_cfg),
            "--prereg-dir", str(ext_dir), "--llm-config", str(ROOT / "configs" / "llm" / "default.yaml")]
    assert run_final.main(args) == run_final.EXIT_OK
    assert os.environ[guards.FINAL_RUN_ENV] == "0", "the rehearsal must not open the lock"
    run = out / "mock"
    man = json.loads((run / "MANIFEST.json").read_text())
    assert man["synthetic"] is True and man["complete"] and man["missing"] == []
    ids = {j["job_id"] for j in man["jobs"]}
    assert ids == {"ours.compose.subset", "ours.compose.full_test", "react.ported.subset", "rewoo.ported.subset",
                   "torres.native.subset", "torres.ported.subset", "simple.arm.subset", "simple.arm.full_test",
                   }  # multi_gnn (T1) has no command here, so the rehearsal leaves it out
    for j in man["jobs"]:
        f = run / j["runlog_path"]
        assert f.exists() and rn.file_sha256(f) == j["runlog_sha256"] and j["status"] == "done"
        df = pl.read_parquet(f)
        assert df["label"].null_count() == df.height  # no label in any run log
        verdicts = df.filter(pl.col("event") == "verdict")
        want = man["n_subset"] if j["alerts"] == "subset" else man["n_full_test"]
        assert verdicts.height == want == j["n_alerts"]
        assert set(verdicts["alert_id"]) <= set(rn.read_ids(run / (man["subset_ids_path"] if j["alerts"] == "subset" else man["full_test_ids_path"])))
    # the LLM jobs went through run_llm_agent in blind mode: no Brier, no label, same alert hash everywhere
    s = json.loads((run / "ours.compose.subset" / "seed-0" / "summary.json").read_text())
    assert s["blind"] is True and s["brier"] is None and s["alerts"] == "synthetic" and s["variant"] == "devtools"
    assert all(json.loads(ln)["label"] is None for ln in (run / "ours.compose.subset" / "seed-0" / "per_alert.jsonl").read_text().splitlines())
    printed = capsys.readouterr().out.lower()
    assert "brier" not in printed and "0.123" not in printed
    # a second invocation skips everything and rewrites the same manifest
    assert run_final.main(args) == run_final.EXIT_OK
    again = capsys.readouterr().out
    assert again.count("skipped (done)") == len(ids) and "done (" not in again.replace("skipped (done)", "")
    assert json.loads((run / "MANIFEST.json").read_text())["jobs"] == man["jobs"]


def test_partial_selection_writes_a_partial_manifest_and_shards_complete_it(tmp_path):
    ext_dir = tmp_path / "prereg"
    ext_dir.mkdir()
    (ext_dir / "external_list.yaml").write_text(EXTERNAL)
    jobs_cfg = write_jobs_config(tmp_path, tmp_path / "unused.parquet")
    base = ["--synthetic", "20", "--subset-n", "6", "--run-id", "shards", "--out-root", str(tmp_path / "out"),
            "--jobs-config", str(jobs_cfg), "--prereg-dir", str(ext_dir)]
    run = tmp_path / "out" / "shards"
    assert run_final.main([*base, "--shard", "1/2", "--only", "ours.compose.*", "react"]) == run_final.EXIT_OK
    partial = json.loads((run / "MANIFEST.partial.json").read_text())
    assert not partial["complete"] and partial["missing"] and not (run / "MANIFEST.json").exists()
    assert run_final.main([*base, "--shard", "2/2", "--only", "ours.compose.*", "react"]) == run_final.EXIT_OK
    for s in ("1/2", "2/2"):
        run_final.main([*base, "--shard", s])  # whatever is left of the whole plan
    assert (run / "MANIFEST.json").exists()
    assert run_final.main([*base, "--manifest-only"]) == run_final.EXIT_OK


def test_changed_plan_after_the_first_start_is_refused(tmp_path):
    ext_dir = tmp_path / "prereg"
    ext_dir.mkdir()
    (ext_dir / "external_list.yaml").write_text(EXTERNAL)
    jobs_cfg = write_jobs_config(tmp_path, tmp_path / "unused.parquet")
    base = ["--synthetic", "20", "--subset-n", "6", "--run-id", "chg", "--out-root", str(tmp_path / "out"),
            "--jobs-config", str(jobs_cfg), "--prereg-dir", str(ext_dir)]
    assert run_final.main([*base, "--only", "react"]) == run_final.EXIT_OK
    cfg = yaml.safe_load(jobs_cfg.read_text())
    cfg["ours"] = {"variants": ["compose", "fixed_list"]}
    jobs_cfg.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit, match="disagrees"):
        run_final.main([*base, "--only", "react"])


def test_run_llm_agent_blind_and_alert_ids_and_test_toolbox_lock(tmp_path, monkeypatch):
    agent = run_final.agent_module()
    ids = tmp_path / "ids.txt"
    ids.write_text("S000001\nS000003\nS000005\n")
    out = tmp_path / "run"
    assert agent.main(["--system", "planner", "--alerts", "synthetic", "--n", "8", "--llm", "mock", "--toolbox", "dev", "--seed", "0",
                       "--run-dir", str(out), "--run-id", "b", "--alert-ids", str(ids), "--blind", "--synthetic-labels"]) == 0
    s = json.loads((out / "summary.json").read_text())
    assert s["n_alerts"] == 3 and s["brier"] is None and s["blind"] is True
    rows = [json.loads(ln) for ln in (out / "per_alert.jsonl").read_text().splitlines()]
    assert sorted(r["alert_id"] for r in rows) == ["S000001", "S000003", "S000005"] and all(r["label"] is None for r in rows)
    monkeypatch.delenv(guards.FINAL_RUN_ENV, raising=False)
    with pytest.raises(guards.TestSplitLocked):
        agent.make_toolbox("harness", 0, "hi_small", "test")  # the test world only opens under the lock
    from ftm.backtest.world import load_world
    with pytest.raises(ValueError, match="locked"):
        load_world("hi_small", "test")


# -- LLM job arguments, settings, endpoint, dependencies ---------------------------------------------------------
def test_llm_argv_of_a_real_job_uses_the_test_alerts_the_frozen_ids_and_blind_mode(tmp_path):
    ctx, _ = make_ctx(tmp_path, endpoint="http://gpu:8000/v1", model="qwen3-8b", serving_stack="vllm", backbone="Qwen/Qwen3-8B",
                      limits={"max_tokens": 6000, "max_llm_calls": 20}, llm_common_args=["--fusion-params", "f.json"])
    job = fp.Job("ours.compose.subset", "ours", "compose", "llm_agent", "subset", [0, 1], "gpu_s3", 1, llm_system="planner",
                 llm_mode="compose", extra_args=["--batch-schedule", "2,1"])
    ids = ctx.run_dir / "subset_ids.txt"
    argv = rn.llm_argv(job, ctx, 1, tmp_path / "o" / "seed-1", ids)
    pairs = dict(zip(argv, argv[1:], strict=False))
    assert pairs["--alerts"] == "test" and pairs["--llm"] == "endpoint" and pairs["--toolbox"] == "harness"
    assert pairs["--alert-ids"] == str(ids) and "--blind" in argv and pairs["--seed"] == "1" and pairs["--mode"] == "compose"
    assert pairs["--endpoint"] == "http://gpu:8000/v1" and pairs["--model"] == "qwen3-8b" and pairs["--n"] == "60"
    assert pairs["--max-prompt-tokens"] == "6000" and pairs["--max-llm-calls"] == "20"
    assert argv[-4:] == ["--fusion-params", "f.json", "--batch-schedule", "2,1"] and "--thinking" not in argv
    ctx.thinking, ctx.max_output_tokens = True, 1024
    argv = rn.llm_argv(job, ctx, 0, tmp_path / "o", ids)
    assert "--thinking" in argv and argv[argv.index("--max-output-tokens") + 1] == "1024"
    ext = fp.Job("react.ported.subset", "react", "ported", "llm_agent", "subset", [0], "gpu_s3", 1, llm_system="react")
    assert "--mode" not in rn.llm_argv(ext, ctx, 0, tmp_path / "o", ids)  # --mode is the planner's


def test_llm_settings_follow_the_llm_config(world):
    paths, _ = world
    s = run_final.llm_settings(paths)
    assert s == {"backbone": "Qwen/Qwen3-8B", "limits": {"max_tokens": 6000, "max_llm_calls": 20}, "thinking": False,
                 "reasoning_effort": None, "max_output_tokens": None}
    paths.llm_config.write_text(yaml.safe_dump({**LLM_CFG, "thinking": True}))
    assert run_final.llm_settings(paths)["max_output_tokens"] == 1024
    paths.llm_config.write_text(yaml.safe_dump({**LLM_CFG, "backbone": "openai/gpt-oss-20b", "reasoning_effort": "low"}))
    s = run_final.llm_settings(paths)
    assert s["reasoning_effort"] == "low" and s["thinking"] is False


def test_check_endpoint_requires_the_chosen_model(monkeypatch):
    from ftm.llm import client as llm_client

    class Fake:
        base_url = "http://fake/v1"

        def __init__(self, models):
            self.models = models

        def get(self, path):
            if self.models is None:
                raise llm_client.LLMError("connection refused")
            return {"data": [{"id": m} for m in self.models]}

        def server_version(self):
            return "0.test"

    for models, want, ok in ((["qwen3-8b"], "qwen3-8b", True), (["other"], "qwen3-8b", False), (None, "qwen3-8b", False), (["a"], None, True)):
        monkeypatch.setattr(llm_client.HttpChatClient, "from_env", classmethod(lambda cls, models=models, **kw: Fake(models)))
        if ok:
            assert run_final.check_endpoint(None, want) == ("http://fake/v1", "0.test")
        else:
            with pytest.raises(SystemExit):
                run_final.check_endpoint(None, want)


def test_a_job_whose_dependency_failed_is_blocked_not_run(tmp_path, capsys):
    ext_dir = tmp_path / "prereg"
    ext_dir.mkdir()
    (ext_dir / "external_list.yaml").write_text(EXTERNAL)
    jobs_cfg = write_jobs_config(tmp_path, tmp_path / "unused.parquet")
    flag = tmp_path / "fail.flag"
    flag.write_text("x")
    writer = tmp_path / "writer.py"
    cmd = f"{sys.executable} {writer} --alert-ids {{alert_ids}} --out {{out}} --system {{system}}"
    cfg = yaml.safe_load(jobs_cfg.read_text())
    cfg["extra_jobs"] = [{"system": "first", "variant": "v", "alerts": ["subset"], "cmd": cmd + f" --fail {flag}"},
                         {"system": "second", "variant": "v", "alerts": ["subset"], "cmd": cmd, "depends_on": ["first.v.subset"]}]
    jobs_cfg.write_text(yaml.safe_dump(cfg))
    base = ["--synthetic", "20", "--subset-n", "6", "--run-id", "dep", "--out-root", str(tmp_path / "out"),
            "--jobs-config", str(jobs_cfg), "--prereg-dir", str(ext_dir), "--only", "first", "second"]
    assert run_final.main(base) == run_final.EXIT_FAILED
    out = capsys.readouterr().out
    assert "second.v.subset: blocked" in out and not (tmp_path / "out" / "dep" / "second.v.subset").exists()
    flag.unlink()
    assert run_final.main(base) == run_final.EXIT_OK  # the failed job reruns, then its dependant
    assert (tmp_path / "out" / "dep" / "second.v.subset" / "done").exists()
