"""T4: the prereg freeze also covers T2's non-LLM arm settings (configs/final/non_llm.yaml, configs/experiments/ablations_nonllm.yaml)."""

import os
import subprocess

from ftm.final import freeze as fz
from ftm.final import preflight as pf


def _git(repo, *args):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.check_output(["git", *args], cwd=repo, text=True, env=env, stderr=subprocess.STDOUT).strip()


def _repo(tmp_path, tables="data/backtest/x/lr_table.jsonl"):
    repo = tmp_path / "repo"
    (repo / "prereg").mkdir(parents=True)
    (repo / "configs/final").mkdir(parents=True)
    (repo / "configs/experiments").mkdir(parents=True)
    (repo / "prereg/hypotheses.yaml").write_text("status: draft\n")
    (repo / "configs/final/non_llm.yaml").write_text(f"data: hi_medium\ntables: {{realistic: {tables}}}\n")
    (repo / "configs/experiments/ablations_nonllm.yaml").write_text("arms: {eig: {}}\n")
    (repo / "configs/final/top_external.json").write_text('{"systems": ["a", "b"], "selected_on": "val"}\n')
    for name in fz.EXTRA_FROZEN:  # every other frozen file exists too (content is irrelevant here)
        if not (repo / name).exists():
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_text("{}\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "c")
    return repo


def test_extra_files_are_in_the_hash_and_a_change_breaks_it(tmp_path):
    repo = _repo(tmp_path)
    pre = repo / "prereg"
    names = fz.frozen_names(pre)
    assert "configs/final/non_llm.yaml" in names and "configs/experiments/ablations_nonllm.yaml" in names and "hypotheses.yaml" in names
    h = fz.combined_hash(fz.hash_files(pre))
    (repo / "configs/experiments/ablations_nonllm.yaml").write_text("arms: {eig: {budget: 4}}\n")
    assert fz.combined_hash(fz.hash_files(pre)) != h


def test_missing_extra_files_are_skipped_and_reported(tmp_path):
    repo = _repo(tmp_path)
    (repo / "configs/experiments/ablations_nonllm.yaml").unlink()
    assert "configs/experiments/ablations_nonllm.yaml" not in fz.frozen_names(repo / "prereg")
    assert fz.missing_extra(repo / "prereg") == ["configs/experiments/ablations_nonllm.yaml"]


def test_preflight_checks_the_extra_files_at_the_freeze_commit(tmp_path):
    repo = _repo(tmp_path)
    pre = repo / "prereg"
    fr = fz.write_freeze(repo, pre)
    fz.record_ta_hash(pre, fr["combined_sha256"])
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "freeze")
    assert "configs/final/non_llm.yaml" in fr["files"]
    paths = pf.Paths(repo=repo, prereg=pre, e18_dir=tmp_path, e18_config=tmp_path / "e", llm_config=tmp_path / "l", top_external=tmp_path / "t")
    assert pf.check_hashes_match(paths) == [] and pf.check_commit_ancestor(paths) == []  # also reads the extra files at the freeze commit
    # a change after the freeze is caught
    (repo / "configs/final/non_llm.yaml").write_text("data: hi_medium\ntables: {realistic: other.jsonl}\n")
    problems = pf.check_hashes_match(paths)
    assert problems and any("non_llm.yaml" in x for x in problems)


def test_unresolved_when_the_question_table_is_not_named(tmp_path):
    repo = _repo(tmp_path, tables="null")
    assert pf.unresolved_fields(repo / "prereg") == ["configs/final/non_llm.yaml:tables.realistic"]
    repo2 = _repo(tmp_path / "b")
    assert pf.unresolved_fields(repo2 / "prereg") == []


def test_the_dspy_program_and_the_frozen_library_are_hashed_when_present_and_a_change_breaks_the_hash(tmp_path):
    repo = _repo(tmp_path)
    pre = repo / "prereg"
    from ftm.final import plan as fp

    lib = fp.load_jobs_config()["llm"]["common_args"]
    lib = lib[lib.index("--library") + 1]
    assert fz.EXTRA_UNTRACKED[0].endswith("react_miprov2.json") and fz.EXTRA_UNTRACKED[1:] == (lib,)  # the library the jobs load, read from jobs.yaml
    assert set(fz.missing_untracked(pre)) == set(fz.EXTRA_UNTRACKED)  # absent: reported, not hashed
    for name in fz.EXTRA_UNTRACKED:
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text('{"v": 1}')
    assert not fz.missing_untracked(pre) and set(fz.EXTRA_UNTRACKED) <= set(fz.frozen_names(pre))
    h = fz.combined_hash(fz.hash_files(pre))
    (repo / fz.EXTRA_UNTRACKED[1]).write_text('{"v": 2}')
    assert fz.combined_hash(fz.hash_files(pre)) != h


def test_the_fitted_pieces_of_the_final_run_are_frozen_and_exist_in_the_repo():
    """The LR table, fusion parameters, decision threshold, tuned profile and p0 map that were fixed before test are in the freeze (and tracked, so `git show` can check them)."""
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    want = {"configs/fusion/headroom_v1.json", "configs/fusion/headroom_v1_table.jsonl", "configs/decision/headroom_v1.yaml", "configs/llm/tuned.yaml", "configs/t1_final/p0_recal_hi_medium.json"}
    assert want <= set(fz.EXTRA_FROZEN)
    tracked = set(subprocess.check_output(["git", "ls-files", *fz.EXTRA_FROZEN], cwd=root, text=True).split())
    assert tracked == set(fz.EXTRA_FROZEN), sorted(set(fz.EXTRA_FROZEN) - tracked)
