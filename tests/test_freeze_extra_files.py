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
