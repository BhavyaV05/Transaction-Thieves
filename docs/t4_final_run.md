# The final run (G5): `scripts/run_final.py`, the prereg freeze, and what is still open (week 14, T4)

Owner: T4 (Shreyas). Specs: `plan.md` 7.2, 8, 13 (G5), `tracks.md` 6.7 and 8, `docs/schedule_nov10.md` (phase C), `docs/t1_final_run.md` (T1's commands). **Nothing here was run on the test split**: `FTM_FINAL_RUN` was never set outside a pytest monkeypatch and `scripts/run_final.py::main`; every test uses tmp dirs, a throw-away git repo and the synthetic world with the mock LLM.

## What the script does

`scripts/run_final.py` is the only place that sets `FTM_FINAL_RUN=1` (`tests/test_static_checks.py` fails if any other line of `ftm/` or `scripts/` writes it). It does nothing on real data until the preflight passes. Order of a real run:

1. **Preflight** (report only, nothing set or written; the checks are in `ftm/final/preflight.py`): `prereg/FREEZE.json` exists; the recomputed sha256 of every prereg file equals it; the working tree is clean (the one tolerated change is the `subset_sha256:` line of `prereg/subset_manifest.txt`); the freeze commit is an ancestor of HEAD and holds the frozen contents; the hash was recorded as sent to the TA; the E18 decision exists and `configs/llm/default.yaml` is the chosen backbone (and the thinking arm); no `fix_on_val` is left in `prereg/*.yaml`; `outputs/g4/top_external.json` (top 2-3 external agents, `selected_on: val`) is valid; the subset rule parses; every selected job and T1 step has a command; **T1's own check** `scripts/t1_final_check.py` passes (configs parse, scripts and flags exist, val artefacts exist; it refuses to run with the lock open, so it runs here, before the lock); `--confirm-final <combined hash>` equals the combined prereg hash. Any failure: exit 2, lock not set, nothing written.
2. The lock opens (`os.environ["FTM_FINAL_RUN"] = "1"`, inherited by every child process). The run dir `outputs/final/<run_id>/` is created; `run.json` pins the prereg hash and dataset (a different pair needs a new `--run-id`).
3. **T1 `pre` steps** (docs/t1_final_run.md rows 1-3: GFP one-bank, GFP see-everything, conformal intake `make_alerts.py --final`): they build `data/alerts/<ds>/test.parquet` (with `test_tail`). A failure here stops the run (exit 3), no subset is drawn.
4. **The frozen 1,000-alert subset, before any other system runs** (`ftm.intake.alerts.stratified_subset(alerts.filter(~pl.col("test_tail")), n=1000, seed=0, strata="bank_p0_decile")`, rule read from `prereg/subset_manifest.txt`; decision 4 of `docs/t4_decisions_week2.md`: the headline is the bulk days). Written to `subset_ids.txt` and `full_test_ids.txt` (all test alerts, tail days included, reported separately) with their sha256 in `subset.json`; the sha256 of the sorted subset ids is also written to the `# subset_sha256:` line of `prereg/subset_manifest.txt` (that line is not part of the freeze hash; `rule_commit` and every other line stay untouched; a different hash already recorded is fatal). Commit that line after the run. A second machine redraws and must get the same hash (`--expect-subset-sha` makes that explicit); a job refuses to start if its id file no longer hashes to `subset.json`.
5. **T1 `post` steps** (rows 4-12: Multi-GNN, MEGA-GNN, edge scorer, explainers, evidence views, DP-FedAvg, layer-wise, FraudGT): each command is its own process, one at a time, stdout and stderr in `<run dir>/t1/<step>/`. A non-zero exit means the `--final` pass did not reproduce the validation numbers (T1's built-in check): the step fails, its dependants and the detection jobs are blocked, and its test scores must not be used.
6. **Jobs** (`ftm/final/plan.py`): our system and its ablations on the subset and the full test set; every external LLM agent on the subset, the G4 top 2-3 also on the full test set; Groups A/B detection run logs (T1) and T2's non-LLM arms as `external_cmd` jobs. LLM jobs call `scripts/run_llm_agent.py` (`--alert-ids`, `--blind`: no Brier and no label in any file or on the console, same code path as every validation run). Machine classes are fixed in `plan.json` at the first start. Jobs are resumable (a `done` marker holds the settings hash; an unfinished job is deleted and rerun; a finished job with other settings is moved to `_superseded/`, never overwritten).
7. **`MANIFEST.json`** when everything required is done, else `MANIFEST.partial.json` (parity checks between machines run either way: same alert set hash, seeds, limits, backbone, model revision, thinking, serving stack, commit).

```
MANIFEST.json  {manifest_version: 1, run_id, synthetic, complete, git_commit, git_commits, freeze_sha256, data,
                subset_sha256, full_test_sha256, n_subset, n_full_test, subset_ids_path, full_test_ids_path, missing,
                jobs: [{job_id, system, variant, alerts: subset|full_test, kind, mode, group, tier, seeds, machine_class,
                        machine, status, runlog_path, runlog_sha256, n_alerts, n_expected, model, git_commit}],
                t1_steps: [{step_id, row, phase, machine_class, required, status, n_commands, git_commit}]}
```

Run logs are run-log v1 parquet (`ftm/eval/runlog.py`), **test labels are not in them** (a run log that carries a label fails its job): the analysis joins the labels in. T1's own test scores stay where its `--final` modes write them (`data/scores/<ds>/final/`, read through the guarded `gfp_gbt.read_p0(..., split="test")`); items 4 to 6 of the T1 hand-off (evidence views through `ftm.retrieval.evidence_views.view`, the retriever table, H10) belong to the analysis and to T3's planner jobs, not to this script.

## Commands

```
python scripts/run_final.py --plan       # job list with machine class and hours, T1 steps, preflight report; sets nothing, reads no data
python scripts/run_final.py --dry-run    # the whole flow on the synthetic world: mock LLM, dev toolbox, FTM_FINAL_RUN untouched,
                                         #   writes outputs/final_synthetic/<run id>/ (MANIFEST has synthetic: true); T1 steps are listed, not run
python scripts/run_final.py --dry-run --only ours.compose --run-id try      # a slice
# the real run (after the 30 Oct freeze), same --run-id on every machine, resumable
python scripts/run_final.py --run-id g5 --confirm-final <combined hash> --machine s1
python scripts/run_final.py --run-id g5 --confirm-final <combined hash> --machine s3 --endpoint http://localhost:8000/v1
python scripts/run_final.py --run-id g5 --confirm-final <combined hash> --machine cpu
python scripts/run_final.py --run-id g5 --confirm-final <combined hash> --manifest-only    # after gathering the job dirs
```

Selection: `--machine s3|s1|cpu`, `--only <job id | system | system.variant | fnmatch | t1 | t1.<step>>`, `--shard i/n` (balanced by hours; jobs only), `--max-jobs`. `S1` is the 3070 Ti laptop (T1's steps, half-speed LLM share), `S3` the 5070 Ti (LLM host, FraudGT). Machine classes: LLM jobs are balanced between `gpu_s3` and `gpu_s1` by longest-job-first at the speeds in `configs/final/jobs.yaml` (`ours.compose.*` pinned to S3; set `machines.gpu_s1.enabled: false` if the backbone does not fit 8 GB).

Planned hours (`--plan`, estimates: 7.5 s per alert from `docs/t3_first_runs.md`, 1,000 subset and 3,000 full-test alerts until the real count exists, T1's own estimates from `docs/t1_final_run.md`): `gpu_s3` about 77 h (LLM jobs 68 h, FraudGT 6 h, HI-Medium see-everything 3 h), `gpu_s1` about 53 h (T1 steps 17.6 h, LLM jobs 36 h incl. the 0.1 h detection run-log jobs), `cpu` about 19 h (T2's non-LLM arms at the 0.5 h placeholder). The plan prints these; replace the placeholders with the validation-run medians before 31 Oct.

## 30 Oct freeze checklist

1. G4 review done: `outputs/g4/top_external.json` written and committed? (it is not in `prereg/`; the preflight reads it from `outputs/`, so write it on every machine that runs, or pass `--top-external`).
2. All `fix_on_val` values in `prereg/hypotheses.yaml` set from validation (the freeze script warns about each one that is left; the preflight refuses them). `rule_commit` in `prereg/subset_manifest.txt` is filled (PR #68, `84c15b7`); this script does not touch that line.
3. Merge everything that must be frozen; clean tree; `python scripts/freeze_prereg.py` **writes `prereg/FREEZE.json`** (sha256 of every `prereg/*.yaml` and of `subset_manifest.txt` without its `subset_sha256:` line, the commit, the combined hash) and prints the combined hash. It never freezes on its own: it runs only when you call it, refuses a dirty tree, and `--check` verifies without writing.
4. Send the combined hash to the TA; then `python scripts/freeze_prereg.py --record-ta-sent <hash>` (refuses a hash that differs from the files). Commit `prereg/FREEZE.json`.
5. After the freeze any edit to a prereg file changes the hash and the preflight refuses; a change to a primary hypothesis goes back to the TA.

## What is NOT done

- **No real run, no real alert counts.** The 3,000 / 1,000 alert counts are placeholders until T1's `make_alerts.py --final` has written the test set.
- **Commands owed by other tracks** (the preflight lists them): T1 detection run-log writer (`external_cmd.templates.detection`; writes a run-log v1 from `data/scores/<ds>/final/`), FraudGT final command (`t1.fraudgt`, to build, 5070 Ti, `T1_EVAL_TEST=1` is set by the step's `env`), T1's HI-Medium see-everything (no val run yet, optional step); T2's non-LLM arm runner (`external_cmd.templates.non_llm_arm`); T3/T4 `llm.common_args` (back-test table, fusion params, frozen library) and the real s-per-alert values.
- **The analysis script** that reads `MANIFEST.json`, joins the labels and computes the tables is a later piece; the run script never prints a label or a Brier score.
- **Rehearsal on a copy where "test" is val** (T1's suggestion for the G5 rehearsal) is not automated: `--dry-run` uses synthetic alerts and the mock LLM and runs none of T1's `--final` commands, so those are exercised only by T1's toy-data tests and `t1_final_check.py`.
- `scripts/t1_final_check.py --help` probing needs the T1 environment (torch); `--plan` skips it, a real run does it.
- The `stratify` refactor of `ftm/intake/alerts.py` also fixes the stratum order (`group_by` order was not deterministic in polars): the same alerts now always give the same ids. The rule (strata, allocation, seed) is unchanged; this is the one behavioural difference from the code commit quoted in `configs/t1_final/subset_rule.yaml`.

## Files

`scripts/run_final.py` (CLI, ordering), `scripts/freeze_prereg.py`, `ftm/final/{freeze,preflight,plan,t1,runner}.py`, `configs/final/jobs.yaml` (job settings, T1 steps, the slots owners fill), `tests/test_run_final.py`. Hooks in other tracks' files: see the PR description.
