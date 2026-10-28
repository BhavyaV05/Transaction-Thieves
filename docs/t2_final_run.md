# T2 in the final run (G5): the non-LLM arms and the raw-sharing arm (weeks 12-13, T2)

Owner: T2 (Bhavya). Specs: `docs/schedule_nov10.md` (phase C: "all non-LLM arms, and the banks for every LLM arm (CPU)"), `docs/t4_final_run.md` (the job contract), `ftm/orchestrator/README.md` (the arms). **Nothing here was run on the test split or on real data**: every number in the tests is the synthetic world, and `FTM_FINAL_RUN` is never set by T2's code.

## What T2 owes the final run

| Piece | Where | Status |
|---|---|---|
| The non-LLM arms (19) and the raw-sharing arm, on the frozen ids | `scripts/final_non_llm.py`, `ftm/orchestrator/final_nonllm.py` | written, tested on the synthetic world, run through T4's `run_final.py --dry-run` |
| The banks for every LLM arm (CPU) | `ftm/harness` through `scripts/run_llm_agent.py --toolbox harness` (T3/T4 call it) | nothing new: it is the code the validation runs use |
| Question tables the arms read | `configs/final/non_llm.yaml` | realistic set; **enumerator and exact-label still empty** (hours of back-testing each) |
| Job plan entries | `configs/final/jobs.yaml` (`external_cmd.templates.non_llm_arm`, `extra_jobs`) | filled in; hour estimates are guesses |

## One configuration, two runs

An arm is built by `ftm.orchestrator.ablations.prepare_arm`, the function the G4 table (`scripts/run_ablations.py`) builds it with, from `configs/experiments/ablations_nonllm.yaml`. G5 adds no second description of an arm, so what is run on test is what was validated on val:

| | G4 (validation) | G5 (`final_non_llm.py`) |
|---|---|---|
| scored on | the `eval` half of val | the frozen ids of the job (`{alert_ids}`: the 1,000-alert subset or the full test set) |
| fitted on | the `fit` half of val (fusions, learned policy) | the same `fit` half of val (`split_val`, the same cut) |
| question tables | the back-test tables | the same files, **not rebuilt, not re-accepted** |
| bank noise | `seed` | the same `seed`, banks over the test ledgers |
| labels | in the rows, Brier in the summary | in no row and no summary (`run_alerts(..., blind=True)`) |

`tests/test_final_non_llm.py::test_the_final_runner_reproduces_the_g4_arms_on_the_same_alerts` pins this on the synthetic world: on the same alerts the final runner's probabilities give the G4 row's Brier to 1e-12 for random, EIG, the eps arms, the cap, the waves, 20 % declines and the per-family fusion.

## Decisions the code makes (change them in the config or tell me)

1. **Fits stay on validation.** The per-family / shared fusion and the learned policy are fitted on the `fit` half of val, exactly as in G4, not on all of val. A fit on all of val would use more labels but would not be the configuration the G4 table validated. (The learned policy can instead be a fitted file: `policy:` in `configs/final/non_llm.yaml`.)
2. **An alert with no modelled partner bank gets a verdict row with no probability** (`stop_reason` = `no_partner_bank`, `p0` from the alert table), so every arm covers exactly the ids in the file and the analysis scores those rows as p0, like any no-verdict. `load_world` drops such alerts silently; here they are counted (`summary.json`).
3. **The per-account cap is per job.** The accountant is built fresh for each job, so the arms with a cap (`eig_cap_2`, `eig_cap_8`) spend the cap over the alerts of that job only: the subset job sees 1,000 alerts, the full-test job all of them. Alerts are processed in time order.
4. **Raw sharing is fitted on train (under review, see the section above: the base-rate shift makes it lose to p0 on val).** `raw_share` fits its ridge model on the train alerts through the train banks, then applies it to the frozen ids through the test banks. It is the only arm that asks the gate for `request_records` (`allow_raw_share`, this arm only).
5. **Run-log names.** `system` = the planner (`random`, `fixed`, `eig`, `learned`, `enumerator`), `variant` = the arm name; raw sharing is `raw_share` / `raw_share`. These are the names `configs/analysis/systems.yaml` already selects on (its raw-share role was marked ASSUMED; it now matches).

## Before the final run (G4, 30 Oct)

```
# 1. the back-test tables the G4 table is run on (real data, train alerts; `--eps` = the `eps_levels` of the ablations YAML; flags: scripts/backtest.py --help, ftm/backtest/README.md)
python scripts/backtest.py --data hi_medium --eps 0.25 0.5 1 2 --questions expert                  # then --assemble; realistic table
python scripts/backtest.py --data hi_medium --eps 0.25 0.5 1 2 --questions enumerate --max-size 3  # then --assemble; enumerator table
python scripts/backtest.py --data hi_medium --exact-labels                                      # then --assemble; exact-label table
#    each run writes its cache under data/backtest/hi_medium/<namespace>/; `--assemble` (same flags) writes lr_table.jsonl next to it

# 2. the G4 table itself
python scripts/run_ablations.py --data hi_medium --split val --table <realistic> --table-enumerator <enum> --table-exact-labels <exact> --out outputs/g4

# 3. put the same three paths into configs/final/non_llm.yaml (tables:), then
python scripts/final_non_llm.py --check            # exit 1 and the list of what is missing
```

`--check` reads no data. The tables, the config and the arm settings are part of what the pre-registration freezes only through the commit; if `prereg/` should hash them, T4 adds `configs/final/non_llm.yaml` and `configs/experiments/ablations_nonllm.yaml` to the freeze list (not done here: `prereg/` is T4's).

## Rehearsal on validation (recommended before 31 Oct)

The same command runs on validation alerts, with a val id file, through exactly the code path of the final run:

```
python scripts/final_non_llm.py --arm eig --split val --alert-ids <val eval-half ids> --out outputs/rehearsal/eig --data hi_medium
```

The rows must give the Brier of the `eig` row of `outputs/g4/ablations.json` when scored against the val labels (the equivalence is unit-tested on the synthetic world, not yet on real data). That check is the cheap way to find a data problem (a missing ledger, a changed table, an id file that is not the eval half) before the lock is open.

Dry run of the whole plan on the synthetic world (about 15 min for all 40 T2 jobs, each a fresh synthetic world): `python scripts/run_final.py --dry-run --synthetic 30 --subset-n 10 --only 'nonllm.*' 'raw_share.*'`. A slice of it (3 jobs) was run for this PR: all three finished, validated, and entered the partial manifest.

## Commands

```
python scripts/final_non_llm.py --list                                   # the 20 arm names (19 + raw_share)
python scripts/final_non_llm.py --check                                  # config and tables, no data
python scripts/final_non_llm.py --arm <arm> --alert-ids <file> --out <dir> [--data hi_medium] [--seed 0]   # what run_final.py calls
```

`run_final.py` fills `{variant}`, `{alert_ids}`, `{out}`, `{data}`, `{seed}`, `{run_id}`, `{commit}`, `{machine}`. Output: `<out>/runlog.parquet` (run-log v1; `label` empty) and `<out>/summary.json` (counts, settings, config hash, ids hash; no Brier). The job fails if a row carries a label (`ftm/final/runner.py::validate_runlog`).

## Hours (guesses until the val runs give medians)

The 19 planner arms are quick once the worlds are loaded (a few minutes per 1,000 alerts in the synthetic world); loading the test world (ledgers of the modelled banks) and, for the three fitted arms, the val world dominate. `configs/final/jobs.yaml` has 0.5 h per arm, 1.0 h for the enumerator, the learned policy and the two fusions, and 1.5 h for raw sharing (it loads train and test): about 20 h of CPU over the 40 jobs, in `--plan`. Replace them with the rehearsal times.

## Raw sharing on real validation: why it loses to p0 (28 Oct)

Rehearsal on the 230 `eval` alerts: Brier 0.2241 against p0 0.1926 (the arm as specified, fitted on the 1,983 train alerts). It is not a code error (no feature is constant on train and non-constant on val, nothing is non-finite, the rehearsal rows are well formed). It is a **base-rate shift**: the launder rate is 0.187 on train, 0.452 on val (0.361 on the `eval` half), and 0.108 in the first 70% of train in time against 0.370 in the last 30%, the same ledger warm-up T4 found (`docs/t4_fusion_findings.md` section 7). A model fitted on train learns "lower p0" (`a` = 0.21, intercept -1.5) and then under-predicts on val. Recalibrating p0 alone, with no raw feature at all, scores 0.2213 on `eval`, so the raw features add nothing to the loss; the loss is the recalibration.

The shared-record features do carry signal: on a random 70/30 split inside train the model scores 0.093 against p0 0.199, and forward in time inside train (first 70% fits, last 30% scores) 0.165 against 0.174. Fitted on the val `fit` half and scored on the `eval` half (230 alerts each, ridge chosen by looking at `eval`, so descriptive only): 0.160 at ridge 50, 0.176 at 500, 0.192 at 5 against p0 0.1926. So the arm is a valid upper limit only if it is fitted in the same period as the alerts it is scored on, which is also what the fused question arms do (decision 1). Recommended change, not made yet because it moves the H2 reference: fit raw sharing on the val `fit` half through the val banks (about a minute, instead of loading and featurising train, 35 minutes), choose the ridge by time-ordered cross-validation inside that half, and report the train-fitted number beside it as the shifted variant.

## Not done

- **Status of the real data (28 Oct):** the realistic table is built and set in `configs/final/non_llm.yaml` (`7df6bad5337d`, 23 of 440 accepted); G4 ran on real val for 17 of 19 arms; the final-run command was rehearsed on val for 5 arms and reproduces the G4 Brier exactly (`ftm/orchestrator/README.md`, "Real validation run"). Still missing: the **enumerator** table (470 programs up to size 3 over both slots is about 1,900 hypotheses, several hours) and the **exact-label** table (T1: hours, set questions over 8,000 alerts), so those two arms are skipped and `final_non_llm.py --check` still fails on them; the hour estimates in `jobs.yaml` are still guesses.
- The eps = infinity (no-noise) EIG arm of `scripts/eps_sweep.py` is not a G5 job: `configs/analysis/systems.yaml` reads only the 0.5 / 1 / 2 / 4 points for the EIG curve and eps = inf for raw sharing.
- Hash-freezing `configs/final/non_llm.yaml` (T4's `prereg/` list).

## Also in this PR (outside T2's files, small)

- `ftm/final/runner.py::command_for` quotes the fields before `shlex.split`. On Windows, `{alert_ids}` and `{out}` lost their backslashes (`C:\Users\...` became `C:Users...`) and every `external_cmd` job failed; Linux is unchanged. Pinned by `tests/test_final_non_llm.py::test_command_templates_keep_windows_paths_intact`.
- `tests/test_run_final.py` has 14 tests that already fail on a Windows machine with `core.autocrlf=true` (the freeze test compares file bytes at a commit, the others build command templates from `sys.executable`). They fail the same way on main without this PR; CI (Linux) is not affected.
