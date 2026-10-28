# 11 Compute and reproducibility

*Owner: T4 (Shreyas); T3 supplies GPU-hours and serving details. Draft 1 (28 Oct 2026), written before the final run: it states the machines, the serving set-up, the planned hours (estimates, not measurements), the rules that make a run repeatable and the command behind each table. The measured hours are `{{TBD:compute_table}}`. Numbers from a script are `{{TBD:<id>}}` (convention in `docs/report/README.md`). Remove this line before the final build.*

This section says what the work ran on (11.1), how long it takes (11.2), what makes a run repeatable (11.3 and 11.4), how the run is logged (11.5), how each table is regenerated (11.6) and what may be released (11.7).

## 11.1 Machines and serving

Four people, four machines (`docs/hardware.md`). The **reference machine** is T3's PC: RTX 5070 Ti with 16 GB, Ryzen 9 7900X, 32 GB RAM, native Linux. It serves every LLM, runs the QLoRA training and the final run, and is the only machine whose timings count. T1's laptop (RTX 3070 Ti, 8 GB) runs the graph models, federated training and the CPU-bound feature jobs; T2's laptop (RTX 3050 Ti, 4 GB) runs back-testing and small-model tests; T4 has no GPU and works on CPU with the mock LLM and the shared servers. Brier and accuracy can come from any machine with the identical checkpoint and settings; timings only from the reference machine. A fixed benchmark (`scripts/bench_llm.py`) converts laptop hours to reference-hours; on the reference machine Qwen3-8B-AWQ under vLLM 0.31.0 decodes 117.4 tokens per second. One workload runs per GPU, so serving and training never share it.

Every serving machine uses the same vLLM version (0.31.0), the same checkpoint revision and the same flags (`configs/llm/serving.yaml`: seed 0, the client sends the temperature, prefix caching on, context 8,192, loopback bind, key in the environment and shared over the private network only). The pinned profiles are in Table 11a. The chosen planner backbone is Gemma 4 12B (E18, stage 2: Brier 0.1741 on 460 validation alerts, the best of the three that reached stage 2, `docs/t3_e18.md`). Its 10.3 GB of weights do not fit the 3070 Ti, so the final run is on the reference machine alone.

**Table 11a.** Serving profiles (full 40-character revisions and licences in `configs/llm/serving.yaml`).

| Profile | Checkpoint | Revision | Used for |
|---|---|---|---|
| `gemma4-12b` | `google/gemma-4-12B-it-qat-w4a16-ct` | `1d2c2d7` | E18 choice: planner, teacher, every external agent |
| `qwen3-8b` | `Qwen/Qwen3-8B-AWQ` | `4da05a8` | first runs, benchmark, E18 candidate |
| `qwen3-8b-bnb` | `Qwen/Qwen3-8B` (4-bit at load) | `b968826` | E16: the prompted arm and the three QLoRA students |
| `qwen3-4b`, `qwen3-4b-bnb` | `Qwen/Qwen3-4B-AWQ`, `Qwen/Qwen3-4B` | `74d4bd2`, `1cfa9a7` | fast model (System 1) and its adapters |
| `qwen3-14b` | `Qwen/Qwen3-14B-AWQ` | `31c69ef` | E18 candidate |
| `gpt-oss-20b`, `qwen3.8-27b-q3`, `gemma4-26b-a4b` | GGUF files via llama.cpp (revisions in `configs/experiments/e18_backbone_sweep.yaml`) | pinned there | E18 candidates; gpt-oss-20b does not load under vLLM beside the desktop |

## 11.2 Compute

The plan's budget is 250 reference-hours of GPU (`plan.md` section 11; one RTX 5070 Ti hour is one reference-hour). Logged by the team through 26 Oct: 14.8 reference-hours (`GPU_CALENDAR.md`), of which the backbone sweep was about 4.6 and the teacher runs 3.2; T1's HI-Medium see-everything Multi-GNN job (budget 8 h) started on 28 Oct. The final-run plan is the output of `python scripts/run_final.py --plan`, which reads no data and sets nothing. **Every hour in Table 11b is an estimate**: 7.5 s per alert for our planner (measured on Qwen3-8B, `docs/t3_first_runs.md`; the Gemma 4 12B median in E18 was 5.6 s), per-system seconds that are guesses for the external agents (6 s for ReWOO to 30 s for LATS, `configs/final/jobs.yaml`), 3,000 full-test and 1,000 subset alerts until the test alert set exists, T1's own step estimates (`docs/t1_final_run.md`) and 0.5 to 1.5 h per non-LLM arm as placeholders for T2's validation medians. All of them are replaced by measured values from the run logs (`gpu_seconds`, `llm_calls`, token counts), summed by machine in the analysis (`compute`, `compute_by_machine`).

**Table 11b.** Planned and measured hours of the final run (the plan of 28 Oct).

| Item | Machine | Planned h | Measured h |
|---|---|---|---|
| Our system, 5 variants (compose, fixed list, act only, no back-test, LLM decides), subset and full test | 5070 Ti | 41.7 | {{TBD:compute_table}} |
| Our system, 10 extra arms on the full test (ε 0.5, 1, 4, no noise; library by account; batched 2,1; three students; prompted Qwen3-8B) | 5070 Ti | 62.5 | {{TBD:compute_table}} |
| 13 external agents, native and ported where both exist, on the subset | 5070 Ti | 60.8 | {{TBD:compute_table}} |
| FraudGT scoring; HI-Medium see-everything (optional) | 5070 Ti | 0.5; 3.0 | {{TBD:compute_table}} |
| T1 steps 1 to 10 (GFP, Multi-GNN, MEGA-GNN, edge scorer, explainers, views, DP-FedAvg, layer-wise) and the 18 detection run-log jobs | 3070 Ti laptop | 19.1; 1.8 | {{TBD:compute_table}} |
| Non-LLM planners and arms (19) and raw sharing, subset and full test | CPU | 26.0 | {{TBD:compute_table}} |
| **Total** | | **5070 Ti 168.5; 3070 Ti 20.9; CPU 26.0** | {{TBD:compute_table}} |

The plan does not yet contain the full-test runs of the external agents chosen at G4, the E15 injection runs (30 to 40 hours, `docs/t4_injection.md`) or the QLoRA training. At 168.5 serial hours the final run takes seven days on one GPU, so starting on 30 Oct it ends on 6 Nov, the day of the full draft. The cut order (`plan.md` section 14) therefore applies: lower-priority arms move to the subset or are dropped, and the report lists what was cut (section 10). [CHECK: T3 to confirm the cumulative GPU-hours, including QLoRA and the E15 runs, against the 250-hour budget before the abstract quotes a total]

## 11.3 Seeds, caches and namespaces

**Seeds.** Every script that draws random numbers takes a seed that is written to its output. The final run uses seed 0 for every job at temperature 0 (`configs/final/jobs.yaml`); the trained arms use three (E16 students, DP-FedAvg and layer-wise exchange: 0, 1, 2); the cluster bootstrap uses 10,000 resamples with seed 0; the frozen subset uses seed 0. Aliases are drawn per run seed and the mapping is kept on the serving side only. A served LLM at temperature 0 is still not bit-deterministic; the benign-note twins of E15 measure that noise, and the run logs record the model revision so that a difference can be traced.

**Caches.** Large derived files are rebuilt from scripts and are not in git (`data/`, `outputs/` and `models/` are ignored); shared copies carry a sha256 manifest. The back-test cache lives in `data/backtest/<dataset>/<namespace>/`, keyed by the canonical form of a question and ε, where the namespace is a hash of the alert set and configuration (`7df6bad5337d` is the realistic HI-Medium table with 23 accepted questions, `4338ebb59b35` the HI-Small one). Each final job writes to its own directory under `outputs/final/<run_id>/`; a finished job with other settings is moved to `_superseded/`, never overwritten.

## 11.4 The test lock and the pre-registration

The test split cannot be read by accident. Every function that reads test data or test labels calls `require_final_run` (`ftm/guards.py`), which raises unless `FTM_FINAL_RUN=1`. That variable is set in exactly one place, `scripts/run_final.py::main`, and `tests/test_static_checks.py` fails if any other line under `ftm/` or `scripts/` assigns it. The script sets nothing until a preflight passes: `prereg/FREEZE.json` exists, the recomputed sha256 of every pre-registration file equals it, the tree is clean, the hash was recorded as sent to the TA, the E18 decision exists, no `fix_on_val` value is left and the G4 choice of external agents is valid. The run then draws the 1,000-alert subset before any system runs and writes its hash to `prereg/subset_manifest.txt`; a job refuses to start if its id file no longer matches. Nothing has been run on test: the preflight fails today on the missing freeze file, and every dry run used the synthetic world and the mock LLM.

## 11.5 Run-log schema v1

Every machine writes one Parquet table (`ftm/eval/runlog.py`, `RUN_LOG_VERSION` = 1) with one row per event, `step` (a question and its answer) or `verdict` (the end of an investigation). Version 1 keeps the v0 columns (run, commit, machine, system, variant, seed, split, alert, question, ε, gate decision, probability, tokens, calls, GPU-seconds) and adds the served model and revision, native or ported mode, p0, the stop reason, ε spent, simulated days, the question family, the log-likelihood ratio fusion used, the gate's reason for a refusal and the id of a planted injection. In the final run the `label` column stays empty: a log that carries a label fails its job, and the analysis joins labels from the alert table. `MANIFEST.json` lists every job with the sha256 of its log, the prereg hash and the subset hash, and the analysis checks them before computing anything.

## 11.6 Regenerating each table

**Table 11c.** The command behind each result (all from the repository root; `<run>` is `outputs/final/<run_id>`).

| Result | Command | Where |
|---|---|---|
| Final run, all jobs | `python scripts/run_final.py --run-id g5 --confirm-final <combined hash> --machine s3` (also `s1`, `cpu`; `--manifest-only` at the end) | 5070 Ti, laptop, CPU |
| Tables 5a and 5b, hypotheses, privacy curve, Pareto, breakdowns, error tags, case-study candidates | `FTM_FINAL_RUN=1 python scripts/final_analysis.py --split test --run <run> --inputs <inputs.yaml> --fixed <val>/fixed_on_val.json` | CPU |
| Numbers into the report | `python scripts/final_analysis.py --fill docs/report --numbers <run>/analysis/numbers.json --out <run>/analysis/report_filled` | CPU |
| Detection models and federated (section 7.1) | the `--final` modes of `train_gfp.py`, `train_gnn.py`, `train_federated.py`, `train_layerwise.py`, `make_alerts.py` (`docs/t1_final_run.md`); validation tables: `scripts/baselines_val_table.py` | laptop |
| Non-LLM arms and raw sharing | `python scripts/final_non_llm.py --arm <arm> --alert-ids <file> --out <dir>`; validation: `scripts/run_ablations.py`, tables from `scripts/backtest.py --assemble` (`docs/t2_final_run.md`) | CPU |
| Backbone sweep, fusion, G2 | `scripts/run_e18.py`; `scripts/g2_check.py`, `scripts/fusion_diagnostics.py` (`docs/t3_e18.md`, `docs/t4_fusion_findings.md`) | 5070 Ti, CPU |
| Prompt injection (E15) | `python scripts/run_injection_tests.py --system <list> --alerts val --val-part eval --arms <list>` (`docs/t4_injection.md`) | 5070 Ti |
| Gate and retrieval | `pytest tests/test_compliance_gate.py`; `scripts/rag_eval.py`; ceiling: `scripts/ceiling_check.py` | CPU |

## 11.7 Licences and availability

Released: the code, the simulator, the frozen alert sets and subset hash, the question language and the harness, so that other agents can be plugged in (`plan.md` section 17). Not released: trained weights, because a model that has learned which questions clear an account could help a criminal (section 8). The data is AMLworld [1], synthetic and distributed on Kaggle under CDLA-Sharing; the repository holds none of it, only the scripts that rebuild every derived file from the raw download with fixed seeds (`ftm/data/README.md`). The external code is used as its licence allows (`docs/external_code_check.md`): Sharma's MIT-licensed code is ported with its notice kept, Multi-GNN and MEGA-GNN are Apache-2.0, ReAct, Reflexion and ReWOO have MIT reference code, and systems with no licence or no code (Torres, Co-Investigator, FinFRE-RAG, BED-LLM, debate, Uncertainty of Thoughts) are reimplementations from the papers and labelled so. FraudGT and the code of Tian et al. have no licence file and are run from separate checkouts that are not part of this repository; IBM's snapml is proprietary and is used as a library only, never bundled (`docs/ta_addendum.md`). The repository itself has no licence file yet. [CHECK: the team to choose MIT or Apache-2.0 before release; `docs/external_code_check.md` lists this as open]
