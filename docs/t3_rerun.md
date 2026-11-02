# T3 rerun: one resumable script for every remaining GPU stage (2 Nov 2026)

Owner: T3 (Akshat). Script: `scripts/t3_rerun.sh <stage>`; helpers `scripts/t3_rerun_tools.py` (chunk list, stop rebalance, selection parsing, hour table), `scripts/t3_rerun_report.py` (points the existing analysis scripts at the new tree); tests `tests/test_t3_rerun.py`. `DRY_RUN=1 scripts/t3_rerun.sh <stage>` prints every command and runs nothing (the e18 and rag wrappers pass `DRY_RUN` on to their own scripts).

## Rules the script enforces

* Every LLM run: `--profile tuned --concurrency 4`, `--toolbox harness`, seed 0, chunks of 80 alerts (`CHUNK`), one run directory per chunk. A chunk is done when its `summary.json` exists; a stage that is killed and started again skips finished chunks and finished outputs (libraries, distilled set, trained seeds via `--skip-existing`, calibration maps).
* Val parts are separate trees: `outputs/t3_val_v3/{seed,fit,cal,eval}/<spec>/c###`. Fit and cal are for choosing; `eval` is run once (stage `val-eval` refuses until every external's calibration map exists; the E16 and System 1 eval parts leave a `.done` marker). The test split is never named: no `--final`, no `--alerts test`, `FTM_FINAL_RUN` untouched.
* New versioned paths only, nothing existing is overwritten: `outputs/t3_teacher_v2`, `outputs/t3_library/frozen_v3`, `outputs/sft_v3`, `models/planner_sft_v3`, `outputs/t3_val_v3`, `outputs/t3_dspy_v3`, `outputs/injection_v3`. The stale E18 Gemma result is moved to `outputs/e18_refresh_stale/` once, not deleted.
* Fit and cal runs use a library built **without** val labels (`*_gemma_train.json`, teacher_v2 only). Only eval (and everything after) uses `*_gemma.json`, which adds the seeding runs on fit and cal. So no fit or cal alert sees its own outcome through the library, and the calibrators are fitted on runs that match the eval runs except for the library contents (a labelled difference: the eval library has 300 more records).
* Prompt-budget guard: `teacher` exits 2 unless `PROMPTFIT_MERGED=1`. The tuned prompt is about 10k tokens against the 6k cap; without the PROMPTFIT fix the run ends on one evidence row and a 20/51 menu (the e18_refresh Gemma result that was stopped; fixed in 1a0a169). The queue line must name a commit that contains that fix. Each chunk's log line reports its `PromptTooLong` and `PromptRefit` counts; look at the first chunk before leaving it unattended.

## Stages, in dependency order

| stage (sub-stages) | what | output |
|---|---|---|
| `teacher` | Gemma 4 12B planner (compose, `--keep-prompts`) on all 1,983 HI-Medium train alerts (25 chunks); then the seeding runs (compose, no library) on val fit (3 chunks) and cal (2 chunks) in the same server session | `outputs/t3_teacher_v2/runs/c000..c024`, `outputs/t3_val_v3/seed/{fit,cal}` |
| `library` (CPU) | `build_library.py --freeze` for shape and account keying, twice: teacher_v2 only (`*_gemma_train.json`) and teacher_v2 plus the seeding runs (`*_gemma.json`) | `outputs/t3_library/frozen_v3/` |
| `distill` (CPU) | `train_planner.py distill --filter expected` (headroom_v1 table and fusion params, stop threshold 0.3227, teacher revision read from `serving.yaml`), then `t3_rerun_tools.py rebalance`: every question example kept, stop examples capped at 25% of the set by a hash of (alert, step), never duplicated; before and after counts in `stop_rebalance.json` (v1 had 252 kept questions and 0 stops, so the cap is a no-op unless the new filter emits many) | `outputs/sft_v3/planner_train.jsonl`, `planner_train_balanced.jsonl`, `stop_rebalance.json` |
| `student` = `student-train` + `student-e16` | QLoRA, 3 seeds, train venv, `configs/llm/sft_planner_v3.yaml` (copy of `sft_planner.yaml`, output dir changed); then the E16 rerun on the new serving profile `qwen3-8b-lora-v3` (prompted Qwen3-8B plus `planner-sft-v3-s0..2`) on fit, cal, eval (eval once). The Gemma reference is `ours_compose` of stage `val` on the same parts. Reports: `t3_rerun_report.py e16-choose --part fit/cal` and `e16` (= `scripts/t3_e16_report.py` on the new tree) | `models/planner_sft_v3`, `outputs/t3_val_v3/{fit,cal,eval}/e16_*`, `outputs/t3_val_v3/analysis/e16*.json` |
| `val` = `val-dspy`, `val-fitcal`, `val-calibrate`, `val-eval`, `val-apply`, `val-report` | MIPROv2 program on val fit (`DSPY=<old program>` reuses the old one); all 33 systems on fit and cal; `calibrate_external.py fit` per external (fit on `fit`, check on `cal`, identity map when cal Brier does not improve); all systems on eval once; the frozen maps applied to the eval run logs (`eval_calibrated/`); `val_runs_table.py` (regenerates `docs/t3_val_runs.md`), `t3_rerun_report.py analysis` (= `t3_val_analysis.py`, tables in `outputs/t3_val_v3/analysis/tables.md` to paste into `docs/t3_val_analysis.md`, whose prose is written by hand), STATUS line draft in `~/projects/rewrite_notes/drafts/status_val_lines.md`. Systems: ours x7 (compose, llm_decides, act_only, fixed_list, batched_2_1, library_account, no_backtest), 10 adapted (`*_adapted`, mediq included), ported Group D x9 (react, reflexion, rewoo, lats, llmcompiler, dspy_react with its program, debate, uot, bed_llm), Group C native+ported x8 (sharma, torres, co_investigator, finfre_rag) with the case memory on (default `--memory-n 500`, stratified; `--no-memory` is not used) | `outputs/t3_val_v3/`, `docs/t3_val_runs.md` |
| `e15` | `run_injection_tests.py` on the real endpoint, val `eval`, diverse val injection family (`data/safety/injection_val_diverse.jsonl`, `generate_diverse(...,'val')`): planner (60 alerts; arms nogate, gate, gate+camel_flow, gate+camel_full, gate+screen with the keyword screen, labelled heuristic) and react (40 alerts; nogate, gate, gate+screen); p0 v2, scorer v2, headroom_v1 table and fusion | `outputs/injection_v3/e15_{planner,react}` |
| `e18` | moves the stale Gemma result aside once, then `scripts/t3_e18_refresh.sh` with the seven candidates (`E18_CANDIDATES`) | `outputs/e18_refresh` |
| `rag` | `scripts/t3_rag_bench.sh library arms report combined eval` (`RAG_STAGES` to subset). Its own `library` step reruns 1.4 GPU-h of train teacher runs on the tuned stack; skip it with `RAG_STAGES="arms report combined eval"` if the frozen_v3 libraries are wanted instead (the arm table names its library files, check `plan`) | `outputs/rag_bench` |
| `system1` | `docs/t3_system1.md` section 8, steps 3 to 8: (3) `t3_qwen_session.sh fast3` unless all three seeds are scored; (4) perturbed twins, `score_fast_baseline.py report --freeze`, consistency report (seed `S1_SEED`, default 0); (5) dual-serve smoke test (8 alerts, stops if `s1_decisions` is 0); (6) six integration arms on fit and cal; (7) `system1_report.py report --part fitcal` writes the selection, then eval once: the off arm and the selected arm only (a selection with no function adopted runs the off arm only), `--part eval`; (8) E15 with the fast-model screen on the held-out families (planner and react, gate vs gate+screen). Steps 0 to 2 are the queue lines `s1_*` that already exist | `outputs/s1v1`, `outputs/s1` |

## GPU hours

Basis: measured per-alert seconds of the existing runs (`outputs/t3_val/eval/*/per_alert.jsonl`; `scripts/t3_rerun_tools.py hours` rebuilds the table). Those runs were serial: wall time equals the sum of the per-alert seconds. At concurrency 4 the wall time is assumed to be serial / 2.5 (`SPEEDUP` in the tool; `docs/t3_system1.md` section 8 quotes 23 s falling to 8 to 10 s per alert; the vLLM KV cache holds only about 14k tokens, so 4 concurrent 10k-token prompts will not all be resident, which is why 2.5 and not 4). The tuned prompt is longer than the one in those runs, so treat every inference figure as a floor: if concurrency gains nothing, the inference rows approach the serial column. The adapted arms have no run yet (their ported counterpart stands in), `mediq_adapted` has none (the planner's seconds stand in).

| stage | serial h | planning h (concurrency 4) | basis |
|---|---|---|---|
| `e18` | | 3.0 | the earlier 6-candidate estimate (2.5 h) plus Phi-4 |
| `teacher` | 3.8 | 1.6 | 1,983 alerts x 5.4 s (teacher v1 measured 2.99 h) + 300 seeding alerts x 7 s |
| `library`, `distill` (CPU) | | 0.1 | CPU only |
| `student-train` | | 10 (formula) | v1 measured 4 s per example per epoch (3 seeds, 252 examples, 1.68 h); 2 epochs: 8 s per example per seed. 1,000 examples: 6.7 h; 1,500 (planning figure): 10 h; 2,000: 13.3 h. Replace once `distill` reports the example count |
| `student-e16` | 1.6 | 0.9 | 4 arms x 460 alerts x 3.1 s (E16 measured), 4 server loads included at the c4 figure |
| `val-dspy` | 2.7 | 2.7 | measured 9,833 s (set `DSPY` to reuse the old program: 0) |
| `val-fitcal` | 34.7 | 14.4 | 300 alerts x 33 systems, 65% of the table below, plus load |
| `val-eval` | 18.5 | 7.9 | 160 alerts x 33 systems, 35% of the table below, plus load |
| `e15` | 8.5 | 5.4 | planner 60 x 5 arms x 6 items x 7.5 s + react 40 x 3 x 6 x 8 s; the injection driver is serial, no concurrency |
| `rag` | | 4.0 | the T3 todo figure, includes its 1.4 h library step |
| `system1` (steps 3 to 8) | | 10.2 | fast scoring 0.5, smoke 0.2, fit+cal 6 arms 4.5, eval 2 arms 0.8, E15 with the fast screen 3.9 (planner 60 x 3 arms, react 40 x 3 arms), CPU and loads 0.3 |
| **total** | | **about 60** | with no concurrency gain the inference stages grow 2.5-fold to roughly 120 |

`val` per-system table (460 alerts: fit 191 + cal 109 + eval 160), measured seconds, regenerate with `scripts/t3_rerun_tools.py hours --root outputs/t3_val/eval`:

| spec | s/alert (serial) | basis | 460 alerts serial h | at concurrency 4 (/2.5) h |
|---|---|---|---|---|
| ours_compose | 6.4 | ours_compose | 0.82 | 0.33 |
| ours_llm_decides | 6.8 | ours_llm_decides | 0.86 | 0.35 |
| ours_act_only | 4.6 | ours_act_only | 0.59 | 0.23 |
| ours_fixed_list | 4.1 | ours_fixed_list | 0.52 | 0.21 |
| ours_batched_2_1 | 6.0 | ours_batched_2_1 | 0.77 | 0.31 |
| ours_library_account | 6.5 | ours_library_account | 0.83 | 0.33 |
| ours_no_backtest | 12.5 | ours_no_backtest | 1.60 | 0.64 |
| react_adapted | 10.3 | react (ported run stands in) | 1.31 | 0.52 |
| reflexion_adapted | 12.3 | reflexion (ported run stands in) | 1.57 | 0.63 |
| rewoo_adapted | 10.5 | rewoo (ported run stands in) | 1.34 | 0.54 |
| lats_adapted | 27.0 | lats (ported run stands in) | 3.45 | 1.38 |
| llmcompiler_adapted | 8.0 | llmcompiler (ported run stands in) | 1.02 | 0.41 |
| dspy_react_adapted | 26.0 | dspy_react (ported run stands in) | 3.32 | 1.33 |
| debate_adapted | 16.2 | debate (ported run stands in) | 2.07 | 0.83 |
| uot_adapted | 8.9 | uot (ported run stands in) | 1.14 | 0.45 |
| bed_llm_adapted | 15.2 | bed_llm (ported run stands in) | 1.94 | 0.78 |
| mediq_adapted | 6.4 | ours_compose (no run of this spec) | 0.82 | 0.33 |
| react | 10.3 | react | 1.31 | 0.52 |
| reflexion | 12.3 | reflexion | 1.57 | 0.63 |
| rewoo | 10.5 | rewoo | 1.34 | 0.54 |
| lats | 27.0 | lats | 3.45 | 1.38 |
| llmcompiler | 8.0 | llmcompiler | 1.02 | 0.41 |
| dspy_react | 26.0 | dspy_react | 3.32 | 1.33 |
| debate | 16.2 | debate | 2.07 | 0.83 |
| uot | 8.9 | uot | 1.14 | 0.45 |
| bed_llm | 15.2 | bed_llm | 1.94 | 0.78 |
| sharma_native | 4.3 | sharma_native | 0.54 | 0.22 |
| sharma_ported | 6.0 | sharma_ported | 0.76 | 0.31 |
| torres_native | 14.7 | torres_native | 1.88 | 0.75 |
| torres_ported | 18.4 | torres_ported | 2.35 | 0.94 |
| co_investigator_native | 21.4 | co_investigator_native | 2.73 | 1.09 |
| co_investigator_ported | 23.1 | co_investigator_ported | 2.95 | 1.18 |
| finfre_rag_native | 1.7 | finfre_rag_native | 0.22 | 0.09 |
| finfre_rag_ported | 4.7 | finfre_rag_ported | 0.59 | 0.24 |
| **total** | | | 53.2 | 21.3 |

The table total (53.2 h serial, 21.3 h at concurrency 4) is the figure the `val-fitcal` and `val-eval` rows split.

## Recommended single queue order

Training takes the whole 16 GB card, so nothing runs beside it; "interleaving" here means ordering so that the long, riskiest job fails early and each CPU step sits straight behind its input. `qlora_planner2` already died once (OOM), so `student-train` goes right after `distill`, before the 25 h of validation runs; its adapters are only needed by `student-e16`, which is placed behind `val-fitcal` because it needs the `ours_compose` reference on the same parts. `e18_refresh2` (already queued by the lead, the 7-candidate refresh on the corrected prompt; the stage `e18` of the script is the same wrapper) runs first and these lines go after it: if it picks a backbone other than Gemma 4 12B, every later stage must be rerun with `MODEL=<new>` (the stage reads `MODEL`, and the libraries and distilled set follow the teacher). If the user prefers the order of the request (teacher first, e18 seventh) move that one line. Everything after `rerun_e18` needs the same commit; `<SHA>` is the commit that contains PROMPTFIT, S1BENCH and this script, merged on `t3/rewrite`. FraudGT hi_small stays deferred behind these (24 h, T1).

```
rerun_teacher      | <SHA> | PROMPTFIT_MERGED=1 bash scripts/t3_rerun.sh teacher
rerun_library      | <SHA> | bash scripts/t3_rerun.sh library
rerun_distill      | <SHA> | bash scripts/t3_rerun.sh distill
rerun_student_train| <SHA> | bash scripts/t3_rerun.sh student-train
rerun_val_dspy     | <SHA> | bash scripts/t3_rerun.sh val-dspy
rerun_val_fitcal   | <SHA> | bash scripts/t3_rerun.sh val-fitcal
rerun_val_calibrate| <SHA> | bash scripts/t3_rerun.sh val-calibrate
rerun_student_e16  | <SHA> | bash scripts/t3_rerun.sh student-e16
rerun_val_eval     | <SHA> | bash scripts/t3_rerun.sh val-eval
rerun_val_apply    | <SHA> | bash scripts/t3_rerun.sh val-apply val-report
rerun_e15          | <SHA> | bash scripts/t3_rerun.sh e15
rerun_system1      | <SHA> | bash scripts/t3_rerun.sh system1
rerun_rag          | <SHA> | bash scripts/t3_rerun.sh rag
```

The prompt-budget fix is merged (1a0a169: cap 6,750, ranked top-8 menu, min_evidence_rows 40); `<SHA>` must be at or after it (this branch merged it at d759bee). The new bench arms menu_topk_{8,16,all} and evidence_min_{0,10,25} are read by `t3_rag_bench.sh` from `t3_rag_bench_report.py`, so stage `rag` runs them without change. The existing `s1_*` lines (fast SFT seeds, large3, soft arm, `s1_fast3`) and `gnn_causal_mega3` stay ahead of these. `rerun_system1` needs `s1_train_s1s2`, `s1_train_soft_s0` and `s1_fast3` done (its step 3 skips when the three `latency.json` files exist). Because the queue runs the step in the pinned `tt-run` checkout, `MODEL`, `V`, `CHUNK` and the like can be put in front of `bash` in a line (`MODEL=qwen35-9b bash ...`). A failed step leaves `failed/<name>`; remove the marker (or run the stage by hand) after fixing it, the stage resumes at the first unfinished chunk.

## Caveats

* `configs/llm/serving.yaml` gained the `qwen3-8b-lora-v3` profile (same base as `qwen3-8b-lora`, adapters at `models/planner_sft_v3`); `configs/llm/sft_planner_v3.yaml` is new.
* `t3_val_analysis.py` and `t3_e16_report.py` hard-code `outputs/t3_val`; `t3_rerun_report.py` re-points their module constants (tree, tuning runs, back-test table `configs/fusion/headroom_v1_table.jsonl` instead of the week-2 table `7df6bad5337d`) without editing them. `t3_e16_report.py` asserts 160 alerts, so fit and cal use `e16-choose` (Brier and paired dBrier vs p0 with the cluster bootstrap) instead.
* The adapted arms' run-log `system` names are not known before a run; `val-calibrate` reads them from the run logs (`runlog-system`).
* No run has been made: everything above is a command list, a DRY_RUN check and unit tests. The cluster file `outputs/e18/clusters.json` must exist for the reports (it does in the main checkout).
