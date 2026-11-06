# Stub audit of the production paths (2 Nov 2026, worker NOSTUB)

Method: `grep -rnIE "STUB|PLACEHOLDER|TODO|FIXME|dev only|not implemented|NotImplementedError|placeholder|fallback|mock|fake|random\.|illustrative"` over `ftm/`, `scripts/`, `configs/`, then a read of every hit that a real run reaches.
Classes: (a) production path, replaced now; (b) test or dev only, labelled, kept; (c) needs data or GPU, command given; (d) outside the files this worker owns, flagged for its owner.
No TODO or FIXME markers exist in `ftm/`, `scripts/`, `configs/`.

## Replaced (a)

| Where | Was | Now |
|---|---|---|
| `ftm/retrieval/views.py` (planner evidence in `scripts/run_llm_agent.py:data_alerts`) | week-2 STUB, top-40 by amount, 1 hop | `evidence_views.view(..., scorer="v2")`, name from `--evidence-view` (profile: `scorer_pcst`, rule below). The old selection is the named view `amount_1hop` (byte-identical, ablation) |
| `data_alerts` default view for callers naming none (`optimize_dspy.py`, `run_injection_tests.py`, exemplar pool, case memory) | the stub trail | the profile's view and scorer |
| `scripts/run_llm_agent.py:461` fusion default | `NaiveLRFusion` ("PLACEHOLDER") for any run without `--fusion-params` | profile `tuned` loads `ConditionalLR` (`configs/fusion/headroom_v1.json`); naive only by `--fusion naive` (named ablation) or `--profile none` (legacy, loud stderr note, recorded in `summary.json`); an explicit `--backtest` without its fusion is refused |
| `ftm/llm/investigation.py:12` docstring calling NaiveLRFusion/LRTable placeholders | misleading | states naive = ablation + shared LR lookup |
| `configs/final/jobs.yaml` `common_args` (old table 7df6bad5337d, naive fusion "until G2") | stale table, naive fusion | `--profile tuned` + the shape library; adapted-arm commands use `--profile tuned`; injection command uses the headroom table and fusion |
| `scripts/run_injection_tests.py:223` | `FusedLR.from_params` only (would fail on the conditional JSON), naive without params | ConditionalLR-aware loader, p0 v2 and scorer v2 env defaults, naive labelled |
| `scripts/t3_val_session.sh`, `scripts/t3_qwen_session.sh` | table 7df6bad5337d, p0 v1, stub trail, concurrency 1 | `--profile tuned` |
| planner `threshold=` and the pipeline's cannot_cross | never passed (threshold None) | `LLMPlanner(threshold=, stop_cannot_cross=)` and `adapted_system(...)` get the threshold of `configs/decision/headroom_v1.yaml` (0.3227) |
| `scripts/demo.py` + `ftm/demo/*` | default `--llm mock`, threshold 0.5 "illustrative", clear-below 0.1, old table, `scorer_pcst` hard-coded, silent fallback to the stub trail | default = live on the endpoint (or `--replay` of a real run); mock only with `--llm mock`, banner "SCRIPTED MOCK LLM" plus the label in titles; threshold from the decision config, two-way decision unless `--clear-below`; the evidence fallback is named (`amount_1hop`) |

### Evidence-view rule (written before the choice, fit+cal only)
`ftm.llm.profile.pick_evidence_view` on `configs/llm/evidence_view_fitcal_v2.json` (= `python -m ftm.retrieval.evidence_views --data hi_medium --parts fit cal --scorer v2`, 1,432 alerts, eval not read): candidates `scorer_pcst`, `scorer_greedy` need a connected time-respecting trail on >= 99% of alerts and a longest evidence <= 4,000 estimated tokens; best pattern-edge recall at k=40 wins, within 0.01 the fewer-token view wins; floor: if the winner is more than 0.01 below `amount_1hop`, `amount_1hop` stays. Result: recall `scorer_pcst` 0.836, `scorer_greedy` 0.839 (tie, pcst has fewer tokens), `amount_1hop` 0.819, `amount` 0.008, `scorer` 0.411, `ppr` 0.707 -> `scorer_pcst`. Prompt cap: `long_prompt` is cut at `--max-prompt-tokens - 2000`; the planner's trim/refit loop is unchanged. This is evidence-quality only; E8 found no Brier difference between views under naive fusion, so the with-vs-without Brier arms (rigor rule) are a GPU run: below.

## Kept (b), labelled

| Where | Why |
|---|---|
| `ftm/demo/mock.py`, `ftm/llm/devtools.py` (`DevRandomToolBox`, `dev_responder`), `*.dev_responder` in `ftm/external/*.py`, `ftm/graph/agents/registry.py:adapted_dev_responder` | dev only, reached via `--llm mock` / `--toolbox dev`; runs are tagged `variant=devtools` and refused for split=test (`refuse_devtools_test`) |
| `ftm/harness/toolbox.py` MockBank, `--alerts synthetic` | random answers behind the real gate; variant `harness_mock`, never a result; `ftm/final/runner.py:218` uses it only for the synthetic dry run |
| `ftm/llm/mock.py`, `ftm/baselines/mega_gnn.py:71`, `multi_gnn.py:97` (all-ones placeholder feature) | test double; the all-ones node feature is the published baseline's design, not a stub |
| `NaiveLRFusion()` constructor defaults in `ftm/external/*.py`, `ftm/graph/pipeline.py:117`, `ftm/planners/llm.py:81`, `tests` | reached only when a caller passes no fusion; `run_llm_agent.py` always passes one. Kept for unit tests and the ablation; owner of `pipeline.py` could make it required |
| `ftm/graph/agents/adapted.py:120 NotImplementedError` | abstract method |
| `ftm/fastmodel/cascade.py` `fallback` (too few calibration points -> always defer) | conservative by design, not a simplification |
| random seeds (`random.`, `np.random`) | seeded sampling/DP noise, not fallbacks |

## Flagged, not mine (d)

| Where | Issue | Action for the owner |
|---|---|---|
| `ftm/eval/final_additions.py:30` `DEFAULT_CAPACITY` "placeholder, to be pre-registered" (top 10%) | validation-only default; test needs `fixed_on_val.json` | T4: fix capacity on val before freeze |
| `configs/experiments/e18_backbone_sweep.yaml:43`, `scripts/e10_final.py:117`, `scripts/train_planner.py:7,153` | old table `7df6bad5337d` / `NaiveLRFusion` | backbone-refresh worker: point at `configs/fusion/headroom_v1_table.jsonl` + `--profile tuned` (planner distillation `llm_distill.py:254` already refuses naive unless `allow_naive`) |
| `configs/final/jobs.yaml` `s_per_alert` | medians of Gemma runs under the OLD stack (naive fusion, stub view, concurrency 1) | refill with `scripts/t3_fill_s_per_alert.py` after the first tuned val eval (GPU) |
| harness run log | `rule_id` for allowed asks and `gate_reason` are not written (`QuestionLogEntry` has no such field; harness/types not mine); the demo re-derives the rule and marks it "re-derived" | harness owner: add the fields to `QuestionLogEntry` and `verdict_rows` |
| `scripts/run_injection_tests.py`, Group C/D ports | do not receive the decision threshold (no use for it) | none |

## GPU / data commands (c)
Server up as in `scripts/t3_val_session.sh` (`serve_llm.py --model gemma4-12b`), then, with-vs-without arms on val `fit` + `cal` (choose) and `eval` once (report):
```
PROFILE="--profile tuned"
# tuned
python scripts/run_llm_agent.py --system planner --alerts val --val-part fit --data hi_medium --n 191 --llm endpoint --toolbox harness --model gemma4-12b $PROFILE --run-dir outputs/t3_tuned/fit/tuned
# evidence-view ablation (one flag each): --evidence-view amount_1hop | amount | long_prompt
# fusion ablation: --fusion naive          decision ablation: --no-cannot-cross          p0 ablation: --p0-variant v1
# the same four on --val-part cal --n 109; eval (160) only after the arms are fixed
python scripts/t3_val_session.sh eval        # every system on the tuned profile
```
Analysis of the arms with ftm/eval/bootstrap.py is the follow-up (not run here: CPU only, no endpoint).
