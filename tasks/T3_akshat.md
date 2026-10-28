# T3: LLM planner, fine-tuning and agent frameworks (Akshat)

Checklist generated from tracks.md section 5.8. Read the whole track section first: it has the specs and acceptance tests.
Tick items off in your PRs and post one line per week in STATUS.md. Weeks count from TA approval (G0).

## Week 0
- [x] Serving smoke test (8B, 4B, gpt-oss-20b): ran 2026-10-23 on native Linux. **Qwen3-8B-AWQ and Qwen3-4B-AWQ pass on sm_120** (decode 117.4 / 172.1 tok/s). gpt-oss-20b failed under vLLM (OOM beside the desktop, `docs/t3_week0.md` §4) and ran on llama.cpp in E18 instead; the teacher is Gemma 4 12B (E18)
- [x] recipe: native Linux recipe ran for the two Qwen profiles (`docs/hardware.md` §5); gpt-oss-20b needs the headless fallback noted there
- [x] conversion benchmark: script published (`scripts/bench_llm.py`); reference run done, **JSON committed at `docs/bench/t3-5070ti_qwen3-8b.json`** (28 Oct), row in `docs/hardware.md` §4
- [x] check each E18 candidate loads at the context cap: done inside the E18 sweep (`outputs/e18/stage*/load_<name>.json`, `docs/t3_e18.md`); the gpt-oss, Qwen3-14B and GGUF weights were deleted after the sweep
- [x] report specs (`docs/hardware.md` §2, `docs/t3_week0.md` §1) and model-card check (`docs/t3_week0.md` §2)

## Week 1
- [x] Planner prompt + JSON schema (`ftm/planners/llm_prompt.py`): written against a **provisional** DSL stand-in (`ftm/planners/llm_dsl.py`) because grammar v0 (hand-off #3) has not arrived; switching is one `DslAdapter`

## Week 2
- [x] **Shared endpoint up**: client, budgets, mock server and README (`ftm/llm/`); the real server runs on the 5070 Ti (`scripts/serve_llm.py`, profile `gemma4-12b`, loopback only)
- [x] top-40-by-amount evidence stub (`ftm/retrieval/views.py`, through `ftm.data.io`, aliased). **G1** support

## Week 3
- [x] LLM planner end-to-end on 20 val alerts. **G2**: first real runs 23 Oct (`docs/t3_first_runs.md`); on real data the expert questions do not reach the 0.01 Brier gain and the planner's Brier is not below p0 under naive fusion (team decision pending, `docs/schedule_nov10.md` rule 6)

## Week 4
- [x] **ReAct through the harness**: runs in `outputs/t3_first/` (Qwen3-8B, 20 alerts); the Gemma 4 12B val run on 160 eval alerts is in `docs/t3_val_runs.md`
- [x] teacher runs: Gemma 4 12B planner on the 1,983 HI-Medium train alerts (26 Oct, 2.99 GPU-h, `outputs/t3_teacher/runs`). **G3**: library schema v2 (shape keys; `ftm/library/shape.py`, `store.py`)

## Week 5
- [x] Strategy library build: draft on the Gemma teacher runs (`docs/t3_library_draft.md`); **frozen shape and account libraries** (adding val `fit`/`cal` planner runs) are built by `scripts/t3_val_session.sh library`, hashes in `docs/t3_library_draft.md`
- [x] teacher runs continue (done, see Week 4)

## Week 6
- [x] Reflexion, ReWOO ports (reimplementations; `ftm/external/reflexion.py`, `rewoo.py`; adaptations in `docs/t3_planner_agents.md` §3)
- [x] small-decision dataset arrived (`data/decisions/hi_medium`, T2): 59,450 train and 13,792 val rows

## Week 7
- [ ] QLoRA planner SFT: distillation done (252 kept questions from 2,817, 0 stops: a very small set, `docs/t3_finetune_fastmodel.md`); training (3 seeds) and the E16 val runs are queued behind the Gemma session
- [ ] Qwen3-4B fast-model fine-tune: training set built (59,450 rows, `train_planner.py fast-data`); 3 seeds, scoring and the cascade are queued (`docs/t3_e10.md` when done)
- [x] planner-view decision with Abhishek: the planner trains and runs on the `amount_1hop` top-40 view (E8, `docs/t3_e8.md`, `docs/t1_planner_view.md`); `scorer_pcst` stays the E8 inference arm

## Week 8
- [x] LATS (adapted: no rollouts, since bank answers cannot be reset), LLMCompiler (parallel waves count the slowest answer) (secondary)

## Week 9
- [ ] DSPy-tuned ReAct: program, fair LM adapter, MIPROv2 `optimize()` (instruction search, no demos), `scripts/optimize_dspy.py` and `--dspy-program` tested on mocks (dspy 3.4.1); the optimisation run on val `fit` is in `scripts/t3_val_session.sh dspy`
- [ ] fine-tuned models ready. **G4** (30 Oct): val table and top-agents proposal in `docs/t3_val_runs.md`

## Week 10
- [x] Conformal cascade (`ftm/fastmodel/cascade.py`: Mondrian split conformal, two tiers, safe defaults, marginal 2α check; tested on synthetic data, real evaluation needs the fast model)
- [ ] SemIf / Laya (stretch)

## Week 11
- [ ] Stretch frameworks if time

## Week 12
- [ ] Final configs: `configs/llm/default.yaml` is Gemma 4 12B; `configs/final/jobs.yaml` T3 slots filled 28 Oct (common args, E2/E11/E13/E16 jobs, `gpu_s1` off); `s_per_alert` are placeholders until the val runs finish

## Week 13
- [ ] Buffer. Final-run scheduling
- [ ] GPU calendar for week 14

## Week 14
- [ ] **Final test run**, LLM jobs split between your server and Abhishek's with identical checkpoint

## Week 15
- [ ] Case studies (10, good and bad)

## Week 16
- [ ] Review
