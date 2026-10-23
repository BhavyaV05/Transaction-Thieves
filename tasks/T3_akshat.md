# T3: LLM planner, fine-tuning and agent frameworks (Akshat)

Checklist generated from tracks.md section 5.8. Read the whole track section first: it has the specs and acceptance tests.
Tick items off in your PRs and post one line per week in STATUS.md. Weeks count from TA approval (G0).

## Week 0
- [ ] Serving smoke test (8B, 4B, gpt-oss-20b): ran 2026-10-23 on native Linux. **Qwen3-8B-AWQ and Qwen3-4B-AWQ pass on sm_120** (decode 117.4 / 172.1 tok/s). **gpt-oss-20b fails**: weight-allocation OOM with the desktop using ~1.6 GB of the 15.47 GiB; three attempts (`docs/t3_week0.md` §4). Open: try the GPU headless, else Qwen3-14B as teacher
- [x] recipe: native Linux recipe ran for the two Qwen profiles (`docs/hardware.md` §5); gpt-oss-20b needs the headless fallback noted there
- [x] conversion benchmark: script published (`scripts/bench_llm.py`); reference run done, `outputs/bench/t3-5070ti_qwen3-8b.json` (not committed), row in `docs/hardware.md` §4
- [ ] check each E18 candidate loads at the context cap: **moved to the E18 sweep** (not week 0). Per candidate: download once, load at `max_model_len` 8192 and record load time and peak VRAM, run its stage-1 alerts, delete the weights before the next (`docs/t3_e18.md`, per-candidate procedure)
- [x] report specs (`docs/hardware.md` §2, `docs/t3_week0.md` §1) and model-card check (`docs/t3_week0.md` §2)

## Week 1
- [x] Planner prompt + JSON schema (`ftm/planners/llm_prompt.py`): written against a **provisional** DSL stand-in (`ftm/planners/llm_dsl.py`) because grammar v0 (hand-off #3) has not arrived; switching is one `DslAdapter`

## Week 2
- [ ] **Shared endpoint up**: client, budgets, mock server and README done (`ftm/llm/`); the real server waits for the GPU smoke test
- [x] top-40-by-amount evidence stub (`ftm/retrieval/views.py`, through `ftm.data.io`, aliased). **G1** support

## Week 3
- [ ] LLM planner end-to-end on 20 val alerts. **G2**: planner loop, 5 modes and run script done and tested on mocks (`ftm/planners/llm.py`, `scripts/run_llm_agent.py`, `docs/t3_planner_agents.md`); the real run waits for harness v0 + banks (T2), the endpoint and val alerts on this machine

## Week 4
- [ ] **ReAct through the harness on 50 val alerts**: ReAct reimplemented (`ftm/external/react.py`) and runs through the shared `Investigation` layer; same blockers as G2
- [ ] teacher runs start (blocked on the GPU smoke test and the harness). **G3** (library schema and keying frozen): schema v1 written (`ftm/library/shape.py`, `store.py`); freeze it at G3 with the team

## Week 5
- [ ] Strategy library build: builder, search, freeze/hash, E11 account keying and `scripts/build_library.py` done and tested; the real build needs teacher/planner runs and the back-test table
- [ ] teacher runs continue

## Week 6
- [x] Reflexion, ReWOO ports (reimplementations; `ftm/external/reflexion.py`, `rewoo.py`; adaptations in `docs/t3_planner_agents.md` §3)
- [ ] small-decision dataset arrives

## Week 7
- [ ] QLoRA planner SFT: distillation filter, `train_planner.py` (distill / train, dry-run) and `configs/llm/sft_planner.yaml` done; training needs teacher runs, the back-test table and a GPU window (`docs/t3_finetune_fastmodel.md`)
- [ ] Qwen3-4B fast-model fine-tune: decisions, single-token SFT format, `LogprobFastModel`, JSON baseline, temperature scaling and `configs/llm/sft_fast.yaml` done; needs the small-decision dataset (T2) and a GPU window
- [ ] planner-view decision with Abhishek (default: train on top-40, the current stub)

## Week 8
- [x] LATS (adapted: no rollouts, since bank answers cannot be reset), LLMCompiler (parallel waves count the slowest answer) (secondary)

## Week 9
- [ ] DSPy-tuned ReAct: program, fair LM adapter (our BudgetedClient) and MIPROv2 `optimize()` written and tested on mocks; the optimisation run needs val alerts through the harness and the endpoint
- [ ] fine-tuned models ready. **G4**

## Week 10
- [x] Conformal cascade (`ftm/fastmodel/cascade.py`: Mondrian split conformal, two tiers, safe defaults, marginal 2α check; tested on synthetic data, real evaluation needs the fast model)
- [ ] SemIf / Laya (stretch)

## Week 11
- [ ] Stretch frameworks if time

## Week 12
- [ ] Final configs

## Week 13
- [ ] Buffer. Final-run scheduling
- [ ] GPU calendar for week 14

## Week 14
- [ ] **Final test run**, LLM jobs split between your server and Abhishek's with identical checkpoint

## Week 15
- [ ] Case studies (10, good and bad)

## Week 16
- [ ] Review
