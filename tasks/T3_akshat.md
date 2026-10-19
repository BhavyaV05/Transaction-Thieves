# T3: LLM planner, fine-tuning and agent frameworks (Akshat)

Checklist generated from tracks.md section 5.8. Read the whole track section first: it has the specs and acceptance tests.
Tick items off in your PRs and post one line per week in STATUS.md. Weeks count from TA approval (G0).

## Week 0
- [ ] Serving smoke test (8B, 4B, gpt-oss-20b): scripts and pinned profiles ready (`scripts/serve_llm.py`, `configs/llm/serving.yaml`); **GPU run pending** (move Docker's disk to E: first; steps in `docs/t3_week0.md` §4)
- [ ] recipe: written (`docs/hardware.md` §5); tick after the smoke test confirms it
- [ ] conversion benchmark: script published (`scripts/bench_llm.py`, `docs/hardware.md` §4); reference run pending the smoke test
- [x] report specs (`docs/hardware.md` §2, `docs/t3_week0.md` §1) and model-card check (`docs/t3_week0.md` §2)

## Week 1
- [x] Planner prompt + JSON schema (`ftm/planners/llm_prompt.py`): written against a **provisional** DSL stand-in (`ftm/planners/llm_dsl.py`) because grammar v0 (hand-off #3) has not arrived; switching is one `DslAdapter`

## Week 2
- [ ] **Shared endpoint up**: client, budgets, mock server and README done (`ftm/llm/`); the real server waits for the GPU smoke test
- [x] top-40-by-amount evidence stub (`ftm/retrieval/views.py`, through `ftm.data.io`, aliased). **G1** support

## Week 3
- [ ] LLM planner end-to-end on 20 val alerts. **G2**

## Week 4
- [ ] **ReAct through the harness on 50 val alerts**
- [ ] teacher runs start. **G3** (library schema and keying frozen)

## Week 5
- [ ] Strategy library build
- [ ] teacher runs continue

## Week 6
- [ ] Reflexion, ReWOO ports
- [ ] small-decision dataset arrives

## Week 7
- [ ] QLoRA planner SFT
- [ ] Qwen3-4B fast-model fine-tune
- [ ] planner-view decision with Abhishek

## Week 8
- [ ] LATS, LLMCompiler (secondary)

## Week 9
- [ ] DSPy-tuned ReAct
- [ ] fine-tuned models ready. **G4**

## Week 10
- [ ] Conformal cascade
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
