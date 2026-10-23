# T2: Banks, questions and back-testing (Bhavya)

Checklist generated from tracks.md section 4.10. Read the whole track section first: it has the specs and acceptance tests.
Tick items off in your PRs and post one line per week in STATUS.md. Weeks count from TA approval (G0).

## Week 0
- [x] Install SimPy, Lark
- [x] DSL grammar sketch
- [x] report RAM/CPU
- [ ] small-model smoke test (Qwen3-1.7B / 4B-4bit)

## Week 1
- [x] DSL parser + executor (`ftm/dsl`, `tests/test_dsl*.py`)
- [x] start expert questions with Shreyas (55 programs for T4's 28 askable families, `ftm/dsl/expert.py`; `entity_accounts` and `internal_chain` sensitivity need his call)
- [ ] agree run-log schema with Shreyas
- [x] **Grammar v0 to Akshat** (`ftm/dsl/README.md`; handed over once the PR is up)

## Week 2
- [ ] Partner banks, DP mechanisms, accountant. **G1**

## Week 3
- [ ] Realistic back-testing
- [ ] e-values + e-BH
- [ ] **harness v0**. **G2**

## Week 4
- [ ] Simple planners + enumerator
- [ ] **harness v1 frozen**. **G3**

## Week 5
- [ ] Small-decision dataset v1
- [ ] raw-sharing baseline
- [ ] per-account cap tests

## Week 6
- [ ] Exact-mode back-testing
- [ ] ε-sweep tooling
- [ ] dataset to Akshat

## Week 7
- [ ] Co-Investigator / FinFRE-RAG ports (secondary)

## Week 8
- [ ] Async mode (stretch prep)
- [ ] back-testing rerun for held-out cycles (with Abhishek)

## Week 9
- [ ] All non-LLM planners + ablation configs on val. **G4**

## Week 10
- [ ] Async runs (stretch)
- [ ] ModernBERT/DeBERTa + RouteLLM-style baselines (secondary)

## Week 11
- [ ] Stretch external agents if time

## Week 12–13
- [ ] Final configs
- [ ] buffer

## Week 14
- [ ] Final test run: all non-LLM arms, banks for all LLM arms

## Week 15–16
- [ ] Results tables
- [ ] review
