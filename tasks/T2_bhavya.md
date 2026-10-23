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
- [x] Partner banks (`ftm/banks`), DP mechanisms and accountant (`ftm/privacy`), see `ftm/banks/README.md`
- [ ] **G1**: needs T1's one-bank/alert results and T3's endpoint; the banks, DP and accountant side is ready

## Week 3
- [x] Realistic back-testing (`ftm/backtest`, `scripts/backtest.py`; real-data run needs T1's data, see `ftm/backtest/README.md`)
- [x] e-values + e-BH (+ e-LOND for new questions at test time)
- [x] **harness v0** (`ftm/harness`, `ftm/harness/README.md`)
- [ ] **G2**: needs the real-data back-test of the expert list (T1's data and alert sets) and the Brier-gain check with T4's fusion

## Week 4
- [x] Simple planners (random, fixed, EIG) + enumerator (`ftm/planners/NONLLM.md`); learned policy is week 9
- [x] **harness v1 frozen** (`tests/test_harness_contract.py`)
- [ ] **G3**: needs real-data back-testing (>= 20 expert questions accepted) and ReAct on 50 val alerts through the harness (Akshat)

## Week 5
- [x] Small-decision dataset v1 (`scripts/make_decision_set.py`, `ftm/orchestrator/README.md`; the real 50k run needs `data/`)
- [x] raw-sharing baseline (`ftm/orchestrator/raw_share.py`)
- [x] per-account cap tests and the cap sweep tool (`scripts/cap_sweep.py`)

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
