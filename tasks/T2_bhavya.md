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
- [x] Exact-mode back-testing (`--exact-labels`, `--compare`; real numbers need `data/`)
- [x] ε-sweep tooling (`scripts/eps_sweep.py`)
- [ ] dataset to Akshat: the builder and format are ready (week 5); the real 50k run needs `data/`

## Week 7
- [x] Co-Investigator / FinFRE-RAG ports (secondary), native + ported (`ftm/external/GROUPC_T2.md`); runs on the real backbone need the endpoint

## Week 8
- [x] Async mode (stretch prep): waves, batched planners, SimPy runner, sequential-vs-batched study (`ftm/orchestrator/README.md`)
- [x] back-testing rerun for held-out cycles: loader flag, CLI flag, transfer report (the real run needs `data/` and T1)

## Week 9
- [x] Learned policy (`ftm/planners/learned.py`, `scripts/train_policy.py`) and all non-LLM planners + ablation configs on val (`scripts/run_ablations.py`, `ftm/orchestrator/README.md`)
- [ ] **G4**: needs the run on real validation (`data/`, T1's alert sets, the back-test tables) and T3's GPU-hour numbers for the compute check

## Week 10
- [x] Async runs (stretch): `scripts/async_study.py` (tooling and synthetic check; the real E13 rows need `data/` and low-eps accepted questions, `ftm/orchestrator/README.md`)
- [x] ModernBERT/DeBERTa + RouteLLM-style baselines (secondary): `ftm/fastmodel/{encoder,router}.py`, `scripts/encoder_baseline.py`, `ftm/fastmodel/BASELINES.md`; training on the 3050 Ti and real numbers need the decision set, the 8B scores and a CUDA torch

## Week 11
- [x] Stretch external agents (one): MediQ port (`ftm/external/mediq.py`, `ftm/external/GROUPC_T2.md`); not in the locked external list, so exploratory unless the TA adds it. CodeAct, GraphDx, NOL*, ExpeL, AWM not ported (reasons in the doc)
- [x] T1's flag from hand-off A4: `cross_currency_out(..., window=after(72))` cannot see the future (`tests/test_window_future.py`: the ledger is cut at the answer time, checked through the bank and the harness clock)

## Week 12–13
- [ ] Final configs
- [ ] buffer

## Week 14
- [ ] Final test run: all non-LLM arms, banks for all LLM arms

## Week 15–16
- [ ] Results tables
- [ ] review
