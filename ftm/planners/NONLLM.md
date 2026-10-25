# Non-LLM planners (C9): random, fixed order, EIG, learned policy, enumerator

Owner: T2 (Bhavya). Spec: `plan.md` 6.10, `tracks.md` 4.7. Files: `pool.py`, `session.py`, `simple.py` (T3 owns `llm*.py` in this package; the orchestrator that runs these over alerts is `ftm/orchestrator/runner.py`).

Each is an `Investigator` over the **same harness, budgets, stop rule and fusion** as every other system. They choose from a **pool** of back-tested, e-BH-accepted questions (`lr_table.jsonl`, `ftm/backtest/README.md`); only the choice differs:

| Planner | Choice |
|---|---|
| `RandomPlanner` | uniform over the questions that fit the eps left (seeded per alert, so alert order does not matter) |
| `FixedOrderPlanner` | the expert ranking (`ftm.dsl.expert.expert_ranking`, the family order of `configs/questions/expert_specs.yaml`) at one eps level |
| `EIGPlanner` | largest expected Brier reduction per unit eps (below) |
| `LearnedPolicyPlanner` | largest predicted Brier gain per unit eps; stops when nothing is predicted to help (below) |
| `EnumeratorPlanner` | EIG over every DSL program up to size n (`ftm/dsl/enumerate.py`), each back-tested like the expert list: `scripts/backtest.py --questions enumerate --max-size 4` |

**Expected information gain.** An answer cell `a` has probability `P(a) = p P(a|launder) + (1-p) P(a|clean)` (measured on the *noisy* back-test answers, `p_cell` in the table) and moves the posterior to `p'(a) = sigmoid(logit p + log LR(a))`, the shift fusion applies. A calibrated posterior `q` has expected Brier `q(1-q)`, so `gain = p(1-p) - sum_a P(a) p'(a)(1-p'(a))`, and EIG picks `max gain / eps`. It stops when no remaining question has a positive expected gain. A test checks that with exact LRs the expected posterior equals the prior (a martingale).

**Shared rules.** Stop rule: T4's `should_stop` (budget, and `cannot_cross` when a `threshold` is given). Posterior: T4's `posterior`. A family is asked at most once per alert by default (two questions of one template are not independent evidence). A refused question is logged and not retried. Notes in replies are never read.

**What a planner needs from an alert** (`alert.summary`, aliased): `direction` (`out` if the home bank sent the money) and `counterparty_bank`. `ftm.banks.binding.alert_context` builds them. For T1/T3: please put these two keys in the alert contexts you create.

**Run.**

```
python scripts/run_experiment.py --system eig --synthetic 360                    # smoke test, never a result
python scripts/run_experiment.py --system eig --data hi_small --split val --table data/backtest/hi_small/<ns>/lr_table.jsonl
```

writes `outputs/runs/<id>/runlog.parquet` (run-log schema) and `summary.json` (Brier before and after, questions and eps per alert).

**Synthetic check** (`tests/test_planners.py`, 5 questions back-tested on 60% of a synthetic world, run on the rest): all three planners cut the validation Brier by more than 0.02 from the base rate inside the eps budget; the enumerator runs on the enumerated search space; runs are bit-reproducible.

**Learned policy (week 9).** `ftm/planners/learned.py`, data in `ftm/orchestrator/policy_data.py` (the part that asks banks, so it is outside `ftm/planners`), `scripts/train_policy.py`. The validation alerts are cut in time (`split_val`, first half `fit`, rest `eval`; the same cut every planner is scored on in `scripts/run_ablations.py`). On `fit`, random rollouts of up to 3 questions; at each state every affordable question (up to 24) is asked of the partner bank and the row's label is the Brier gain the answer really gave, `(p-y)^2 - (p'-y)^2`. Features: state (logit p, p0, eps left, questions asked), alert (direction, amount in half-decades, partner bank), question (eps, kind, family, sensitivity, cells, largest |log LR|) and EIG's closed-form expected gain. The model is LightGBM boosted on the residual over EIG (`init_score`), with the number of rounds chosen by cross-validation over whole alerts, so with little signal it stops early and **is** the EIG planner. `train_policy.py` prints the held-out rank correlation of its prediction and of plain EIG with the realised gain, so "the learned policy beats EIG" is something the held-out `eval` half shows, not something assumed. Questions asked to build the rows are charged to eps_cal (printed). Needs `pip install -e ".[gbt]"` (LightGBM); without it the ablation runner lists the arm as skipped.

```
python scripts/train_policy.py --data hi_medium --split val --table data/backtest/hi_medium/<ns>/lr_table.jsonl --out outputs/policy/hi_medium.json
python scripts/run_experiment.py --system learned --policy outputs/policy/hi_medium.json --data hi_medium --split val --table <same table>
```

**Not here yet.** The real-data runs (G3, G4: they need T1's data on the machine). On the synthetic world the 7-question pool gives the learned policy nothing to learn beyond EIG: its rounds stop at 1 and it matches EIG, which is the design working, not a result.
