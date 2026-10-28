# ftm/orchestrator: running systems over alerts (C15), the small-decision set, the raw-sharing arm

Owner: T2 (Bhavya). Spec: `tracks.md` 4.8 and 4.9.

| File | What |
|---|---|
| `runner.py` | Sequential orchestrator for the non-LLM arms: build a planner, run it over alerts in fresh harness sessions, score, write run-log rows |
| `decision_set.py`, `scripts/make_decision_set.py` | **Small-decision dataset v1 (hand-off #13, T2 to T3)** |
| `ablations.py`, `scripts/run_ablations.py`, `configs/experiments/ablations_nonllm.yaml` | **The G4 table for T2**: every non-LLM planner and ablation switch on validation |
| `policy_data.py` | Training rows for the learned policy (`ftm/planners/NONLLM.md`) |
| `raw_share.py` | **The raw-sharing baseline arm (H2)**: the one place where rows leave a bank |

## Small-decision dataset (hand-off #13)

```
python scripts/make_decision_set.py --data hi_medium --table data/backtest/hi_medium/<ns>/lr_table.jsonl --threshold 0.3 --max-total 50000
python scripts/make_decision_set.py --synthetic 300 --out <dir>          # smoke test, never a result
```

Writes `data/decisions/<ds>/decisions_{train,val}.jsonl` and `manifest.json` (counts, label balance, sha256). Rows are `{decision, context, label, alert_id, split}`, the format of T3's `ftm/fastmodel/decisions.py`; the label sets are in `DECISION_LABELS` and a test pins them. 30 decisions per alert by default (`DEFAULT_MIX`), about 60,000 on HI-Medium train and val, capped by `--max-total`. Not committed (README rule 7): share the files and the manifest through the cloud folder.

| Decision | Label comes from | Contexts |
|---|---|---|
| `worth_asking` | the alert label (the alerted transfer is in a real laundering attempt) | alert summary and what the home bank saw about the counterparty in the last 24 / 48 / 72 h, with and without p0 |
| `typology` | the pattern file's typology of the attempt, `none` for a clean alert; laundering alerts with no typology are skipped | same evidence |
| `rule_applies` | T4's gate rules: yes iff the cited rule permits `ask_bank` in the home jurisdiction for the answer kind | a candidate expert question and a permitting or non-permitting citation (US and SG) |
| `stop` | yes iff no remaining back-tested question has an answer that could move the posterior across the decision threshold (or there is no eps for one) | the answers so far, produced by the simulated banks, with posterior, threshold, eps left |
| `injection_present` | T4's injection generator (`ftm/safety/injection.py`): 4 attack types and benign notes | a partner-bank reply with the note |

Contexts are aggregate text: counts, amounts, public bank ids. No account id, alias, name or date (tested against the gate's own literal detectors); evidence is cut at the alert time (tested); worth_asking and typology contexts are identical when the label is flipped (tested), so a context cannot encode its label. Test-split alerts raise an error.

## Raw-sharing arm (H2)

`raw_share.py`: three banks hand over raw records (the bank holding the alert's partner-side account, then the two banks it dealt with most, for the first account's neighbours), the home bank computes every expert question **exactly** on them plus second-hop features, and a ridge logistic model on those and logit(p0), fitted on training alerts, gives the probability. Unlimited eps, no noise, any feature: what full sharing buys. `gain_ratio(brier_p0, brier_questions, brier_raw)` is H2's ratio.

It is gated twice. Every hand-over asks the compliance gate for `request_records`, which the production gate blocks (`allow_raw_share: false`); only a gate built with `allow_raw_share=True` allows it, and the harness toolbox has no such tool. And only this module defines or calls `share_records` (a test greps the package), so `ftm/banks` still has no code path that returns rows.

## Per-account cap

`ftm/privacy/cap_sweep.py` and `scripts/cap_sweep.py`: how often one partner account is the subject of several alerts, and the refused share of questions per cap. Run it on the real alert sets to settle `per_account_cap` (currently 8.0): `python scripts/cap_sweep.py --data hi_medium --split val`.

## eps sweep (E2)

`eps_sweep.py`, `scripts/eps_sweep.py`: one system at per-alert budgets 0.5, 1, 2, 4 and no noise on the same validation alerts through the same harness. The no-noise point (eps = inf) answers exactly from a pool back-tested the same way; it is an explicit opt-in of the harness (`Harness.allow_exact`): the gate still checks everything but the eps level and nothing is priced. Every row carries the theoretical ceiling (basic composition, unit sensitivity: `B` eps can move the log-odds by at most `B`, the LR by at most `e^B`), the line to draw on the privacy plot.

```
python scripts/eps_sweep.py --system eig --synthetic 360                       # smoke test, never a result
python scripts/eps_sweep.py --system eig --data hi_medium --split val --table-noisy <lr_table.jsonl> --table-exact <lr_table_exact.jsonl>
```

## Async mode: question waves and the SimPy runner (week 8, E13 / H8)

* **Waves.** `HarnessToolBox.ask_banks(requests)` asks several questions at the same simulated moment. Each goes through the gate and the accountant on its own (a refusal is returned in its slot, not raised, so it does not lose the others); the wave costs its **slowest answer**, not the sum, the rule T3's LLMCompiler port also uses, so E13's two arms are comparable. It is an extension (`BatchToolBox` in `ftm/harness/interface.py`); the frozen v1 `ToolBox` is unchanged.
* **Batched planners.** `waves=(2, 1)` on any non-LLM planner: two questions asked together, then one more (never two of one family, all fitting the eps left together). `build_planner(..., waves=...)`.
* **SimPy runner** (`simpy_run.py`): one investigation = one SimPy process on a shared clock; alert i arrives at `i * arrival_gap_days`. Reports each alert's latency, the makespan and the most investigations open at once. Gate and accountant are touched in arrival order, so the per-account cap sees alerts as it would live.
* **Study** (`async_study.py`): the same planner, pool, budget and validation alerts, sequential vs batched: Brier, questions and eps per alert, simulated days per alert (mean, p95), makespan, investigations in flight. The run summaries now carry `mean_sim_days` and `p95_sim_days`.

## Non-LLM planners and ablations on validation (week 9, G4)

`scripts/run_ablations.py` runs every arm of `configs/experiments/ablations_nonllm.yaml` through the same harness, gate and accountant on the same alerts: the validation alerts are cut in time, the first half trains what needs training (the learned policy, the fitted fusions), **every arm is scored on the second half**. One row per arm: Brier before and after, the gain with a cluster-bootstrap 95% CI (T4's `ftm.eval.bootstrap`, clusters = alerts sharing an account), the paired Brier difference to the reference arm (`eig`) with its CI, questions and eps per alert, the share of questions that got no reply, simulated days.

| Group | Arms |
|---|---|
| planners (Table 5b, H3) | random, fixed order, EIG, learned policy, enumerator |
| budget (E2) | EIG at eps 0.5 / 1 / 4 per alert (the full sweep with the no-noise point is `scripts/eps_sweep.py`) |
| cap (E14) | per-account cap 8 (production) and 2, against off |
| questions | a template asked more than once per alert |
| stop | the cannot-cross stop rule at threshold 0.3 |
| async (E13) | waves (2, 1) |
| robustness (E14) | 20% and 50% declines, 20% timeouts |
| fusion | summed log-LRs vs T4's shared and per-family fusion, fitted on the `fit` half |
| backtest (E4) | LRs from the exact-label back-test, questions answered noisily |

```
python scripts/run_ablations.py --synthetic 600                      # smoke test, never a result
python scripts/run_ablations.py --data hi_medium --split val --table <lr_table.jsonl>     --table-enumerator <lr_table_enum.jsonl> --table-exact-labels <lr_table_exact.jsonl> --out outputs/g4
```

Writes `ablations.json` and `ablations.md`. An arm whose input is missing (no enumerated table, no exact-label table, no LightGBM) is listed as **skipped with the reason**, never dropped. Adding an arm is a line in the YAML; unknown settings raise.

**Reading it.** Differences between arms on a small validation set are mostly inside the CIs; do not rank arms whose paired interval contains 0. On the synthetic world the pool has 7 questions in 5 families and EIG stops after about one, so the cap, repeat-family, stop-rule and wave arms come out identical to EIG there: the tests check the switches are wired, the real validation run is what can separate them.

**G4 from T2's side.** Done: all non-LLM planners and the ablation configs run end to end on the synthetic world and are tested. **Not done: the run on real validation** (needs `data/`, T1's alert sets and the three back-test tables), so the G4 numbers do not exist yet; the compute check needs the LLM arms' GPU hours from T3.

## Async runs (week 10, E13 / H8, stretch)

`scripts/async_study.py`: for each non-LLM planner and per-alert budget, one question at a time vs waves of (2, 1) on the eval half of the validation alerts (the same alerts as the G4 table), the SimPy makespan, and **`share_2plus`**, the share of alerts that asked two or more questions at all. That column matters: a wave only holds two questions if two fit the eps left. T3's LLM run (`docs/t3_e13.md`) found the (2, 1) schedule never reached its second slot at eps 2, because every accepted question cost eps >= 1. The same holds here: on the synthetic world at budget 2 only 3 to 9% of alerts ask two questions, so the two arms are identical; at budget 4 every alert does and waves take about 75% of the sequential simulated time. `h8` rows give the share of the sequential Brier gain the batched arm keeps and its share of the time (the claim is 80% in 60%), and read `null` when nothing was batched or the sequential arm gained nothing. `load` rows put the same alerts on one SimPy clock at several arrival gaps (makespan, investigations open at once, mean latency).

```
python scripts/async_study.py --synthetic 600 --systems random eig --budgets 2 4          # smoke test, never a result
python scripts/async_study.py --data hi_medium --split val --table <lr_table.jsonl> --systems random fixed eig learned --policy outputs/policy/hi_medium.json --budgets 2 4 --out outputs/async
```

A real E13 row needs accepted questions at low eps in the back-test table (or a larger per-alert budget for both arms), as T3 noted. Not run on real data (no `data/`).

## The final run (weeks 12-13, G5)

`scripts/final_non_llm.py` is the command behind T2's jobs of the G5 plan (`nonllm.<arm>.<subset|full_test>` and `raw_share.raw_share.*`): it builds an arm with `ablations.prepare_arm`, the function the G4 table uses, and runs it blind (`run_alerts(..., blind=True)`: no label in any row, no Brier in the summary) on the frozen ids, writing a run-log v1 file. Fits (fusion, learned policy) stay on the `fit` half of validation; question tables come from `configs/final/non_llm.yaml` and are not rebuilt. Alerts with no modelled partner bank get a verdict row without probability (scored as p0). Full description, decisions and the rehearsal on validation: `docs/t2_final_run.md`. Tested on the synthetic world only (`tests/test_final_non_llm.py`); the real tables and numbers need `data/`.

