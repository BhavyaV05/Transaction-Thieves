# T3 validation analysis: why no planner mode beats p0, and what a replay says

Owner: T3. Script: `scripts/t3_val_analysis.py` (CPU only, no LLM calls, seeded cluster bootstrap with 2,000 resamples; prints JSON and writes `outputs/t3_val/analysis/{analysis.json,tables.md}`). Tests: `tests/test_t3_val_analysis.py`. The tables below are a snapshot of `tables.md`; rerun the script to regenerate them.

Data: val `eval` runs `outputs/t3_val/eval/<spec>/c000,c001` (160 askable alerts, Gemma 4 12B, realistic back-test table `data/backtest/hi_medium/7df6bad5337d/lr_table.jsonl`, 23 of 440 entries accepted), and the tuning runs `outputs/t3_val/fitcal/fit/c000,c001` (191 alerts) plus `outputs/t3_val/fitcal/cal/c000` (109 alerts), compose mode. CIs are paired attempt+accounts cluster bootstrap (`outputs/e18/clusters.json`, same loaders and bootstrap as `scripts/val_runs_table.py`). Brier of p0 alone: 0.1985 on eval, 0.1554 on fit+cal. The teacher runs (`outputs/t3_teacher/runs`) are on the train split, where the back-test LRs were fitted, so their answers are in-sample for the LRs; they are not used.

**Protocol.** Every fitted quantity (shrinkage, caps, abstain threshold, dropped families, recalibration) comes from `fit`+`cal` only. The variants in Table 8 were fixed in advance (the list a to f in the task); eval is scored once for each. No variant was picked by its eval number. Test split not touched.

## 1. What is failing (summary)

1. **The answers carry no usable information beyond p0, and the naive fusion adds them with full weight.** Over the alerts where the posterior moved, the slope of `logit p0 + lambda * sum(log LR)` is lambda_hat = +0.01 [-0.97, +0.94] on fit+cal compose, -0.10 [-1.54, +1.28] on eval compose, -0.20 [-0.87, +0.46] on eval fixed_list, -0.39 [-1.11, +0.27] on eval act_only (Table 5). The claimed value is 1. Zero fits every interval; 1 is excluded for act_only and fixed_list. At answer level, the labels after an answer claimed "more suspicious" and after one claimed "less suspicious" are the same (mean of label minus posterior: -0.104 versus -0.098 on fixed_list, -0.141 versus -0.061 on act_only; Table 3b). A mean-zero update with full weight costs Brier, and that is the whole loss.
2. **The loss is where the posterior moves, not where the planner asks.** Compose moves only 52 of 160 alerts (0.28 updating answers per alert) because 217 of its 264 questions (82%) are either not in the back-test table (132) or in it but rejected (85), and a rejected or missing entry carries no LR (checked: 0 updating steps are unaccepted; Table 0 and checks in `analysis.json`). Fixed_list and act_only ask only or mostly accepted questions, move 158 and 128 alerts, and lose 4 to 6 times more (+0.0161, +0.0115). Compose's small loss is mostly because it rarely updates.
3. **The harm sits at eps 2 and in volume_spike / new_account, not at low eps.** The table accepts nothing at eps 0.25 and 0.5, 5 entries at eps 1 (largest |log LR| 0.75) and 18 at eps 2 (largest 1.34). Fixed_list: eps 2 answers contribute +0.0150 of the +0.0161, eps 1 +0.0011. The eps 2 answers are the least noisy ones, so DP noise is not what hurts. The back-test LRs do not transfer to val: volume_spike answers point the right way in only 4 of 22 (0.18) fixed_list cases (+0.0103), new_account +0.0054. This matches T4's shift / double-counting hypotheses (`docs/t4_fusion_findings.md`); this analysis cannot separate them.
4. **Timeouts, declines and refusals are harmless by construction.** They never update (0 of 23 non-replies in compose eval and 0 of 51 in fit+cal carry a log-LR). Variant (f) is therefore a no-op.
5. **The loss is moderately concentrated in compose.** The five worst alerts are 42% of the sum of per-alert losses, and compose without them is -0.0025 (Table 4). In fixed_list and act_only it is spread (top 5 hold 19% and 17%): there it is a systematic shift, not outliers.
6. **p0 itself is miscalibrated on eval, which makes every upward update more expensive.** Eval label rate is 0.325 against mean p0 0.455 (fit+cal: 0.520 against 0.545). In the 0.5 to 0.9 range p0 overstates by 0.2 to 0.3 on eval (Table 7). Alerts with p0 >= 0.99 (saturated, the logit clips at 1e-6) have a label rate of 0.90 on fit+cal. Recalibrating p0 alone is a different thing from planner merit and is reported separately in Table 9.
7. **No replay variant beats p0 through the answers.** The best in-sample (fit+cal) gain over p0 is -0.0006 (CI includes 0). On eval the fitted variants shrink to p0 and tie it: compose shrinkage +0.0001 [-0.0005, +0.0008]. Two variants aimed at the root cause also fail. (g) LRs re-estimated from the fit+cal answers (including the questions the live run discarded), smoothing chosen by grouped CV: it selects infinite smoothing, i.e. p0 (CV Brier 0.1554 for every alpha >= 100, 0.1624 at alpha 0.5). (h) A stacked logistic model on logit(p0) plus family answer features beats p0 on eval (-0.0110 [-0.0194, -0.0026]), but only through its recalibration part: against the model with logit(p0) alone it adds +0.0001 [-0.0011, +0.0011] on compose and is slightly worse on fixed_list, act_only and llm_decides (+0.0006 to +0.0013, CIs above 0). The answers add nothing on top of p0 in this data: not with the back-test LRs, not with re-estimated LRs, not in a fitted model.
8. **ReAct, Reflexion and UoT are bad for a different reason**: they do not use p0 as a prior (ReAct and Reflexion never see it) and output extreme or default probabilities (section 4).

## 2. Evidence: failure decomposition

Definitions: dBrier of an alert = (p_final - y)^2 - (p0 - y)^2 (positive = worse than p0). "Contribution" of a group = sum of its dBrier / all alerts in the run, so groups add to the total. For the fusion modes the step attribution telescopes: step dBrier = (posterior after - y)^2 - (posterior before - y)^2.

### 2.1 Overall per mode

**Table 1. Each mode on val `eval` (160 alerts) and compose on `fit`+`cal` (300 alerts)**
| mode | Brier | vs p0 [95% CI] | alerts moved | questions/alert | updating answers/alert | alerts with >=1 update |
|---|---|---|---|---|---|---|
| ours_compose | 0.2014 | +0.0029 [-0.0035, +0.0098] | 52 | 1.65 | 0.28 | 45 |
| ours_fixed_list | 0.2146 | +0.0161 [+0.0005, +0.0323] | 158 | 1.42 | 1.35 | 158 |
| ours_act_only | 0.2100 | +0.0115 [+0.0003, +0.0231] | 128 | 1.61 | 0.84 | 125 |
| ours_batched_2_1 | 0.2219 | +0.0233 [+0.0119, +0.0351] | 145 | 1.43 | 0.89 | 142 |
| ours_library_account | 0.2048 | +0.0063 [-0.0006, +0.0138] | 49 | 1.71 | 0.27 | 43 |
| ours_llm_decides | 0.1996 | +0.0011 [-0.0027, +0.0051] | 144 | 1.63 | 0.31 | 49 |
| ours_no_backtest | 0.2480 | +0.0494 [-0.0024, +0.0959] | 160 | 3.85 | 0.00 | 0 |
| fit+cal ours_compose (tuning) | 0.1574 | +0.0020 [-0.0017, +0.0062] | 93 | 1.82 | 0.22 | 62 |


Back-test table by eps:

**Table 0. Back-test table entries by eps**
| eps | (question, eps) entries | accepted | max |log LR| among accepted |
|---|---|---|---|
| 0.25 | 110 | 0 | 0.00 |
| 0.5 | 110 | 0 | 0.00 |
| 1 | 110 | 5 | 0.75 |
| 2 | 110 | 18 | 1.34 |


Question-level accounting on eval compose (`analysis.json`, key `checks`): 264 questions asked, 238 answered, 47 on accepted entries (45 answered and updating), 85 on rejected entries, 132 on programs not in the table (composed questions without any back-tested LR). On fit+cal: 547 questions, 65 updating, 248 rejected, 228 not in the table.

### 2.2 By label, p0 bucket and number of questions

(Table 2, excerpts for eval compose, fit+cal compose and eval fixed_list; the full table is in `outputs/t3_val/analysis/tables.md`.)

**eval ours_compose, by label**

| label | alerts | moved | dBrier contribution | mean dBrier per alert | mean shift toward the label |
|---|---|---|---|---|---|
| y=0 | 108 | 29 | +0.0038 | +0.0056 | -0.0077 |
| y=1 | 52 | 23 | -0.0009 | -0.0026 | +0.0053 |

**fit+cal ours_compose, by label**

| label | alerts | moved | dBrier contribution | mean dBrier per alert | mean shift toward the label |
|---|---|---|---|---|---|
| y=0 | 144 | 37 | +0.0028 | +0.0059 | -0.0023 |
| y=1 | 156 | 56 | -0.0009 | -0.0017 | +0.0034 |

**eval ours_fixed_list, by label**

| label | alerts | moved | dBrier contribution | mean dBrier per alert | mean shift toward the label |
|---|---|---|---|---|---|
| y=0 | 108 | 106 | +0.0190 | +0.0281 | -0.0270 |
| y=1 | 52 | 52 | -0.0029 | -0.0089 | +0.0151 |


Reading: in compose the loss is on label 0 (+0.0038) and low p0 (< 0.5: +0.0039); label 1 gains slightly (-0.0009). Updates move label-0 alerts up on average (-0.0077 toward the label, i.e. away) and label-1 alerts toward their label by only +0.0053. In fixed_list the same pattern is larger (label 0: +0.0190; p0 0.5 to 0.8: +0.0105). Alerts with p0 >= 0.99 cannot move (logit clip), so they contribute nothing. `stop_reason` is empty in every verdict of these runs, so the split by stop reason is not possible; number of questions is shown instead (more questions did not help: 3+ questions are 10 alerts in compose, too few to read).

### 2.3 By response, back-test status, eps, family (step attribution)

(Table 3, excerpts.)

**eval ours_compose**

| response | steps | updating | dBrier contribution | mean claimed log-LR | direction hit |
|---|---|---|---|---|---|
| decline | 17 | 0 | +0.0000 |  |  |
| over_budget | 3 | 0 | +0.0000 |  |  |
| reply | 238 | 45 | +0.0029 | +0.07 | 0.44 |
| timeout | 6 | 0 | +0.0000 |  |  |

| in back-test table / accepted | steps | updating | dBrier contribution | mean claimed log-LR | direction hit |
|---|---|---|---|---|---|
| accepted | 47 | 45 | +0.0029 | +0.07 | 0.44 |
| in table, rejected | 85 | 0 | -0.0000 |  |  |
| not in table | 132 | 0 | -0.0000 |  |  |

| eps | steps | updating | dBrier contribution | mean claimed log-LR | direction hit |
|---|---|---|---|---|---|
| 1 | 185 | 22 | +0.0002 | -0.07 | 0.55 |
| 2 | 79 | 23 | +0.0027 | +0.21 | 0.35 |

| family (updating or asked) | steps | updating | dBrier contribution | mean claimed log-LR | direction hit |
|---|---|---|---|---|---|
| cross_currency_out | 1 | 1 | +0.0001 | -0.25 | 0.00 |
| dormant | 5 | 3 | +0.0014 | -0.33 | 0.00 |
| inflow_bucket | 11 | 1 | -0.0006 | -0.40 | 1.00 |
| n_banks:in | 2 | 2 | +0.0005 | +0.61 | 0.00 |
| n_repeated | 4 | 1 | -0.0006 | -0.43 | 1.00 |
| new_account | 29 | 23 | +0.0002 | -0.08 | 0.57 |
| volume_spike | 16 | 14 | +0.0019 | +0.42 | 0.36 |

**eval ours_fixed_list**

| response | steps | updating | dBrier contribution | mean claimed log-LR | direction hit |
|---|---|---|---|---|---|
| decline | 8 | 0 | +0.0000 |  |  |
| reply | 216 | 216 | +0.0161 | +0.06 | 0.47 |
| timeout | 3 | 0 | +0.0000 |  |  |

| in back-test table / accepted | steps | updating | dBrier contribution | mean claimed log-LR | direction hit |
|---|---|---|---|---|---|
| accepted | 227 | 216 | +0.0161 | +0.06 | 0.47 |

| eps | steps | updating | dBrier contribution | mean claimed log-LR | direction hit |
|---|---|---|---|---|---|
| 1 | 134 | 127 | +0.0011 | -0.10 | 0.54 |
| 2 | 93 | 89 | +0.0150 | +0.30 | 0.38 |

| family (updating or asked) | steps | updating | dBrier contribution | mean claimed log-LR | direction hit |
|---|---|---|---|---|---|
| cross_currency_out | 4 | 4 | +0.0003 | -0.25 | 0.25 |
| dormant | 54 | 49 | -0.0032 | -0.30 | 0.61 |
| inflow_bucket | 3 | 3 | -0.0000 | -0.07 | 0.33 |
| n_banks:in | 22 | 22 | +0.0024 | +0.37 | 0.36 |
| n_new_counterparties | 6 | 6 | +0.0002 | +0.03 | 0.50 |
| n_repeated | 24 | 22 | +0.0007 | +0.34 | 0.55 |
| new_account | 91 | 88 | +0.0054 | +0.06 | 0.49 |
| volume_spike | 23 | 22 | +0.0103 | +0.40 | 0.18 |

**fit+cal ours_compose**

| response | steps | updating | dBrier contribution | mean claimed log-LR | direction hit |
|---|---|---|---|---|---|
| decline | 34 | 0 | +0.0000 |  |  |
| over_budget | 5 | 0 | +0.0000 |  |  |
| reply | 491 | 65 | +0.0020 | +0.04 | 0.54 |
| timeout | 17 | 0 | +0.0000 |  |  |

| in back-test table / accepted | steps | updating | dBrier contribution | mean claimed log-LR | direction hit |
|---|---|---|---|---|---|
| accepted | 71 | 65 | +0.0020 | +0.04 | 0.54 |
| in table, rejected | 248 | 0 | -0.0000 |  |  |
| not in table | 228 | 0 | -0.0000 |  |  |

| eps | steps | updating | dBrier contribution | mean claimed log-LR | direction hit |
|---|---|---|---|---|---|
| 1 | 458 | 38 | +0.0007 | -0.09 | 0.63 |
| 2 | 89 | 27 | +0.0013 | +0.23 | 0.41 |

| family (updating or asked) | steps | updating | dBrier contribution | mean claimed log-LR | direction hit |
|---|---|---|---|---|---|
| dormant | 7 | 5 | +0.0001 | +0.25 | 0.80 |
| inflow_bucket | 16 | 6 | -0.0000 | -0.34 | 0.17 |
| n_banks:in | 6 | 4 | -0.0003 | +0.61 | 0.75 |
| n_repeated | 7 | 5 | +0.0008 | +0.67 | 0.20 |
| new_account | 53 | 35 | +0.0017 | -0.13 | 0.57 |
| volume_spike | 13 | 10 | -0.0003 | +0.22 | 0.60 |


### 2.4 Direction of the answers

**Table 3b. Sign check per updating answer: mean(label - posterior before) when the claimed log-LR is positive / negative (correct independent LRs give > 0 / < 0)**
| run | eps | answers claimed up | mean(y - p) up | answers claimed down | mean(y - p) down |
|---|---|---|---|---|---|
| fit+cal ours_compose | all | 37 | -0.049 | 28 | -0.051 |
| fit+cal ours_compose | eps 1 | 20 | -0.083 | 18 | -0.103 |
| fit+cal ours_compose | eps 2 | 17 | -0.008 | 10 | +0.044 |
| eval ours_compose | all | 28 | -0.083 | 17 | -0.113 |
| eval ours_compose | eps 1 | 12 | +0.003 | 10 | -0.078 |
| eval ours_compose | eps 2 | 16 | -0.147 | 7 | -0.164 |
| eval ours_fixed_list | all | 121 | -0.104 | 95 | -0.098 |
| eval ours_fixed_list | eps 1 | 56 | -0.014 | 71 | -0.051 |
| eval ours_fixed_list | eps 2 | 65 | -0.181 | 24 | -0.236 |
| eval ours_act_only | all | 83 | -0.141 | 52 | -0.061 |
| eval ours_act_only | eps 1 | 49 | -0.014 | 42 | -0.099 |
| eval ours_act_only | eps 2 | 34 | -0.323 | 10 | +0.097 |


### 2.5 Concentration

**Table 4. Concentration of the loss (alerts whose posterior moved)**
| run | moved | worse | better | sum worse / N | sum better / N | total | top-5 share of worse | total without worst 5 |
|---|---|---|---|---|---|---|---|---|
| eval ours_compose | 52 | 31 | 21 | +0.0128 | -0.0099 | +0.0029 | 0.42 | -0.0025 |
| fit+cal ours_compose | 91 | 57 | 34 | +0.0073 | -0.0053 | +0.0020 | 0.41 | -0.0010 |
| eval ours_fixed_list | 152 | 87 | 65 | +0.0441 | -0.0280 | +0.0161 | 0.19 | +0.0079 |
| eval ours_act_only | 124 | 70 | 54 | +0.0346 | -0.0231 | +0.0115 | 0.17 | +0.0056 |
| eval ours_llm_decides | 144 | 87 | 57 | +0.0056 | -0.0045 | +0.0011 | 0.82 | -0.0036 |

Worst five alerts, eval ours_compose:

| alert | label | p0 | p final | dBrier |
|---|---|---|---|---|
| V001833 | 1 | 0.3635 | 0.2126 | +0.2149 |
| V001549 | 1 | 0.4413 | 0.28 | +0.2063 |
| V001576 | 1 | 0.6391 | 0.4556 | +0.1661 |
| V001970 | 0 | 0.6499 | 0.75 | +0.1402 |
| V001617 | 0 | 0.5397 | 0.6546 | +0.1373 |


### 2.6 Calibration of the summed LRs

**Table 5. Do the realised summed log-LRs predict the labels as claimed? (slope lambda of logit(p0) + lambda * sum; 1 = as claimed, 0 = no information)**
| run | alerts with update | lambda_hat [95% cluster CI] | mean |sum log-LR| | alerts sum>0: n, label rate, mean p0 | alerts sum<0: n, label rate, mean p0 |
|---|---|---|---|---|---|
| fit+cal compose | 62 | +0.01 [-0.97, +0.94] | 0.56 | 35, 0.49, 0.55 | 27, 0.44, 0.48 |
| eval compose | 45 | -0.10 [-1.54, +1.28] | 0.56 | 28, 0.36, 0.44 | 17, 0.41, 0.52 |
| eval fixed_list | 158 | -0.20 [-0.87, +0.46] | 0.55 | 104, 0.33, 0.45 | 54, 0.33, 0.47 |
| eval act_only | 125 | -0.39 [-1.11, +0.27] | 0.57 | 77, 0.34, 0.49 | 48, 0.35, 0.42 |


Per family on fit+cal (the data any family selection would use; 4 to 35 answers each, so every row is noisy):

**Table 6. Updating families on `fit`+`cal` (compose): answers used, claimed mean log-LR, share whose sign matches the label, and Brier change if the family is dropped**
| family | used answers | mean claimed log-LR | direction hit | dBrier if dropped |
|---|---|---|---|---|
| dormant | 5 | +0.25 | 0.80 | -0.0001 |
| inflow_bucket | 6 | -0.34 | 0.17 | +0.0000 |
| n_banks:in | 4 | +0.61 | 0.75 | +0.0003 |
| n_repeated | 5 | +0.67 | 0.20 | -0.0008 |
| new_account | 35 | -0.13 | 0.57 | -0.0017 |
| volume_spike | 10 | +0.22 | 0.60 | +0.0003 |


### 2.7 Reliability of p0

**Table 7. Reliability of p0 on the askable alerts**
**fit+cal (300 alerts)**

| p0 bin | alerts | mean p0 | label rate |
|---|---|---|---|
| 0.1-0.3 | 98 | 0.186 | 0.122 |
| 0.3-0.5 | 48 | 0.409 | 0.438 |
| 0.5-0.7 | 45 | 0.577 | 0.644 |
| 0.7-0.9 | 41 | 0.804 | 0.805 |
| 0.9-0.99 | 28 | 0.955 | 0.893 |
| 0.99-1 | 40 | 0.999 | 0.900 |

**eval (160 alerts)**

| p0 bin | alerts | mean p0 | label rate |
|---|---|---|---|
| 0.1-0.3 | 71 | 0.195 | 0.155 |
| 0.3-0.5 | 28 | 0.387 | 0.286 |
| 0.5-0.7 | 21 | 0.602 | 0.286 |
| 0.7-0.9 | 20 | 0.800 | 0.550 |
| 0.9-0.99 | 12 | 0.950 | 0.750 |
| 0.99-1 | 8 | 1.000 | 0.875 |


## 3. Offline replay

The replay takes the logged answers and logged log-LRs of each alert and recomputes the verdict as `sigmoid(logit p0 + lambda * sum(clip(log LR)))` with the variant's rule (`replay_prob` in the script). With no change it reproduces every logged verdict to within 1e-6 (the logit clip), for compose, fixed_list, act_only, batched_2_1, library_account and fit+cal.

**Limit of the replay.** The questions asked and the stopping point are those of the live run. In every mode except no_backtest the planner prompt shows the current posterior and tags each past answer `[no back-tested LR: not used]` (`ftm/planners/llm_prompt.py`, `shows_posterior`; `ftm/planners/llm.py` builds the state from `session.posterior` before every step). So the model's choice of the next question, of eps, and of when to stop can depend on the posterior in compose, act_only, fixed_list (it selects from the list with the posterior shown), batched_2_1 and library_account, and llm_decides. A live run with a shrunk or capped posterior would show a different number to the model and could ask different questions. The replay is therefore an approximation, most reliable for fixed_list-like cases where the same few questions are asked regardless. For llm_decides and no_backtest the logged verdict is the model's own belief; the replay rows for them score the formula posterior over the questions they asked, a different system from the logged one.

Variants (all pre-declared; parameters from fit+cal compose, grids in the script): (a) lambda shrinkage, grid 0 to 1.5 step 0.05; (b) accepted entries only (the live rule, so a no-op), and (b') the opposite: also use unaccepted table entries; (c) drop harmful families by greedy backward elimination on fit+cal Brier (families with >= 3 answers); (d) cap per answer and per alert, joint grid; (e) abstain unless |sum| > tau, tau grid 0 to 3 step 0.1; (f) no update on timeout or refusal (no-op, see summary point 4).

**Table 8. Replay variants, parameters fitted on `fit`+`cal` (compose, 300 alerts), scored once on eval**
Fit/cal Brier column: in-sample for a to f; grouped 10-fold CV (whole clusters) for g and h, whose strength was chosen by that CV.

| variant | chosen on fit/cal | fit/cal Brier (p0 = 0.1554) | eval compose Brier | eval compose vs p0 [CI] | eval llm_decides formula shadow vs p0 [CI] |
|---|---|---|---|---|---|
| identity (live posterior) |  | 0.1574 | 0.2014 | +0.0029 [-0.0035, +0.0098] | +0.0101 [+0.0036, +0.0176] |
| (a) shrink lambda | {"lam": 0.1} | 0.1553 | 0.1986 | +0.0001 [-0.0005, +0.0008] | +0.0007 [+0.0000, +0.0015] |
| (b) accepted entries only [live rule; no-op] |  | 0.1574 | 0.2014 | +0.0029 [-0.0035, +0.0098] | +0.0101 [+0.0036, +0.0176] |
| (b') ALL table entries incl. unaccepted, lambda=1 |  | 0.1587 | 0.2000 | +0.0014 [-0.0053, +0.0087] | +0.0072 [-0.0002, +0.0157] |
| (b'') all table entries, shrink lambda' | {"lam": 0.0} | 0.1554 | 0.1985 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| (c) drop harmful families | {"dropped": ["dormant", "n_repeated", "new_account"]} | 0.1548 | 0.2004 | +0.0019 [-0.0026, +0.0066] | +0.0066 [+0.0022, +0.0118] |
| (d) cap per answer / per alert | {"cap_answer": 0.1, "cap_total": 0.25} | 0.1554 | 0.1986 | +0.0001 [-0.0012, +0.0013] | +0.0013 [+0.0001, +0.0027] |
| (e) abstain unless |sum| > tau | {"tau": 0.9} | 0.1550 | 0.1985 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| (f) no update on timeout/refusal [no-op] |  | 0.1574 | 0.2014 | +0.0029 [-0.0035, +0.0098] | +0.0101 [+0.0036, +0.0176] |
| (g) re-estimated LRs on fit/cal answers, smoothed | {"alpha": 1000000.0, "cells": 44} | 0.1554 | 0.1985 | +0.0000 [-0.0000, +0.0000] | +0.0000 [-0.0000, +0.0000] |
| (h) stacking: logit(p0) + family answer features | {"ridge": 30, "families": 8} | 0.1601 | 0.1875 | -0.0110 [-0.0194, -0.0026] | -0.0101 [-0.0185, -0.0018] |

Same variants on the other modes (eval, vs p0 [95% CI]). For llm_decides and no_backtest the verdict is the model's own belief; the replay is of the formula posterior over the questions they asked, so the row is a different system (formula verdict) from the logged one.

| variant | compose | fixed_list | act_only | batched_2_1 | library_account | llm_decides | no_backtest |
|---|---|---|---|---|---|---|---|
| identity (live posterior) | +0.0029 [-0.0035, +0.0098] | +0.0161 [+0.0005, +0.0323] | +0.0115 [+0.0003, +0.0231] | +0.0233 [+0.0119, +0.0351] | +0.0063 [-0.0006, +0.0138] | +0.0101 [+0.0036, +0.0176] | +0.0000 [+0.0000, +0.0000] |
| (a) shrink lambda | +0.0001 [-0.0005, +0.0008] | +0.0007 [-0.0009, +0.0024] | +0.0005 [-0.0007, +0.0017] | +0.0017 [+0.0005, +0.0030] | +0.0004 [-0.0003, +0.0012] | +0.0007 [+0.0000, +0.0015] | +0.0000 [+0.0000, +0.0000] |
| (b) accepted entries only [live rule; no-op] | +0.0029 [-0.0035, +0.0098] | +0.0161 [+0.0005, +0.0323] | +0.0115 [+0.0003, +0.0231] | +0.0233 [+0.0119, +0.0351] | +0.0063 [-0.0006, +0.0138] | +0.0101 [+0.0036, +0.0176] | +0.0000 [+0.0000, +0.0000] |
| (b') ALL table entries incl. unaccepted, lambda=1 | +0.0014 [-0.0053, +0.0087] | +0.0161 [+0.0005, +0.0323] | +0.0119 [+0.0004, +0.0242] | +0.0250 [+0.0133, +0.0370] | +0.0092 [+0.0005, +0.0182] | +0.0072 [-0.0002, +0.0157] | +0.0026 [-0.0030, +0.0085] |
| (b'') all table entries, shrink lambda' | +0.0000 [+0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] |
| (c) drop harmful families | +0.0019 [-0.0026, +0.0066] | +0.0131 [+0.0052, +0.0211] | +0.0131 [+0.0079, +0.0189] | +0.0179 [+0.0118, +0.0246] | +0.0049 [+0.0015, +0.0091] | +0.0066 [+0.0022, +0.0118] | +0.0000 [+0.0000, +0.0000] |
| (d) cap per answer / per alert | +0.0001 [-0.0012, +0.0013] | +0.0014 [-0.0014, +0.0044] | +0.0018 [-0.0002, +0.0038] | +0.0040 [+0.0019, +0.0063] | +0.0007 [-0.0006, +0.0020] | +0.0013 [+0.0001, +0.0027] | +0.0000 [+0.0000, +0.0000] |
| (e) abstain unless |sum| > tau | +0.0000 [+0.0000, +0.0000] | +0.0036 [-0.0070, +0.0141] | +0.0008 [-0.0006, +0.0032] | +0.0000 [+0.0000, +0.0000] | -0.0001 [-0.0005, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| (f) no update on timeout/refusal [no-op] | +0.0029 [-0.0035, +0.0098] | +0.0161 [+0.0005, +0.0323] | +0.0115 [+0.0003, +0.0231] | +0.0233 [+0.0119, +0.0351] | +0.0063 [-0.0006, +0.0138] | +0.0101 [+0.0036, +0.0176] | +0.0000 [+0.0000, +0.0000] |
| (g) re-estimated LRs on fit/cal answers, smoothed | +0.0000 [-0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] | +0.0000 [-0.0000, +0.0000] | +0.0000 [-0.0000, +0.0000] | +0.0000 [-0.0000, +0.0000] | +0.0000 [-0.0000, +0.0000] |
| (h) stacking: logit(p0) + family answer features | -0.0110 [-0.0194, -0.0026] | -0.0105 [-0.0191, -0.0020] | -0.0097 [-0.0182, -0.0016] | -0.0095 [-0.0178, -0.0011] | -0.0116 [-0.0206, -0.0030] | -0.0101 [-0.0185, -0.0018] | -0.0096 [-0.0193, -0.0003] |


Reading:
- Fit+cal in-sample gains over p0 are at most -0.0006 (dropping families, 0.1548 against 0.1554); shrinkage 0.1553, abstain 0.1550. In-sample, nothing is worth more than noise.
- On eval, compose: identity +0.0029 [-0.0035, +0.0098]; shrinkage (lambda 0.1) +0.0001 [-0.0005, +0.0008]; caps (0.1 per answer, 0.25 per alert) +0.0001 [-0.0012, +0.0013]; abstain (tau 0.9) collapses to p0 exactly (no compose alert has |sum| > 0.9). Using unaccepted entries as well (b') gives +0.0014 [-0.0053, +0.0087]; with a fitted lambda it fits lambda = 0, i.e. p0.
- The two modes that are significantly worse live become ties with shrinkage: fixed_list +0.0161 [+0.0005, +0.0323] to +0.0007 [-0.0009, +0.0024]; act_only +0.0115 [+0.0003, +0.0231] to +0.0005 [-0.0007, +0.0017]. Batched_2_1 improves from +0.0233 to +0.0017 [+0.0005, +0.0030], still slightly worse.
- Dropping the families chosen on fit+cal (dormant, n_repeated, new_account) does not help on eval: the family that hurts most on eval, volume_spike, was kept by fit+cal (10 answers, direction hit 0.6). Family selection on 4 to 35 answers per family is noise.
- llm_decides: the formula posterior over its questions is significantly worse than p0 (+0.0101 [+0.0036, +0.0176]), the model's own belief is +0.0011 [-0.0027, +0.0051], and the belief beats the formula by -0.0090 [-0.0159, -0.0034] (Table 11). The model's belief stays within 0.012 of p0 on average (corr 0.99): it behaves like a heavy shrinkage of the shown posterior. That is why it is the best live mode.

Variants (g) and (h), added after the first version of this report, with the same protocol: strength chosen by grouped 10-fold CV on fit+cal (whole clusters per fold), refit on all of fit+cal, eval scored once. (g): for every (family, eps, answer bucket) seen in fit+cal compose answers, an empirical log-LR with additive smoothing alpha over the buckets of the group (`smoothed_log_lrs`), applied to every answered step of the eval run whatever its back-test status. The bucket uses the LR table's mapping (bool as true/false, numbers as `b<i>` from the entry's bin edges). Numeric answers to questions that are not in the table have no edges and get no bucket, so only 89 of 238 answered eval compose questions get an estimated LR (98 have a bucket). Bins differ between programs of one family, so pooling by bucket index is approximate. (h): ridge-penalised logistic regression of y on logit(p0) (clipped to +-5, unpenalised) and, for 8 families with >= 8 answers, the sum of signed answer scores (bool +1/-1; numbers as clipped z-scores); ridge chosen by CV.

**Table 8b. Learned variants (g) and (h): details**
| quantity | value |
|---|---|
| (g) smoothing alpha (grouped CV on fit+cal) | 1e+06; CV Brier 0.1554 vs p0 0.1554; in-sample 0.1554; cells 44 |
| (g) CV Brier by alpha | 0.5: 0.1624, 1: 0.1603, 2: 0.1587, 5: 0.1572, 10: 0.1564, 20: 0.1559, 50: 0.1555, 100: 0.1554, 300: 0.1554, 1000: 0.1554, 1e+06: 0.1554 |
| (g) largest estimated cells (family|eps|bucket, log-LR) | n_counterparties:out|eps2|b2 +0.00; n_counterparties:out|eps2|b1 -0.00; n_tx:in|eps1|b1 +0.00; n_tx:in|eps1|b0 +0.00; n_tx:in|eps1|b3 -0.00; n_tx:in|eps1|b2 +0.00; n_tx:in|eps1|b4 -0.00; passed_on|eps1|false +0.00 |
| (h) ridge on answer features (grouped CV) | 30; CV Brier 0.1601 against 0.1602 for logit(p0) only; in-sample 0.1570 against 0.1582 |
| (h) CV Brier by ridge | 1: 0.1631, 3: 0.1614, 10: 0.1603, 30: 0.1601, 100: 0.1601, 300: 0.1602, 1000: 0.1602, 10000: 0.1602 |
| (h) slope on logit(p0), families used, largest answer coefficients | 0.79, 8, n_tx:in -0.14; matching_amount_out -0.08; volume_spike -0.05; passed_on +0.04; n_counterparties:in -0.04; n_counterparties:out -0.02 |
| share of eval compose answers that get an estimated LR (g) | 89 of 238 answered (98 have a bucket) |

(h) against the logit(p0)-only model (the Table 9 recalibration, same fit+cal data), eval: the difference is what the answers add. Paired cluster CI.

| mode | recalibrated p0 vs p0 | (h) vs p0 | (h) vs recalibrated p0 |
|---|---|---|---|
| compose | -0.0111 [-0.0196, -0.0028] | -0.0110 [-0.0194, -0.0026] | +0.0001 [-0.0011, +0.0011] |
| fixed_list | -0.0111 [-0.0196, -0.0028] | -0.0105 [-0.0191, -0.0020] | +0.0006 [+0.0001, +0.0011] |
| act_only | -0.0111 [-0.0196, -0.0028] | -0.0097 [-0.0182, -0.0016] | +0.0013 [+0.0006, +0.0020] |
| batched_2_1 | -0.0111 [-0.0196, -0.0028] | -0.0095 [-0.0178, -0.0011] | +0.0016 [+0.0010, +0.0023] |
| library_account | -0.0111 [-0.0196, -0.0028] | -0.0116 [-0.0206, -0.0030] | -0.0005 [-0.0023, +0.0010] |
| llm_decides | -0.0111 [-0.0196, -0.0028] | -0.0101 [-0.0185, -0.0018] | +0.0010 [+0.0002, +0.0017] |
| no_backtest | -0.0111 [-0.0196, -0.0028] | -0.0096 [-0.0193, -0.0003] | +0.0015 [-0.0017, +0.0046] |


Reading: the CV Brier of (g) equals p0 only when alpha is large (answers ignored); any real update is worse (0.1559 at alpha 20, 0.1603 at alpha 1). The CV Brier of (h) is 0.1601 against 0.1602 for logit(p0) alone: the answer features add 0.0001. Both are worse than p0 on fit+cal itself (the recalibration costs about 0.005 there), which is why Table 9's eval gain is a base-rate effect. The replay limit applies here too.

### 3.1 Recalibrating p0 alone (reported separately; not planner merit)

**Table 9. Recalibrating p0 alone (not the planner's merit), logistic recalibration fitted on fit/cal**
| quantity | value |
|---|---|
| fit/cal: intercept a, slope b on logit(p0) | -0.178, 0.791 |
| fit/cal: slope-only fit (no intercept) | 0.798 |
| label rate fit/cal vs eval | 0.520 vs 0.325 |
| mean p0 fit/cal vs eval | 0.545 vs 0.455 |
| fit/cal Brier p0 -> recalibrated p0 | 0.1554 -> 0.1582 |
| eval: recalibrated p0 vs p0 | -0.0111 [-0.0196, -0.0028] |
| eval: recalibrated p0 + updates (lambda=1) vs p0 | -0.0093 [-0.0197, +0.0001] |
| eval: recalibrated p0 + updates (lambda=1) vs recalibrated p0 alone | +0.0018 [-0.0037, +0.0077] |
| eval: recalibrated p0 + shrunk updates vs recalibrated p0 alone | +0.0000 [-0.0005, +0.0006] |


The recalibration (intercept -0.18, slope 0.79 on logit p0 clipped to +-5, fitted on fit+cal) improves eval Brier of p0 by -0.0111 [-0.0196, -0.0028], but it makes fit+cal itself worse (0.1554 to 0.1582). The eval gain is the shift in base rate (label rate 0.325 on eval, 0.520 on fit+cal), not a stable calibration error. It would be applied to p0 and to every system alike, so it does not change any system's difference to p0 and cannot be counted for the planner. The test base rate is unknown.

## 4. ReAct, Reflexion, UoT

**Table 10. Agents that output their own probability (eval)**
| agent | verdicts / 160 | Brier vs p0 [CI] | mean p (y=1 / y=0) | mean p0 (y=1 / y=0) | corr(p, p0) | corr(p, y) vs corr(p0, y) | most common p | share p>=0.8 | mean |p - p0| |
|---|---|---|---|---|---|---|---|---|---|
| react | 148 | +0.2572 [+0.1901, +0.3271] | 0.68 / 0.77 | 0.65 / 0.38 | -0.15 | -0.22 vs +0.44 | 0.85 (53%), 0.75 (9%), 0.4 (7%) | 0.63 | 0.40 |
| reflexion | 142 | +0.2380 [+0.1695, +0.3100] | 0.72 / 0.76 | 0.64 / 0.37 | -0.11 | -0.10 vs +0.44 | 0.85 (54%), 0.9 (9%), 0.75 (7%) | 0.67 | 0.40 |
| uot | 160 | +0.0894 [+0.0329, +0.1487] | 0.57 / 0.32 | 0.64 / 0.36 | +0.42 | +0.28 vs +0.45 | 0 (11%), 1 (5%), 0.006 (4%) | 0.28 | 0.30 |


- ReAct and Reflexion never see p0: `alert.summary` holds sender, receiver, banks, amount, currency, format, time and the evidence text (`ftm/retrieval/views.py::_base_summary`, `scripts/run_llm_agent.py`), and `ftm/external/react.py` and `text_agent.py` do not read `alert.p0`. Their final probability is the model's own. 53 to 54% of their verdicts are exactly 0.85, 63 to 67% are >= 0.8. Their probability is slightly lower for launderers (0.68) than for non-launderers (0.77) on ReAct; corr(p, y) is -0.22 against +0.44 for p0. The verdict is an anchor, not a reading of the answers. (I did not find 0.85 in the code; the exemplar finishes with 0.8.) 12 and 18 alerts end with no verdict and are scored p0.
- In the traces I read, ReAct asks at eps 0.5, and the table accepts nothing at eps 0.5, so the formula would not update on those answers either; the Thought text alone moves the number.
- UoT starts from p0 and updates a belief with answer probabilities that the model states itself (floored at 1e-3). It does use p0 (corr 0.42) but 52.5% of its verdicts are below 0.02 or above 0.98, mean |p - p0| is 0.30, and corr(p, y) is +0.28, lower than p0's +0.45. Overconfident self-stated likelihoods dominate.
- ReWOO and LLMCompiler are being fixed separately and are not covered.

## 5. Recommendations, ranked

Expected gains are relative to p0 on val eval; none can be positive in expectation because lambda_hat is not distinguishable from 0. What can be bought is "not worse than p0".

1. **Shrunk fusion for the final runs (needs the team's G2 decision; one cheap GPU rerun to confirm).** Weight the summed log-LR by lambda = 0.1 (fitted on fit+cal compose; it is the same idea as T4's zero-prior ridge, `fit_cv` with `prior="zero"`, and either is acceptable if G2 picks it). Replay on eval: compose +0.0029 to +0.0001 [-0.0005, +0.0008]; fixed_list +0.0161 to +0.0007; act_only +0.0115 to +0.0005; fit+cal in-sample 0.1573 to 0.1553 against p0 0.1554. Change: one number in the fusion (`NaiveLRFusion` posterior line, or a wrapper) read from the config; it is a parameter, not a feature, so allowed before 30 Oct if tuned on fit/cal as here. Risk: it makes the planner equal to p0 in effect; the model sees a posterior that barely moves, so its choices may change (replay limit). Confirm with one live run of compose and llm_decides on eval (about 160 alerts each). Do not describe the result as a win.
2. **Report the result as a pre-registered negative with the diagnosis, no GPU (before the freeze).** Compose / llm_decides tie p0 (+0.0029 and +0.0011, CIs span 0); act_only and fixed_list are significantly worse; lambda_hat intervals contain 0. The defensible claims are cost, validity and safety rows, and the explanation (back-test LRs do not transfer; 82% of composed questions have no accepted LR). Needs no G2 decision; consistent with T4's G2 text.
3. **Keep llm_decides as the planner headline over compose if one must be chosen, but pre-state it.** On val it is the best live mode (+0.0011 [-0.0027, +0.0051]), because the model shrinks the formula (belief against formula: -0.0090 [-0.0159, -0.0034]). It was not tuned on eval, but the difference to compose (+0.0029) is within noise (both CIs overlap), so the choice should come from fit+cal runs of llm_decides, which do not exist. A GPU run of llm_decides on fit and cal (300 alerts) would settle it before the freeze if there is time; otherwise do not switch.
4. **Fix the LR table, not the planner (T2/T4, after the freeze for the next iteration).** The accepted entries are the problem: 18 of 23 are at eps 2, none at eps <= 0.5, and the realised direction on val is no better than a coin flip (direction hit 0.44 to 0.47 of updating answers on eval compose and fixed_list). Re-estimate LRs on a time-adjacent train block, shrink them with empirical Bayes, or condition on p0 (T4's `cond LR`). Needs a back-test rerun (CPU) and a new G2 decision; not before 30 Oct.
5. **Treat p0 recalibration separately (T1/T4).** Base-rate shift and the saturated p0 >= 0.99 group (label rate 0.90, 48 of 460 val alerts at exactly p0 = 1) are p0 problems. The eval gain of -0.0111 is a base-rate effect and the fit+cal fit gets worse, so do not adopt a recalibration chosen here; if G2/G3 want one, it applies to every system including the p0 baseline.
6. **ReAct / Reflexion: show the same posterior line our planner gets and drop the 0.8 exemplar.** Expected: Brier closer to p0 (their 0.45 is mostly the missing prior); tuning needs fit/cal runs of these agents (none exist) and a GPU; it also changes the fairness protocol (they would then receive p0), which is the lead's call. Lowest priority and probably after the freeze. UoT: floor the stated likelihoods further (e.g. cap |log LR| per step) with the same fit/cal tuning requirement.

Variants (g) and (h) do not change the ranking. The diagnosis (no usable signal beyond p0 at this sample size) holds with re-estimated LRs and with a fitted model, so fixing the LR table (recommendation 4) should be read as "stop the loss", not "unlock a gain", unless a larger answer sample shows signal. Recommendation 1 (shrinkage) is unchanged and is the same decision as (g) with infinite smoothing.

## 6. What I am unsure about

- 45 compose eval alerts carry an update; every CI on lambda_hat is wide, so "no information" and "weak information" are not separable here. Pooling fit+cal and eval alerts for compose gives the same sign pattern but I did not fit on the pool, to keep eval held out.
- Fit+cal and eval differ in base rate (0.52 against 0.325), so lambda fitted on fit+cal may not transfer; that it is 0.1 here and the eval-optimal value is also near 0 is the only evidence for transfer.
- The replay does not model the model's reaction to a changed posterior.
- Per-family and cap / threshold choices rest on 5 to 35 answers per family and are not trustworthy.
- The source of the 0.85 anchor in ReAct and Reflexion was not found in the code.

## Appendix: the other tables

**Table 11. Belief-verdict modes (eval): the model's belief against the formula posterior over the same questions**
| mode | belief vs p0 | formula shadow vs p0 | belief vs shadow | alerts where belief != p0 | mean |belief - p0| | corr(belief, p0) |
|---|---|---|---|---|---|---|
| ours_llm_decides | +0.0011 [-0.0027, +0.0051] | +0.0101 [+0.0036, +0.0176] | -0.0090 [-0.0159, -0.0034] | 144 | 0.012 | +0.99 |
| ours_no_backtest | +0.0494 [-0.0024, +0.0959] | +0.0000 [+0.0000, +0.0000] | +0.0494 [-0.0024, +0.0959] | 160 | 0.278 | -0.09 |

