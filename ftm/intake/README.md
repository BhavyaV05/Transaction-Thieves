# ftm/intake: conformal alert intake (C2) and alert sets (hand-off #6, draft)

Owner: T1 (Abhishek). Spec: `plan.md` 6.3 and 5.2 step 3, `tracks.md` 3.6. Config: `configs/intake/conformal.yaml`.

```
python scripts/make_alerts.py --data hi_small          # reads data/scores/<ds>/p0_one_bank.parquet
python scripts/make_alerts.py --data hi_medium --alpha-launder 0.55
```

Writes `data/alerts/<ds>/{train,val}.parquet` (gitignored) and `data/runs/alerts_<ds>.json` (thresholds, achieved vs target, sweep, ablations, projection). **There is no test alert set and no 1,000-alert subset: both need test labels.** See "Decisions for the team".

## Method

Input: one-bank `p0` (calibrated, `ftm/baselines`). Non-conformity of label k is 1 - P(k|x). Calibration rows are the val **`cal` part** (the quarter of the val time span after the half used for early stopping and the isotonic/Platt fit), pooled over the 28 scored bank views. Everything is reported on the val **`eval` part** (the last quarter), which no step above has seen.

- **Mondrian split conformal** (`conformal.mondrian_thresholds`): thresholds are computed separately on laundering and on clean calibration rows. `launder in set <=> p0 >= t_launder`, `clean in set <=> p0 <= t_clean`. Targets: miss rate `alpha_launder` among laundering rows, `alpha_clean` among clean rows (config). Finite-sample (n+1) correction. **Alert = launder is in the prediction set.** This deviates from the plan's wording ("the set is {launder, clean}") on purpose: at every alpha that gives a usable alert budget, `t_launder > t_clean`, so *no* row has both labels in its set (0 of 320k HI-Medium eval rows); alerts are all `{launder}`-only sets, and the plan's literal rule would produce zero alerts. The rows in between (`t_clean < p0 < t_launder`) have an **empty** set; they are not alerts and hold 87 of the 527 eval laundering rows on HI-Medium (see `set_types_eval` in `data/runs/alerts_<ds>.json`). Decision 5: empty sets are not alerts but are counted as intake misses in the end-to-end numbers.
- **Time-weighted calibration** (`time_weights`): calibration row weight `0.5 ** (age / half_life)`, age in fractional days from the last calibration row; the test point has weight 1 (weighted conformal prediction, Tibshirani et al. 2019, applied to the drift idea of plan ref [39]). `half_life_days: 0.25` because the `cal` part spans under one day. Equal weights are `half_life_days: null`.
- **Selective risk bound** (`selective_clear_threshold`, `selective_risk_upper_bound`): the simplest correct variant, not the exact procedure of plan ref [38]. The auto-cleared region is `p0 < lam`; `lam` is the largest value at which a one-sided Clopper-Pearson (1-delta) upper bound on P(laundering | cleared) from the calibration rows is <= `scrc_beta`, scanning `lam` upwards with a fixed start (4 ln(1/delta)/beta cleared rows) and stopping at the first failure (fixed-sequence testing as in Learn-then-Test, Angelopoulos et al. 2021). Unweighted. It is **reported, not applied**, by default (`apply_scrc: false`): see the finding below. **TODO:** implement [38] exactly (weighted, finite-sample selective guarantee) if the team wants it applied.
- **Ablation**: a fixed `0.15 <= p0 <= 0.85` window.
- The conformal guarantee is marginal per class under exchangeability of calibration and eval rows. Rows are not exchangeable in time (drift) and one transaction can appear in two bank views, so treat achieved-vs-target as the check, not the guarantee.

## Alert-set schema (hand-off #6, draft)

`data/alerts/<ds>/train.parquet`, `val.parquet`. One row per (transaction, bank view) with `p0 >= t_alert`.

| column | note |
|---|---|
| alert_id | `T000123` (train) / `V000123` (val), by (timestamp, tx_id, bank) |
| tx_id, bank, timestamp | bank = the view that raised the alert |
| p0, label | calibrated one-bank score, `is_laundering` |
| attempt_id, typology | from `patterns.parquet`; null when the transaction is in no attempt (clean rows and the laundering rows missing from the pattern file) |
| split, val_part | val rows carry `fit`/`cal`/`eval`; train alerts use out-of-fold scores |
| spans_split | the laundering attempt spans a split boundary (plan.md 5.2 step 3). **A flag for the sensitivity analysis only; the headline alert set is never filtered by it** (decision 3 of `docs/t4_decisions_week2.md`). The 72 h evidence-window rule was removed: it only drops alerts whose evidence reaches into an earlier split, which is past data and allowed at `as_of`. |

Counts below are for the full alert set. Load with `pl.read_parquet` in a script or notebook (these are not transaction rows; no time-guard issue, but they contain labels, so do not hand them to a tool).

## Results

Both datasets: calibration = val `cal` part, results on the val `eval` part (last quarter of the val time span; 93,240 rows / 122 laundering on HI-Small, 320,372 rows / 527 laundering on HI-Medium, pooled over the bank views), `alpha_clean = 0.02`, `half_life_days = 0.25`. HI-Small is the dev set. Scores are the GFP+LightGBM one-bank `p0` of `ftm/baselines`.

**Achieved vs target miss rate (the C2 "done when" check):**

| | alpha_launder (target miss, laundering) | achieved miss, laundering | clean target | achieved miss, clean | selective risk target | achieved selective risk |
|---|---|---|---|---|---|---|
| HI-Small (alpha 0.60) | 0.600 | **0.598** (n=122) | 0.020 | **0.0236** (n=93,118) | 0.0005 | 0.00079 (92,908 auto-cleared) |
| HI-Medium (alpha 0.63) | 0.630 | **0.624** (n=527) | 0.020 | **0.0238** (n=319,845) | 0.0005 | 0.00103 (319,566 auto-cleared) |

The laundering target is met; the clean miss rate is 0.4 points over its 2% target (probably drift between the `cal` and `eval` quarters; not investigated). The selective-risk target is **not** met by this alert rule (see finding 1).

**Time weighting matters under drift (ablation, HI-Medium, same alpha 0.63):**

| half-life (days) | t_launder | achieved laundering miss | eval alerts |
|---|---|---|---|
| none (equal weights) | 0.385 | **0.713** (target 0.63: misses the target) | 333 |
| 0.25 (shipped) | 0.129 | **0.624** | 806 |
| 0.1 | 0.059 | 0.573 | 1,328 |

On HI-Small the same ablation moves the miss rate only from 0.590 to 0.598 (no drift to correct). So the weighting is what lets the intake hit its target on HI-Medium. 

**Fixed-window ablation (0.15 <= p0 <= 0.85), same eval rows:** HI-Small 67 alerts, 23 positives (recall 0.189) vs conformal 332 alerts, 49 positives (recall 0.402, precision 0.148); HI-Medium 590 alerts, 118 positives (recall 0.224) vs conformal 806 alerts, 198 positives (recall 0.376, precision 0.246).

**Alert sets** (`data/alerts/<ds>/{train,val}.parquet`; train alerts use out-of-fold scores; rebuilt 2026-10-23 under the decisions, no drop rule applied; thresholds and counts unchanged by the rebuild):

| | alerts | positives | unique tx | flagged `spans_split` | alerts / positives if the span rule were applied (sensitivity) | attempts in views / with an alert |
|---|---|---|---|---|---|---|
| HI-Small train | 5,198 | 255 | 5,063 | 91 | 5,107 / 164 | 127 / 91 |
| HI-Small val | 1,284 | 177 | 1,221 | 166 | 1,118 / 11 | 85 / 68 |
| **HI-Medium train** | **10,962** | 1,448 | 10,535 | 583 | 10,379 / 865 | 623 / 496 |
| **HI-Medium val** | **2,238** | 798 | 2,113 | 756 | 1,482 / 42 | 423 / 350 |

**Per val part (decision 9; thresholds from `cal`, so `cal` is in-sample for the laundering target; `eval` is the clean check).** `fit` is the half of val on which the p0 model chose its early-stopping point and fitted the isotonic/Platt calibrator, so its numbers are optimistic. One row = one (transaction, bank view). "Empty set" = neither label in the prediction set (`t_clean < p0 < t_launder`); not an alert, **counted as an intake miss** (decision 5). Intake miss = share of laundering rows not alerted (decision 2; equal to the achieved laundering miss while `apply_scrc` is false); "by tx" counts a transaction as alerted if any bank view alerts it.

| | part | rows | laundering rows | alerts | positives | target / achieved laundering miss | target / achieved clean miss | empty-set rows (of which laundering) | intake miss, rows | intake miss, by tx |
|---|---|---|---|---|---|---|---|---|---|---|
| HI-Small | fit | 189,966 | 218 | 648 | 88 | 0.60 / 0.596 | 0.02 / 0.018 | 2,875 (34) | 0.596 | 0.591 (193 tx) |
| HI-Small | cal | 96,615 | 104 | 304 | 40 | 0.60 / 0.615 | 0.02 / 0.019 | 1,542 (15) | 0.615 | 0.652 (89 tx) |
| HI-Small | **eval** | 93,240 | 122 | **332** | 49 | 0.60 / **0.598** | 0.02 / 0.024 | 1,946 (30) | **0.598** | 0.590 (105 tx) |
| HI-Small | all val | 379,821 | 444 | 1,284 | 177 | 0.60 / 0.601 | 0.02 / 0.019 | 6,363 (79) | 0.601 | 0.605 (387 tx) |
| HI-Medium | fit | 267,722 | 757 | 935 | 419 | 0.63 / 0.447 | 0.02 / 0.022 | 5,583 (141) | 0.447 | 0.452 (679 tx) |
| HI-Medium | cal | 196,709 | 387 | 497 | 181 | 0.63 / 0.532 | 0.02 / 0.021 | 3,873 (86) | 0.532 | 0.533 (347 tx) |
| HI-Medium | **eval** | 320,372 | 527 | **806** | 198 | 0.63 / **0.624** | 0.02 / 0.024 | 7,106 (87) | **0.624** | 0.627 (477 tx) |
| HI-Medium | all val | 784,803 | 1,671 | **2,238** | 798 | 0.63 / 0.522 | 0.02 / 0.023 | 16,562 (314) | 0.522 | 0.526 (1,503 tx) |

Read: the end-to-end intake misses 60% (HI-Small) and 62% (HI-Medium) of the laundering rows on the untouched `eval` part, and the empty-set rows hold 30 of the 73 missed laundering rows on HI-Small and 87 of the 330 on HI-Medium (41% and 26%). The HI-Medium laundering miss on `fit` and `cal` is below its 0.63 target and reaches it only on `eval` (with the same threshold the `fit` miss is 0.45), probably the same drift that makes time weighting necessary (not investigated further). For the same reason the HI-Medium 2,238 val alerts (798 positives) are mostly in `fit` and `cal`; `eval` alone has 806 alerts and 198 positives, which is what E18, fusion and G2 can use without re-fitting on rows the calibration saw (decision 9: cross-fit inside val).

HI-Medium targets (plan.md 6.3): train about 20,000 (we have 10,962 at the alpha that satisfies the test target; alpha 0.55 gives 18,515 train alerts but about 5,900 projected test alerts), val at least 1,000 (**2,238, met**), test 2,500-4,000 with at least 375 positives (**projected 3,475 with 1,011 positives, met**; see below).

**PROJECTION of test alerts (not a measurement; no test row was read):** alerts per day on val `cal`+`eval` (half of the val span) x the bulk test days (HI-Medium: 16 bulk days - 12 = 4; HI-Small: 10 - 8 = 2). It ignores the sparse test tail (0.08% of rows) and assumes the val rate carries over; val is 3 days (HI-Medium) so the day-to-day variation is unknown. The test laundering rate per transaction differs from val (0.151% vs 0.173% on HI-Medium), so read the positives as plus or minus 30%.

| dataset | alpha_launder | t_alert (p0) | val recall (tx) | val precision | projected test alerts | projected test positives | train alerts | val alerts (all val rows) |
|---|---|---|---|---|---|---|---|---|
| HI-Medium | 0.50 | 0.0244 | 0.525 | 0.125 | 10,221 | 1,280 | 31,035 | 6,334 |
| HI-Medium | 0.55 | 0.0561 | 0.480 | 0.200 | 5,867 | 1,171 | 18,515 | 3,789 |
| HI-Medium | 0.60 | 0.0909 | 0.443 | 0.253 | 4,264 | 1,080 | 13,443 | 2,724 |
| HI-Medium | **0.63** | 0.1292 | 0.415 | 0.291 | **3,475** | **1,011** | 10,962 | 2,238 |
| HI-Medium | 0.66 | 0.1765 | 0.376 | 0.374 | 2,453 | 917 | 7,351 | 1,599 |
| HI-Medium | 0.70 | 0.3333 | 0.338 | 0.499 | 1,651 | 824 | 4,589 | 1,100 |
| HI-Small | 0.55 | 0.0137 | 0.469 | 0.091 | 2,326 | 212 | 10,540 | 2,262 |
| HI-Small | **0.60** | 0.0323 | 0.394 | 0.140 | **1,272** | 178 | 5,198 | 1,284 |
| HI-Small | 0.70 | 0.1125 | 0.270 | 0.281 | 434 | 122 | 1,746 | 439 |

(HI-Small has no test-alert target; it is the dev set. alpha jumps fast between 0.4 and 0.55 because `p0` has long plateaus: the alert count goes from 25k to 2.3k projected on HI-Small.)

## Findings and decisions for the team

1. **The selective-risk bound and the alert budget conflict.** With 1 laundering row per ~950 transactions, a budget of a few thousand test alerts clears 99%+ of rows and misses about 60% of laundering **transactions** (tx-level recall 0.38-0.40 on val at the chosen alpha). `scrc_beta` is therefore not met (0.00079 achieved on HI-Small and 0.00103 on HI-Medium vs 0.0005; the base rates are 0.0013 and 0.0016, so the auto-cleared set holds laundering at 60-65% of the base rate). Applying it (`apply_scrc: true`) widens the alerts to `p0 >= 0.0018` (HI-Small) or `0.0015` (HI-Medium), about 10% of all rows. Decide whether the bound or the budget wins; the shipped default is the budget.
2. **Attempt-level recall is far better than transaction-level**: 83% of HI-Medium val attempts (350 of 423) and 80% of HI-Small (68 of 85) have at least one alerted transaction (HI-Medium train: 496 of 623 = 80%). If the agent can follow an alert to the rest of its pattern, the useful recall is the attempt-level one. Tell T3.
3. **Drop rules (resolved, decision 3 of `docs/t4_decisions_week2.md`).** The 72 h window rule is removed from the config, the code and the tests. The span rule is a flag column (`spans_split`) only: it keeps 42 of 798 HI-Medium val positives (11 of 177 on HI-Small), so report headline numbers on all alerts and the kept-only subset as a sensitivity row.
4. **The 1,000-alert external subset (resolved, decision 1).** `alerts.stratified_subset` now defaults to the label-free `strata="bank_p0_decile"` (home bank x decile of p0 over the alerts passed in, seed 0, proportional allocation with at least one per non-empty stratum). It calls `require_final_run()` first, so it can only be built at G5 from test alerts; the earlier `strata="label_typology_bank"` rule is kept as an option for synthetic and val checks. Unit-tested on synthetic data under a monkeypatched flag (`tests/test_intake.py`). `prereg/subset_manifest.txt` records that the option exists.
5. **alpha_launder is an alert-budget knob**, not a coverage promise: to reach thousands of alerts the target miss rate on laundering rows is 0.55-0.6, so the guarantee is "recall of at least 40-45%".
6. Calibration data are small (`cal` part: 104 laundering rows on HI-Small, 387 on HI-Medium, a handful per bank), so thresholds are pooled over banks. Per-bank thresholds would need more positives than a 3-day val has.
7. **Time weighting is not optional on HI-Medium**: equal weights miss the laundering target by 8 points (0.713 vs 0.63). The calibration window is short (under a day), so the half-life is 0.25 day, fixed before looking at HI-Medium `eval` as a fraction of the `cal` span; 0.1 is shown only for sensitivity (it over-shoots the other way).
