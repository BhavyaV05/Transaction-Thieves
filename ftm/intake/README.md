# ftm/intake: conformal alert intake (C2) and alert sets (hand-off #6, draft)

Owner: T1 (Abhishek). Spec: `plan.md` 6.3 and 5.2 step 3, `tracks.md` 3.6. Config: `configs/intake/conformal.yaml`.

```
python scripts/make_alerts.py --data hi_small          # reads data/scores/<ds>/p0_one_bank.parquet
python scripts/make_alerts.py --data hi_medium --alpha-launder 0.55
```

Writes `data/alerts/<ds>/{train,val}.parquet` (gitignored) and `data/runs/alerts_<ds>.json` (thresholds, achieved vs target, sweep, ablations, projection). **There is no test alert set and no 1,000-alert subset: both need test labels.** See "Decisions for the team".

## Method

Input: one-bank `p0` (calibrated, `ftm/baselines`). Non-conformity of label k is 1 - P(k|x). Calibration rows are the val **`cal` part** (the quarter of the val time span after the half used for early stopping and the isotonic/Platt fit), pooled over the 28 scored bank views. Everything is reported on the val **`eval` part** (the last quarter), which no step above has seen.

- **Mondrian split conformal** (`conformal.mondrian_thresholds`): thresholds are computed separately on laundering and on clean calibration rows. `launder in set <=> p0 >= t_launder`, `clean in set <=> p0 <= t_clean`. Targets: miss rate `alpha_launder` among laundering rows, `alpha_clean` among clean rows (config). Finite-sample (n+1) correction. **Alert = launder is in the prediction set** (a superset of the plan's "set is {launder, clean}": a set `{launder}` only, i.e. `p0 > t_clean`, is also an alert; there are none on HI-Small because `t_clean` is far below `t_launder` at the chosen alpha, so the two definitions coincide there).
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
| drop_spans_split | the laundering attempt spans a split boundary (plan.md 5.2 step 3) |
| drop_window | the 72 h evidence window `[timestamp - 72h, timestamp]` starts before the start of the alert's own split |
| keep | `not drop_spans_split and not drop_window` |

Counts below say which rule is applied. Load with `pl.read_parquet` in a script or notebook (these are not transaction rows; no time-guard issue, but they contain labels, so do not hand them to a tool).

## Results: HI-Small (dev set; HI-Small has 2 val days, 2 bulk test days)

HI-Small is the debugging set; the 2,500-4,000-test-alert target is for HI-Medium.

**Achieved vs target miss rate** (val eval part, 93,240 rows, 122 laundering, `alpha_launder = 0.60`, `alpha_clean = 0.02`):

| | target | achieved | rows |
|---|---|---|---|
| laundering rows not alerted (launder not in set) | 0.600 | **0.590** | 122 |
| clean rows with clean not in set | 0.020 | **0.0203** | 93,118 |
| laundering share among auto-cleared (selective risk) | 0.0005 | 0.00078 (CP bound at delta 0.1 on cal: 0.00077) | 92,884 cleared |

The Mondrian targets are met. The selective-risk target is **not** met by this alert rule (see finding). Fixed-window ablation on the same rows: 67 alerts, 23 positives (recall 0.189), vs conformal 356 alerts, 50 positives (recall 0.410, precision 0.140). Time weighting changes the thresholds a little and does not change the conclusions (half-life 0.25 d: miss 0.598 vs 0.590 unweighted; `data/runs/alerts_hi_small.json`, `time_weighting_ablation`).

**Alert sets at `alpha_launder = 0.60`:**

| | alerts | positives | after span rule only | positives after span rule | after both rules | attempts in views / with an alert |
|---|---|---|---|---|---|---|
| train (OOF scores) | 5,518 | 260 | 5,426 | 168 | 1,447 (65 pos) | 127 / 91 |
| val (all val rows) | 1,352 | 180 | 1,184 | 12 | 0 | 85 / 68 |

**Alpha sweep and PROJECTION to test** (from val rates per day x 2 bulk test days; no test row was read; the true test rate will differ, val has 2 days):

| alpha_launder | t_alert (p0) | val tx-level recall | val precision | projected test alerts | projected test positives |
|---|---|---|---|---|---|
| 0.2 | 0.0013 | 0.739 | 0.008 | 43,284 | 334 |
| 0.4 | 0.0017 | 0.664 | 0.012 | 24,400 | 300 |
| **0.6** | 0.030 | 0.403 | 0.135 | 1,352 | 182 |
| 0.7 | 0.105 | 0.274 | 0.273 | 454 | 124 |
| 0.8 | 0.273 | 0.190 | 0.467 | 184 | 86 |

## Results: HI-Medium

HI_MEDIUM_SECTION

## Findings and decisions for the team

1. **The selective-risk bound and the alert budget conflict.** With 1 laundering row per ~950 transactions, a budget of a few thousand test alerts clears 99%+ of rows and misses about 60% of laundering **transactions** (tx-level recall 0.40 on HI-Small val at the chosen alpha). `scrc_beta` is therefore not met (0.00078 achieved vs 0.0005; the base rate is 0.00105). Applying it (`apply_scrc: true`) widens the alerts to `p0 >= 0.0016`, about 10% of all rows. Decide whether the bound or the budget wins; the shipped default is the budget.
2. **Attempt-level recall is far better than transaction-level**: 80% of val attempts (68 of 85) have at least one alerted transaction, 72% of train attempts. If the agent can follow an alert to the rest of its pattern, the useful recall is the attempt-level one. Tell T3.
3. **The plan.md 5.2 step 3 drop rules remove almost everything on a short split.** The span rule keeps only 12 of 180 val positives on HI-Small (long attempts straddle the val/test boundary). The 72 h evidence-window rule, read literally, drops every val alert (val is 2-3 days long) and the first 72 h of test. We ship both flags and `keep`; counts are given with each rule separately. Options: (a) apply only the span rule to val, (b) shorten the evidence window, (c) define the window start against the dataset start instead of the split start (history before the split is legitimate evidence under the `as_of` guard). Needs a ruling before the alert sets are frozen.
4. **The 1,000-alert external subset** is selected from test alerts stratified by label x typology x bank, so it needs test labels. `alerts.stratified_subset` is written (calls `require_final_run()` first, unit-tested on synthetic data) and is **not** run. Options: a TA-approved one-off locked build at G3, or stratify without labels (typology and label are unknown at selection time, so only bank and `p0` bucket).
5. **alpha_launder is an alert-budget knob**, not a coverage promise: to reach thousands of alerts the target miss rate on laundering rows is 0.55-0.6, so the guarantee is "recall of at least 40-45%".
6. Calibration data are small (HI-Small `cal` part: 104 laundering rows in the pooled set, a few per bank), so thresholds are pooled over banks. Per-bank thresholds would need more positives than a 3-day val has.
