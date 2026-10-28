# T4: why naive fusion hurts, the diagnostics, and fusion v1

Owner: T4 (Shreyas). Code: `ftm/fusion/diagnostics.py`, `scripts/fusion_diagnostics.py`, `ftm/fusion/fuse.py` (`prior`, `fit_cv`), `scripts/g2_check.py --fusion-select cv`. Tests: `tests/test_fusion_diagnostics.py`, `tests/test_fusion.py`, `tests/test_g2.py`.

**Status: tooling only. Nothing here has been run on real data** (written on a laptop with no `data/`). Every number below that is not from `docs/t1_handoffs.md` comes from a synthetic world built for the tests. The real run is a command for Abhishek / Akshat (section 4).

## 1. The problem

G2 fails on real data (`docs/t1_handoffs.md`, section A4): with naive fusion, `logit p = logit p0 + sum of back-tested log LRs`, the posterior moves p0 the wrong way. HI-Small gain -0.0053 [-0.0149, +0.0042], HI-Medium -0.0008 [-0.0135, +0.0125] (val `eval`, need >= +0.01), and in the eps sweep more questions or less noise make it worse (HI-Medium -0.0058 at eps 2, -0.0287 with no noise). Per-family fitted fusion, cross-fitted on HI-Medium, gives +0.0129 with CI [-0.034, +0.064]. So the evidence is informative but naive Bayes uses it wrongly. Three candidate reasons, each with a test:

| # | Hypothesis | What would be true | Table |
|---|---|---|---|
| 1 | Double counting | the one-bank model behind p0 already uses the patterns the questions measure (fan-in, pass-through, ...), so p0 and the answers are not independent given the label; Bayes counts the evidence twice | conditional vs marginal LR |
| 2 | Shift / winner's curse | LRs are measured on train alerts, and the same labels pick the accepted questions (e-BH on those labels); val is later in time, so the claimed LRs are too extreme or even point the wrong way | claimed vs val LR, direction |
| 3 | Calibration | the fused posterior is simply less calibrated than p0 | Brier / ECE of each posterior |

The hypotheses are not exclusive. Hypothesis 1 predicts "more questions and less noise make it worse" (sharper copies of what p0 knows are counted twice with more force), which is what the eps sweep shows. That makes 1 the first suspect, not a result.

## 2. What the diagnostics measure and how to read each table

`python scripts/fusion_diagnostics.py ...` runs the same planner as `g2_check.py` (fixed-order expert questions, eps 0.5 each, budget 2, through the harness), takes every answered question with a back-tested LR, and writes `data/runs/fusion_diagnostics_<dataset>.json` and `.md`. It prints one line, for example `fusion diagnostics [hi_medium]: 160 alerts, N answers; explained by p0 X%, direction disagrees Y%; Brier gain naive ..., best fitted ...`.

### Answer-cell table (hypotheses 1 and 2)

One row per (question at one eps, answer cell), with the per-family roll-up under it.

| column | meaning |
|---|---|
| n, launder rate | validation answers in the cell and the share of them that are launderers (Wilson CI in the JSON) |
| claimed LR | the LR from the train back-test that fusion actually adds |
| val LR (95% CI) | the same LR measured on the validation answers of this question: `P(cell given launder) / P(cell given clean)`, half-count correction, CI from the log-ratio (delta method). Read the CI first: most cells have few answers |
| cond LR | the Mantel-Haenszel pooled LR **within strata of p0** (terciles by default, `--strata`). Same quantity as val LR but with p0 held fixed |
| dir | claimed and val LR point the same way (`NO` = they disagree; `-` = one of them is within 0.05 of log 0, no direction) |
| mean p0 | mean one-bank score of the alerts that got this answer |

Two numbers summarise the table.

* **explained_by_p0** (hypothesis 1) = per cell `clip(1 - log cond LR / log val LR, 0, 1)`, averaged with weights `n * |log val LR|`, over cells with `|log val LR| >= 0.05`. 0 = the answers carry the same evidence once p0 is fixed; 1 = once p0 is fixed the answers say nothing. It comes with a cluster-bootstrap CI (whole clusters resampled, 200 resamples by default). Cutoffs used in the interpretation string: >= 50% "p0 already holds most of the evidence", 25 to 50% "part of it", below 25% "on top of p0". A CI wider than 60 points is flagged inconclusive.
* **share_answers_disagree** (hypothesis 2) = share of judged answers sitting in a cell where claimed and val direction differ (`cells_contradicted` counts the disagreeing cells whose val CI also excludes 0). Cutoffs: >= 30% "shift or winner's curse", 15 to 30% "some".

Limits, so they are not over-read: terciles of p0 are coarse, so some dependence on p0 survives inside a stratum and `explained_by_p0` is more likely to be too low than too high (try `--strata 5`); sparse cells give noisy cond LRs (the half-count correction pulls both LRs toward 1, equally); `explained_vs_claimed` in the JSON compares cond LR to the claimed LR instead, which mixes in hypothesis 2. The test worlds give: independent answers 1% [0%, 4%], answers that copy p0 98% [94%, 100%] (strata 6).

### Calibration table (hypothesis 3, and the decision)

Brier, ECE (equal-mass bins, `ftm.eval.metrics.ece`, 10 bins), mean p, and the gain over p0 = `Brier(p0) - Brier(fused)` (positive = the fusion helps; paired cluster bootstrap, 10,000 resamples) for: p0; `naive`; `shared` (one weight for all families); `per_family`; `per_family_cv` (v1: ridge and prior chosen by inner CV). Fitted rows are cross-fitted over whole clusters inside the val alerts, like `g2_check.py --fusion per_family`; `per_family_cv` also lists the (ridge, prior) chosen in each outer fold. **Check: the `naive` gain must equal `gain_naive` of `g2_check.py` on the same alerts** (-0.0053 for HI-Small and -0.0008 for HI-Medium in A4); if not, the two scripts see different alerts or caches. A naive row with worse Brier and a larger ECE or a shifted mean p than p0 is the calibration symptom.

## 3. Fusion v1

* `Fusion(prior="naive" | "zero")`, default `"naive"` (nothing existing changes). The ridge always pulls `a` toward 1. With `"zero"` it pulls the weights `w` toward 0: ignore the answers unless validation supports them. With `"naive"` it pulls toward 1, which is the wrong prior if naive hurts. The intercept `c` is not penalised in either case (as before).
* `fit_cv(p0, X, y, clusters, mode, ridges, priors, n_folds, seed)` fits every (prior, ridge) of the grid (defaults: ridge 1, 10, 100, 1000 pseudo-observations; both priors) on k-1 folds of whole clusters, scores the held-out fold, picks the lowest pooled held-out Brier (ties go to the earlier grid entry), refits on all rows, and returns `(Fusion, {"ridge", "prior", "cv_brier", ...})`. It sees only the rows it is given, so the selection is inner CV and never touches an outer test fold.
* `g2_check.py --fusion per_family --fusion-select cv`: the outer cross-fit calls `fit_cv` inside each training part. The JSON gets `fusion_select`, `fusion_chosen` (one choice per outer fold) and `fusion_chosen_all` (`fit_cv` on all val alerts: the setting to pre-register).
* **Rules.** The hyperparameters are chosen on **validation only**, never on test. The choice (mode, grid, priors, folds, seed, selection rule) **must be written into `prereg/` before G3**, and the lead edits `prereg/`; this PR does not. A fitted fusion can also undo a sign flip (a weight below 0), which naive cannot; `per_family_cv` gained +0.0305 on the synthetic flipped world where naive lost 0.096.

On synthetic double-counted data (`tests/test_fusion.py`, `tests/test_fusion_diagnostics.py`): naive loses 0.026 Brier, `per_family_cv` ends within 0.0001 of p0; with independent answers the weights come back near 1 and the gain matches naive's.

## 4. Commands for the real run

Back-test caches (from `docs/t1_handoffs.md` A4; they are in `data/backtest/<ds>/<namespace>/`, not in git): HI-Small realistic `4338ebb59b35` (1 accepted question), HI-Medium realistic `7df6bad5337d` (23 accepted). Check the directory exists; without `--backtest-dir` the script loads the train split and re-runs the back-test (slow but resumable, and writes the same cache).

```
# from the repo root, on a machine with data/ and the caches
python scripts/fusion_diagnostics.py --world real --data hi_medium --val-part eval --backtest-dir data/backtest/hi_medium/7df6bad5337d
python scripts/fusion_diagnostics.py --world real --data hi_small  --val-part eval --backtest-dir data/backtest/hi_small/4338ebb59b35
# the G2 check with fusion v1 (the JSON lands in --out; the default data/runs/g2_real.json is shared by both datasets)
python scripts/g2_check.py --world real --data hi_medium --val-part eval --fusion per_family --fusion-select cv --backtest-dir data/backtest/hi_medium/7df6bad5337d --out data/runs/g2_hi_medium_cv.json
python scripts/g2_check.py --world real --data hi_small  --val-part eval --fusion per_family --fusion-select cv --backtest-dir data/backtest/hi_small/4338ebb59b35  --out data/runs/g2_hi_small_cv.json
```

Optional, more alerts: `--val-part all` (HI-Medium 460 alerts instead of 160) for the diagnostics only. The `eval` part is the one G2 reports on; do not pick settings by looking at several variants and keeping the best one. CPU only, no LLM; HI-Medium needs the val bank views, so run it where `scripts/g2_check.py` already runs. `--strata 5` is a good second run for the double-counting number. Please send back `data/runs/fusion_diagnostics_<ds>.md` and `.json` and the two `g2_*_cv.json` files. Exit code 2 means data or cache is missing.

## 5. What each outcome means for the decision

Expect wide intervals: 160 askable alerts (52 positive) at about one answered question per alert over 23 accepted questions leaves many cells with fewer than 10 answers. Read points and CIs together, and say "inconclusive" when the CI says so.

| Outcome | Reading | Decision |
|---|---|---|
| explained_by_p0 >= 50% with a lower CI bound >= 25%; naive gain < 0; per_family_cv gain >= naive | double counting is the main cause | Report naive fusion as a **pre-registered negative result** (G2 not met by naive Bayes, with the diagnostic as the reason). Adopt fitted fusion (`per_family`, `prior="zero"`, `fit_cv`) as the v1 default for G3 and pre-register its grid. A cleaner fix is upstream (T1: a p0 that leaves the questioned patterns out); that is a different experiment, not a fusion change |
| explained_by_p0 < 25%; share_answers_disagree >= 30% (several cells contradicted) | shift or winner's curse: the train LRs do not transfer | Same reporting of naive. Fitted fusion with the zero prior is still the right default (it shrinks every family toward ignoring it). Ask T2 whether LRs can be re-estimated on a time-adjacent part of train or shrunk (empirical-Bayes) before fusion; do not touch val for that |
| both low; naive has a worse ECE or a shifted mean p than p0 | calibration / intercept problem | Try `shared` fusion (it has a free intercept and one weight); adopt only if the cross-fitted gain CI supports it |
| per_family_cv gain >= +0.01 point estimate | G2's criterion (point estimate >= 0.01) is met by the fitted fusion | Report as met **with the CI next to it** (HI-Medium A4 already had +0.0129 with CI [-0.034, +0.064]); do not describe it as significant if the CI covers 0. State that the fusion was selected by CV on validation |
| per_family_cv gain < +0.01 | fusion is not the bottleneck; the planner asks about one question per alert | G2 stays unmet; the lever is coverage (more accepted questions or alerts with answers, T2/T3), not fusion. Report that honestly |
| `naive` gain here differs from `g2_check.py`'s `gain_naive` | the two runs see different alerts / caches | Stop and compare `--val-part`, `--backtest-dir`, seed. No decision |
| diagnostics pass cleanly but all CIs are wide | the data cannot separate the hypotheses | Say so. Naive remains a negative result as measured; the zero-prior fit is adopted on the grounds that it cannot do materially worse than p0 by construction, not on a measured gain |

Whatever the outcome, `prereg/` needs the final fusion setting before G3, and the choice stays on validation.

## 6. Real-data results (run 2026-10-26 by Akshat on the 5070 Ti, reported in PR #66)

Commands of section 4, `--val-part eval`; outputs stay in `data/runs/` (not committed): `fusion_diagnostics_hi_{medium,small}.{md,json}`, `g2_hi_{medium,small}_cv.json`. Back-test namespaces `7df6bad5337d` (HI-Medium) and `4338ebb59b35` (HI-Small).

| | HI-Medium (160 alerts, 52 positive) | HI-Small (87 alerts, 17 positive) |
|---|---|---|
| explained by p0 | 0% (CI 0 to 55%) | 27% |
| answers whose direction disagrees (train vs val LR) | 41% | 85% |
| naive gain | -0.0008 (matches A4) | -0.0053 (matches A4) |
| per_family, fixed ridge | +0.0129 [-0.034, +0.064] | n/a |
| **per_family_cv gain (G2)** | **+0.0053 [-0.041, +0.054], not met** | **+0.0007 [-0.015, +0.019], not met** |
| CV choice on all val | ridge 100, prior zero | n/a |

Reading (section 5, row 2): the shift / winner's-curse row, not double counting. On HI-Medium `dormant(ALERT.receiver, days=3)` flips direction (claimed LR 0.49 for `true`, validation LR 1.15). Naive fusion is reported as a pre-registered negative result; `per_family`, `prior="zero"`, `fit_cv` is the proposed v1 default; G2 is not met either way (team decision pending, `docs/schedule_nov10.md` rule 6). All intervals are wide: "inconclusive" is the honest word for the fitted fusion.

## 7. The time confound and the replication check

Source: GitHub issue #84 (T4 prototype, CPU, real HI-Medium train, descriptive, not a pre-registered result). Code: `ftm/eval/replication.py`, `scripts/replication_check.py`, the `--replication-filter` / `--replication-blocks` options of `scripts/g2_check.py` and `scripts/fusion_diagnostics.py` (off by default; nothing in the default runs changes).

**The confound.** The back-test LR of the history-based questions (`dormant`, `new_account`) is a time confound (Simpson's paradox). For `dormant(ALERT.receiver, days=7)` at eps 2 (760 answers) the pooled log LR of `true` is -0.92, but in the two time halves of train it is +0.02 and -0.15: the launder rate jumps from 5% to 37% and the share of `true` answers from 88% to 15% between the halves (very likely ledger warm-up: early in the data an account has little history). Inside each half the question carries almost no signal; the pooled number is the drift. The label-permutation e-value assumes labels are exchangeable given the answers, which fails when both drift with time, so such a question can pass e-BH.

**The rule.** For each accepted question take the cell with the largest |pooled log LR| (alpha 0.01 smoothing as in `estimate_lr`, no ceiling clip), compute its log LR in each of K equal-count time blocks (K = 2 halves by default, `--blocks 4` for quarters) and keep the question only if pooled and every block have the same strictly non-zero sign. A zero or undefined log LR, or a block with fewer than 30 answers or fewer than 5 launder or clean answers, counts as "does not replicate". The report also gives a Mantel-Haenszel time-stratified log LR over 4 blocks (`--mh-blocks`). Train data only; no val label enters the rule.

**Real HI-Medium (23 accepted questions, namespace `7df6bad5337d`).**

| | all 23 | replicating (12) | not replicating (11) |
|---|---|---|---|
| families | | `returns_home`, `cross_currency_out`, `inflow_bucket`, `time_to_forward`, `n_banks` | `dormant` (sender, receiver, days 3 and 7), `new_account` (days 2 and 5), `n_repeated`, `n_new_counterparties`, `volume_spike` |
| val `eval` direction agrees with train | 14/23 (61%) | 10/12 (83%) | 4/11 (36%) |

Mean |log LR|: 0.82 as claimed, 0.46 Mantel-Haenszel, 0.50 observed on val (the stratified number is the one that matches val). Caveats: small val cells (5 to 81 answers), and the 23 questions are not independent (about 9 families repeated over eps and subject).

G2 with the 11 non-replicating questions made ineligible (e_value = 1 before e-BH, m unchanged, 12 accepted; 2,000 resamples; descriptive):

| val part | fusion | all 23 accepted | replicating only (12) |
|---|---|---|---|
| eval | naive | -0.0008 [-0.013, +0.011] | +0.0073 [-0.003, +0.018] |
| eval | per_family, cv | +0.0053 [-0.040, +0.054] | +0.0238 [-0.005, +0.058] |
| cal | naive | -0.0013 [-0.017, +0.014] | +0.0086 [-0.001, +0.018] |
| cal | per_family, cv | -0.0027 [-0.028, +0.025] | +0.0054 [-0.019, +0.031] |

**Status.** This is a lead, not a G2 pass: every interval contains 0, and `eval` was looked at for two variants (all accepted vs replicating only), so its numbers are descriptive. Adopting the filter needs T2's agreement (it is a change to what is accepted; the cleaner fixes are label permutation within time blocks in the e-process or a Mantel-Haenszel LR, both in T2's `ftm/backtest/`, which this PR does not touch) and the rule fixed in the pre-registration before the 30 Oct freeze, then reported once. If it is not adopted, the confound is reported as a finding (claimed 0.82 vs stratified 0.46 vs val 0.50) and G2 stays not met. Note that after the filter e-BH is re-run over the same m, so a kept question can in principle lose acceptance (rank effect); the output states `accepted_before` and `accepted_after`.

**Reproduce (a machine with `data/`; about 30 min of CPU, no GPU, no LLM; not run in CI).**

```
# smoke test, no data
python scripts/replication_check.py --synthetic
# the table (writes outputs/replication/hi_medium.{json,md}); add --val-part eval for the val direction column, --blocks 4 for quarters
python scripts/replication_check.py --data hi_medium --backtest-dir data/backtest/hi_medium/7df6bad5337d --val-part eval
# G2 with the filter, from that JSON (the output JSON gets a "replication_filter" block: filter used, n_dropped, accepted before and after)
python scripts/g2_check.py --world real --data hi_medium --val-part eval --backtest-dir data/backtest/hi_medium/7df6bad5337d --replication-filter outputs/replication/hi_medium.json --out data/runs/g2_hi_medium_repl.json
python scripts/g2_check.py --world real --data hi_medium --val-part eval --fusion per_family --fusion-select cv --backtest-dir data/backtest/hi_medium/7df6bad5337d --replication-filter outputs/replication/hi_medium.json --out data/runs/g2_hi_medium_cv_repl.json
# the same for the diagnostics, and for the cal part as the independent check
python scripts/fusion_diagnostics.py --world real --data hi_medium --val-part eval --backtest-dir data/backtest/hi_medium/7df6bad5337d --replication-filter outputs/replication/hi_medium.json --out data/runs/fusion_diagnostics_hi_medium_repl.json
python scripts/g2_check.py --world real --data hi_medium --val-part cal --backtest-dir data/backtest/hi_medium/7df6bad5337d --replication-filter outputs/replication/hi_medium.json --out data/runs/g2_hi_medium_cal_repl.json
```

`--replication-blocks K` instead of `--replication-filter` recomputes the flags inside the run (it loads the train world, so it is slower). Do not run it on the test split; it cannot (the test lock stays).
