# ftm/eval and ftm/fusion: evaluation library (hand-off #16) and fusion v0 (hand-off #18)

Owner: T4 (Shreyas). numpy + scipy only, so everything runs in CI on the base install.

## Evaluation (`ftm.eval`)

| Module | What |
|---|---|
| `metrics` | `brier`, `brier_skill`, `gap_closed` (NaN below `MIN_GAP_DENOMINATOR` = 1e-5: report raw Brier differences instead), `pr_auc` (average precision, same as sklearn), `balanced_accuracy`, `caught_at_capacity` (top-k per day), `flips` (right→wrong and back), `ece` (equal-mass bins), `end_to_end_probability` (a missing verdict is scored as p₀; returns the no-verdict rate to report alongside), `detection_summary` |
| `bootstrap` | `cluster_ids` (attempt + shared accounts, union-find), `cluster_report`, `bootstrap_means` (fast path, 10,000 resamples), `cluster_bootstrap` (any statistic), `brier_diff`, `gap_closed_ci`, `bootstrap_pvalue` |
| `stats` | `holm`, `fixed_sequence`, `equivalence_margin`, `equivalent`, `non_inferior` |
| `runlog` | shared run-log schema **v1** (v0 columns unchanged; adds model, thinking, mode, p0, stop_reason, eps_spent, sim_days, trials, family, kind, rule_id, gate_reason, log_lr, note_present, injection_id) and `validate` |
| `runs` | `score(log, labels, clusters, upper, fusion)`: per (system, variant, seed) end-to-end / questions-only / p0 Brier, gap closed vs each upper limit with cluster CIs, no-verdict rate, cost, stop reasons |

```python
from ftm.eval import bootstrap as bs, metrics as mt, stats as st
cl = bs.cluster_ids(attempt_id, [src_id, dst_id])          # one label per alert
print(bs.cluster_report(cl))                               # check largest_share before trusting a CI
g = bs.gap_closed_ci(y, p_one_bank, p_system, p_upper, cl)  # point, 95% CI, share of tiny-denominator resamples
d = bs.brier_diff(y, p_system, p_baseline, cl)               # paired, negative = system better
p = bs.bootstrap_pvalue(-d["samples"], alternative="greater")
st.holm({"H1": p1, "H2": p2, "H3": p3, "H4": p4, "H9": p9})
```

Every CI resamples whole clusters, never rows. All columns of one call share the same resamples, so differences and ratios are paired.

**Check the clusters first.** Linking alerts through shared accounts can chain most alerts into one giant component (hub accounts), and then the bootstrap has almost nothing to resample. `cluster_report` prints `largest_share`. `scripts/eval_report.py` reports both clusterings: attempt + accounts (the plan's definition) and attempt + transaction (the fallback). Which one the report uses is fixed in `prereg/` at G3, before any test number exists.

**Gap check on real data:** `python scripts/eval_report.py --data hi_small --on alerts` (or `--on rows`). It needs T1's p0 for both modes, and writes `data/runs/eval_gap_<ds>_<on>.json`. `--on alerts` is the denominator of gap closed for our system, since the agent only sees alerts.

## Fusion v0 (`ftm.fusion.fuse`)

logit p = a·logit(p₀) + Σ_f w_f·x_f + c, where x_f is the summed back-tested log-LR of family f's answers for that alert.

| Mode | a, w, c | Use |
|---|---|---|
| `naive` | 1, 1, 0 | Bayes' rule, independent answers (the "summed log-LRs" ablation) |
| `shared` | fitted, one w for all families | v0 |
| `per_family` | fitted, one w per family | v1 (week 7); already works |

- **Ridge** pulls a and w toward 1 (naive Bayes), and c is free. A family with little data stays near "trust the back-tested LR". A new family with no fitted weight takes the prior.
- **Inputs are numbers only.** Text raises `TypeError`, so reply notes and LLM text cannot get in.
- **Answers in:** one row per answered question, `(alert_id, family, log_lr)`, with `log_lr` already clipped at the privacy ceiling. `answer_features` sums them into the matrix X. This is the format proposed to T2 for hand-off #9; declines and timeouts add no row.
- **Planner stop rule (T3):** `posterior(p0, {family: summed_log_lr}, model)` gives the running posterior for one alert.
- **Shuffled-answers test:** `shuffled_answers_test(p0, X, y)` is cross-fitted, so in-sample optimism cannot fake a gain. Real answers should give a gain; shuffled answers should give about 0. Pass cluster labels as `folds` on real data.

**Not done yet** (needs T2's back-test table): fitting on real validation answers, and the G2 check (expert questions improve Brier by at least 0.01 at ε=2 on val).
