# T1 hand-offs for the other tracks (week 7, branch `t1/week7-handoffs`)

Owner: T1 (Abhishek). Machine: 3070 Ti laptop, system python 3.12, torch 2.5.1+cu121 untouched, no new package. Test split never read, `FTM_FINAL_RUN` never set. Every result below carries its commit in the JSON it came from (`data/runs/`, gitignored). Section A4 was completed on branch `t1/week7-9`.

## A1. `direction` and `counterparty_bank` in alert contexts (asked by T2, T3: `ftm/planners/NONLLM.md`)

- **Alert sets** `data/alerts/<ds>/{train,val}.parquet` have two new columns, `direction` (`out` if the home bank `bank` sent the money, else `in`) and `counterparty_bank` (the other bank, a public bank id; null when both parties are at the home bank: 237 train and 55 val alerts on HI-Small, 344 and 82 on HI-Medium). Built in `ftm/intake/alerts.py::add_direction`, written by `scripts/make_alerts.py`. The alert sets were rebuilt for both datasets: **every old column is identical** to the previous files (checked with `DataFrame.equals`), only the two columns are new.
- **`alert.summary`**: T3's `scripts/run_llm_agent.py::data_alerts` builds the summary with `ftm.retrieval.views.alert_summary(row, aliases)` after `views.top_k_by_amount_view(views_dir, bank, row, aliases)`. `top_k_by_amount_view` now records `home_bank` (and the alert directory) on `row`, and `alert_summary` then adds `direction`, `counterparty_bank` and, when present, `top_features` (A2). No edit to T3's or T2's code. If `alert_summary` is called without that first call (no `home_bank` on the row) it behaves as before. T2's own contexts (`ftm.banks.binding.alert_context`) already had the two keys.
- **Check** (`scripts/check_alert_context.py`, `tests/test_handoffs.py`): on 300 seeded alerts per split (600 per dataset) the parquet columns, the new `alert_summary` keys and T2's `alert_context(bind_alert(...))` agree on every alert (0 mismatches of 600 on HI-Small and on HI-Medium). Same rule in all three places (`views.direction_keys`).
- **Flag for T3.** `Alert` in the planner and ReAct prompts is `json.dumps(summary)` minus `evidence`, `partner_banks`, `shape` (`_SUMMARY_HIDDEN` in `ftm/planners/llm.py` and `ftm/external/text_agent.py`). The new keys therefore also reach every prompt; `top_features` adds about 200 tokens (rounded to 4 decimals). T3's trim loop shrinks the evidence to fit, but prompt sizes and any fixed-seed run change; if only Sharma should see `top_features`, add it to `_SUMMARY_HIDDEN` there (T3's file, not touched).

## A2. `top_features` per alert (asked by T4, Sharma port: `summary["top_features"]`)

- `scripts/top_features.py --data <ds>` refits the home bank's one-bank GFP+LightGBM model with the exact settings of `scripts/train_gfp.py` (seed 0, 4 threads), saves it (`models/gfp_lgbm/<ds>/bank<id>.txt`, gitignored) and writes **`data/alerts/<ds>/top_features_{train,val}.parquet`**: `alert_id`, `tx_id`, `bank`, `top_features` = list of 5 `{feature: str, value: float, contribution: float}`, sorted by |contribution|. Contribution = LightGBM `pred_contrib=True` (TreeSHAP) in log-odds of the raw model (before the monotone calibrator). Val alerts use the full model, train alerts the out-of-fold model that produced their p0.
- **p0 reproduces exactly.** The refit raw scores and p0 equal the stored `p0_one_bank.parquet` rows with max absolute difference 0.0 on every bank (HI-Small 23 banks with alerts, 6,482 alerts; HI-Medium 29 banks, 13,200 alerts; none without top features). `data/runs/top_features_<ds>.json`.
- **Names are readable**: the 66 snapml columns were decoded with designed toy graphs (`tests/test_handoffs.py::test_gfp_labels_follow_the_snapml_layout`): e.g. `sender_out_amount_sum_24h`, `receiver_in_distinct_counterparties_24h`, `fan_out_size3plus_24h`, `temporal_cycle_size3plus_24h`, `hour_of_day`, `home_bank_is_sender`; categoricals as `payment_format=ACH` (value 1.0). Time-guard: only the features the model already uses (past data; snapml batches of 128 edges, documented in `ftm/baselines/gfp_gbt.py`).
- **Reading.** The top feature is `payment_format=ACH` in 94 to 99 % of alerts (HI-Small train 94 %, val 98 %; HI-Medium 99 %), because nearly all alerts are ACH payments and the model's largest effect is the format; the 2nd to 5th items carry the alert-specific information (`*_tx_per_counterparty_24h`, amounts, counterparties).
- Unit tests on synthetic data (`tests/test_handoffs.py`): sorted, complete, readable, equal to the booster's own contributions.

## A3. Repeat-account statistics (asked by T2, per-account cap)

`scripts/cap_sweep.py --data <ds> --split <split>` on the real alert sets (askable alerts only: those whose partner bank is one of the 30 modelled views), 2 questions of eps 1 per alert, per-alert budget 2, JSON in `data/runs/cap_sweep_<ds>_<split>.json`. `repeat_stats`:

| | alerts | partner-side accounts | max alerts / account | accounts in 2+ / 3+ / 5+ alerts | alerts touching a repeat account |
|---|---|---|---|---|---|
| HI-Small train | 1,038 | 845 | 14 | 112 / 36 / 7 | 305 |
| HI-Small val | 324 | 274 | 9 | 27 / 11 / 3 | 77 |
| HI-Medium train | 1,983 | 1,493 | 14 | 294 / 93 / 15 | 784 |
| HI-Medium val | 460 | 389 | 5 | 52 / 10 / 4 | 123 |

Refused share of questions by cap (alerts with a refusal in brackets):

| cap | HI-Small train | HI-Small val | HI-Medium train | HI-Medium val |
|---|---|---|---|---|
| 2 | 18.6 % (193) | 15.4 % (50) | 24.7 % (490) | 15.4 % (71) |
| 4 | 7.8 % (81) | 7.1 % (23) | 9.9 % (196) | 4.1 % (19) |
| **8 (current)** | 3.1 % (32) | 2.2 % (7) | 3.2 % (64) | 0.9 % (4) |
| 16 | 1.1 % (11) | 0.3 % (1) | 1.0 % (19) | 0 |
| 32 / off | 0 (max 28 eps on one account) | 0 | 0 (28) | 0 |

Reading: between a quarter and two fifths of the askable alerts touch a partner account that is the subject of another alert, one account is the subject of up to 14. A cap of 8 refuses 1 to 3 % of the questions (3 % on the 4-5x larger train sets, under 1 % on HI-Medium val); a cap of 4 refuses 4 to 10 %, 2 refuses 15 to 25 %. The cap bites on a small tail of repeat accounts; 8 looks safe, 16 almost never binds. Caveat: an alert stream of 2 questions each, not what a planner asks.

## A4. T2's real-data runs (done on branch `t1/week7-9`)

Back-tests are on the **train** alerts (the LRs are fitted on the past and applied to val; fitting them on val alerts would leak into every val result); `--eps 0.25 0.5 1 2` (the planners' eps levels), 110 expert programs (both subject slots). Commands, with the namespaces of the caches in `data/backtest/<ds>/<ns>/` (each has `report.json` and `lr_table.jsonl`):

```
python scripts/backtest.py --data <ds> --split train --eps 0.25 0.5 1 2 [--assemble]     # realistic
python scripts/backtest.py --data <ds> --split train --exact-labels [--assemble]         # exact mode proper (every bank's labels, no noise)
python scripts/backtest.py --data <ds> --split train --exact [--assemble]                # no noise, same alerts (for the eps sweep)
python scripts/backtest.py --compare <realistic_dir> <exact_dir>
python scripts/eps_sweep.py --system eig --data <ds> --split val --table-noisy <ns_realistic>/lr_table.jsonl --table-exact <ns_exact_same_alerts>/lr_table.jsonl --out data/runs/eps_sweep_<ds>.json
python scripts/make_decision_set.py --data <ds> --table <ns_realistic>/lr_table.jsonl --threshold 0.3 --max-total <n>
python scripts/g2_check.py --world real --data <ds> --val-part eval [--fusion per_family] --backtest-dir <ns_realistic>
```

| | HI-Small | HI-Medium |
|---|---|---|
| askable train alerts (partner bank in the 30 modelled views) | 1,038 (4,160 without) | 1,983 (8,979 without) |
| realistic table | `4338ebb59b35`, 440 hypotheses, **1 accepted**, about 10 min | `7df6bad5337d`, 440 hypotheses, **23 accepted**, about 1 h 40 min (same namespace as T3's table, so the alert set and configuration are identical) |
| exact-labels (8,158 / 8,396 sampled alerts, no noise) | `2d6016dc09d2`, 110 hypotheses, 106 eligible, 4 accepted | `a829616cb2a1`, 109 hypotheses so far (`n_repeated(ALERT.sender, window=around(72))` was still running when this was written; re-assemble when `data/backtest/hi_medium/a829616cb2a1` has 110 entries), 105 eligible, **16 accepted** |
| exact, same alerts (eps inf) | `75f9f282dc82`, 110 hypotheses, 7 accepted | `650db027008b`, 110 hypotheses, 38 accepted |

Runtimes are wall clock on the laptop with other jobs running; the exact-labels runs are dominated by five to six set questions (`n_repeated`, `n_round`, `n_format` over 8,000 alerts) that take hours each, so the last ones were started as separate processes with `--questions-file` into the same cache (no code change).

**`--compare` realistic vs exact-labels** (matched by program; the realistic entries are at four eps levels): HI-Small 424 pairs, accepted 1 vs 16, both 1, strength rank correlation 0.21, median strength ratio 0.98; **HI-Medium 420 pairs, accepted 23 realistic vs 64 exact, accepted in both 9, only realistic 14, only exact 55, Spearman 0.10, median strength ratio realistic/exact 0.67** (noise and the narrower, harder alert set cost a third of a question's strength). Against the same-alerts exact run: HI-Small 1 vs 28, ratio 0.66; HI-Medium 23 vs 152, ratio 0.47, Spearman 0.19. Files `data/runs/a4_compare_<ds>_{labels,same}.json`.

**HI-Small realistic table, accepted question.** `cross_currency_out(ALERT.sender, window=after(72))` at eps 2 (e-value 28,398; LR 3.83 for "true", 0.54 for "false"). **Flag for T2:** it looks at the 72 h *after* the alert time; check that a live answer cannot see the future (the back-test ledger is the train split, which is why it works there).

**eps sweep** (`scripts/eps_sweep.py`, EIG planner, val alerts through the harness; Brier gain = Brier(p0) minus Brier(final), positive = better), `data/runs/eps_sweep_<ds>.json`:

| per-alert budget | HI-Small gain (p0 Brier 0.1473) | HI-Small questions/alert | HI-Medium gain (p0 Brier 0.1704) | HI-Medium questions/alert |
|---|---|---|---|---|
| 0.5 | 0.0000 | 0 | 0.0000 | 0 |
| 1 | 0.0000 | 0 | -0.0100 | 0.94 |
| 2 | +0.0035 | 0.50 | -0.0058 | 1.56 |
| 4 | +0.0035 | 0.50 | -0.0155 | 2.22 |
| no noise | +0.0091 | 2.33 | -0.0287 | 6.0 |

(theoretical LR ceilings 1.65, 2.72, 7.39, 54.6 for 0.5, 1, 2, 4.) On HI-Medium the naive-fusion planner is **worse than p0 at every budget that asks anything, and worse still without noise**: more questions make it worse, so this is the fusion rule (T4), not the noise. G2 says the same below.

**Decision set for T3** (`data/decisions/<ds>/{decisions_train,decisions_val}.jsonl`, `manifest.json` with sha256; not committed). HI-Medium: **73,242 rows** (train 59,450 + val 13,792, the full set at 30 decisions per alert; the same 13,792 val rows T3 already used). Asked for 50,000: `--max-total 50000` stops after 50,000 train rows and **writes no val rows at all** (the cap is applied before the val split), so that version is kept only as `data/decisions/hi_medium_50k_trainonly/`; send T3 the full set (29 MB). HI-Small: 36,792 rows. Threshold 0.3 for the `stop` labels.

**G2 on real data** (`scripts/g2_check.py`, eps 2, 4 questions of 0.5, fixed-order expert planner; gain over p0, cluster bootstrap CI, need >= 0.01):

| | alerts (pos) | questions/alert | Brier p0 -> after | gain [95 % CI] | verdict |
|---|---|---|---|---|---|
| HI-Small val eval, naive fusion | 87 (17) | 0.5 | 0.1514 -> 0.1568 | -0.0053 [-0.0149, +0.0042] | FAIL |
| HI-Small, per-family fusion | 87 (17) | 0.5 | 0.1514 -> 0.1577 | -0.0063 [-0.0214, +0.0095] | FAIL |
| HI-Medium val eval, naive fusion | 160 (52) | 1.0 | 0.1985 -> 0.1994 | -0.0008 [-0.0135, +0.0125] | FAIL |
| **HI-Medium, per-family fusion** (cross-fitted) | 160 (52) | 1.0 | 0.1985 -> 0.1856 | **+0.0129** [-0.0342, +0.0637] | point estimate passes, CI far from excluding 0 |
| HI-Medium, all val parts, naive fusion | 460 (208) | 1.0 | 0.1704 -> 0.1719 | -0.0015 [-0.0107, +0.0077] | FAIL; the run prints "WARNING: a budget or gate breach in the run" (for T2/T4) |

The ceiling check passes everywhere (1,037 cells on HI-Medium, 0 violations). So G2 is **not met on real data**: the point estimate is positive only with per-family fusion on HI-Medium and its CI covers 0 widely (160 askable alerts); 23 accepted questions exist but a planner that asks one question per alert barely moves p0. For T4: the naive summed fusion moves p0 the wrong way on real LRs (as T3 also found).

## A5. T4's `scripts/eval_report.py` on the current real p0

Val `eval` part, paired bootstrap 10,000 resamples. "alerts" = the 332 / 806 val alerts the agent sees; "rows" = every scored bank-view row. Brier(one bank) minus Brier(see everything), positive = see-everything better; CI over attempt+accounts clusters (T4's definition).

| HI-Small | n (pos) | one-bank Brier | see-everything Brier | gap [95 % CI] |
|---|---|---|---|---|
| GFP, alerts | 332 (49) | 0.1093 | 0.0682 | +0.0411 [0.0104, 0.0748] |
| GFP, rows | 93,240 (122) | 0.001167 | 0.000906 | +0.00026 [0.00011, 0.00040] |
| Multi-GNN, alerts | 332 (49) | 0.0901 | 0.0333 | +0.0567 [0.0341, 0.0816] |
| Multi-GNN, rows | 93,239 (122) | 0.001110 | 0.000719 | +0.00039 [0.00020, 0.00060] |

PR-AUC one-bank / see-everything: GFP alerts 0.449 / 0.775, rows 0.202 / 0.419; Multi-GNN alerts 0.576 / 0.871, rows 0.243 / 0.548. Row-level CIs with the hub cap 20 and the attempt+tx clustering are in `data/runs/eval_gap_hi_small_*.json` and `data/runs/eval_multi_gnn_hi_small_*.json` and barely differ. The Multi-GNN rows come from `scripts/eval_report_models.py` (a wrapper around T4's `report`, because `eval_report.py` reads only the GFP p0 file).

**HI-Medium, GFP one-bank only** (no see-everything p0: BLOCKED-COMPUTE, `ftm/baselines/README.md`): `scripts/eval_report_models.py --data hi_medium`:

| | n (pos) | Brier | constant Brier | PR-AUC | Brier minus constant [95 % CI, attempt+accounts] |
|---|---|---|---|---|---|
| alerts (val eval) | 806 (198) | 0.1712 | 0.2443 | 0.553 | -0.0731 [-0.1190, -0.0271] |
| rows (val eval) | 320,372 (527) | 0.001451 | 0.001636 | 0.233 | -0.00019 [-0.00028, -0.00000]; attempt+tx: [-0.00032, -0.00007] |

**Cluster check on HI-Medium (decision 6 of `docs/t4_decisions_week2.md`: "repeat on HI-Medium val, if `largest_share` passes 0.2 switch to the hub cap").** Largest cluster share of attempt+accounts: **alerts 1.5 % (val eval, 806 alerts, 650 clusters), 2.5 % (all val 2,238 alerts, 1,417 clusters)**: fine. **Rows: 58.4 %** (one component of 187,047 of 320,372 rows through hub accounts; HI-Small rows were 16.3 %), above the 0.2 cut. With a hub cap of 20 accounts the largest cluster is 0.17 % (98,481 clusters); attempt+tx gives 0.004 %. So headline numbers on alerts can use attempt+accounts as decided; row-level numbers on HI-Medium need the hub cap (N = 20 here, to be fixed in `prereg/` by T4 before test).

## Files and who asked

| item | files | asked by |
|---|---|---|
| A1 | `ftm/intake/alerts.py`, `scripts/make_alerts.py`, `ftm/retrieval/views.py`, `scripts/check_alert_context.py`, alert parquets | T2, T3 |
| A2 | `ftm/baselines/gfp_gbt.py` (`top_contributions`, `readable_names`, `score_graph(explain_tx=)`), `scripts/top_features.py`, `data/alerts/<ds>/top_features_*.parquet` | T4 |
| A3 | `data/runs/cap_sweep_*.json` (T2's script) | T2 |
| A4 | `data/backtest/`, `data/runs/a4_backtest.sh` (T2's scripts) | T2 |
| A5 | `scripts/eval_report_models.py`, `data/runs/eval_*.json` (T4's script) | T4 |
