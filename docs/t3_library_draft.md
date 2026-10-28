# T3 draft strategy library (C12), 2026-10-24

Owner: T3 (Akshat). Engineering draft, **not frozen** (freezing is the G3 decision). Test split untouched, `FTM_FINAL_RUN` unset, val `eval` not read.

## Runs

Planner (`--mode compose`, select menu on, harness toolbox, realistic back-test table `data/backtest/hi_medium/7df6bad5337d/lr_table.jsonl`, `--keep-prompts`), Qwen3-8B-AWQ via vLLM, thinking off, seed 0, HI-Medium **train**. Estimate from the first 50 alerts: 113 s, so the full set (about 40 min per 1,000) fit far under the 2 h limit and **all 1,983 askable train alerts were run** (40 chunks of 50, `--offset` on `run_llm_agent.py`, chunk dirs `outputs/t3_library/runs/c000..c039`, finished chunks skipped on restart; none failed). Traces with prompts are in each `traces.jsonl` (reusable as teacher-format data).

- Run time: 4,717 s = **1.31 GPU-h**. Server up 5,826 s = 1.62 h (load plus about 0.3 h idle at the end while the chunk loop finished and before it was stopped); GPU back to 1.6 GB.
- 2,095 questions asked in total; 1,869 (89 percent) are `n_repeated(ALERT.receiver, window=around(168))` at eps 2, the other 226 spread over a few families (`n_new_counterparties` 116, `dormant` 72, `volume_spike` 13, ...).

## Library (`outputs/t3_library/library/shape_v2_draft.json`, hash `1089cb08c084aa13`, `frozen: false`)

`scripts/build_library.py --runs c0?? --data hi_medium --backtest <table>`, shape keying, n_min 20, 1,938 (alert, question) records (answered, allowed asks).

| | |
|---|---|
| alerts | 1,983 |
| shape buckets (v2 keys) | 248 |
| buckets with >= 20 alerts | 16 (cover 1,067 alerts) |
| strategy notes | 4 (success 0, failure 4) |

Top 10 shape keys (typology, payers in, payees out, banks, amounts, span; alerts):
1. any, 1, 0, 2, 1k-10k flat, <6h: 326
2. any, 1, 0, 2, 10k-100k flat, <6h: 215
3. any, 1, 0, 2, 100k+ flat, <6h: 123
4. any, 2-3, 4-7, 5+, 1k-10k varied, 1-3d: 63
5. any, 2-3, 4-7, 5+, 100-1k varied, 1-3d: 51
6. any, 1, 0, 2, 1k-10k flat, 1-3d: 49
7. any, 2-3, 8+, 5+, 1k-10k varied, 1-3d: 34
8. any, 4-7, 2-3, 5+, 1k-10k varied, 1-3d: 25
9. any, 2-3, 2-3, 5+, 100-1k varied, 1-3d: 25
10. any, 4-7, 4-7, 5+, 1k-10k varied, 1-3d: 24

Sample of the 4 notes (all failures; the question is accepted in the table but its mean Brier gain is negative with mean + 2 se < 0):
- (1 payer in, 0 payees out, 2 banks, 1k-10k flat, 1-3d) `n_repeated(ALERT.receiver, window=around(168))` did not help (-0.0491; 46 past alerts)
- (2-3 in, 4-7 out, 5+ banks, 1k-10k varied, 1-3d) same question, -0.0511; 61 alerts
- (2-3 in, 4-7 out, 5+ banks, 1k-10k varied, 6-24h) same question, -0.0559; 20 alerts
- (4-7 in, 4-7 out, 5+ banks, 1k-10k varied, 1-3d) same question, -0.0552; 24 alerts

## Honest reading

The planner collapses onto one menu question (as in `docs/t3_first_runs.md`), so the library mostly records that question's behaviour: no (shape, question, eps) group with another question reaches 20 alerts, there are **no success notes**, and the 4 failures say the single question, with its accepted LR, moves the naive fused posterior the wrong way in four shape buckets. That is the same T4 fusion issue as before, not evidence about shapes. A useful library needs a planner that spreads over several accepted questions (a fused-LR fit, E16/student, or a randomised menu order for data collection). Mean Brier on chunk 0 was 0.2246 (descriptive only).

## Rebuild on the E18 backbone (Gemma 4 12B), 2026-10-26

Same planner settings as above (compose, select menu, harness, table `7df6bad5337d`, `--keep-prompts`, seed 0), now with the E18 backbone `gemma4-12b` (Google QAT W4A16 via vLLM, thinking off). These are also the **teacher runs** for distillation (E16). All **1,983** askable HI-Medium train alerts, `scripts/chunked_runs.sh gemma4-12b outputs/t3_teacher/runs train 1983 50 --keep-prompts --backtest <table>`, 40 chunks `c000..c039`, none failed. 26 Oct 04:55-08:08, run time 10,771 s = **2.99 GPU-h** (median 5.7 s per alert). Val is not used: the library feeds the val experiments (E8, E16, Group D), so val labels stay out of it until the freeze decision (then train + val, as plan 6.13 says).

- Valid JSON 1.000, valid DSL 3,077 of 3,077 questions, 6 refusals, 254 duplicate asks.
- 2,823 asks (2,600 replies, 141 declines, 76 timeouts); **560 asks on 548 alerts had an accepted LR** (the 8B draft: 1,869 of 2,095 asks were one question). Spread: 30 distinct questions; the top six (shape-free) are `n_tx(receiver, in, before 72, max 3000)` 413, `new_account(receiver, 2)` 363, `n_tx(receiver, in, before 24, max 3000)` 261, `n_tx(receiver, out, after 24, max 3000)` 169, `volume_spike(receiver, 5)` 115, `matching_amount_out(receiver, 0.05)` 84.
- Train Brier 0.1875 vs p0 0.1841 (370 positives; naive fusion; descriptive).

| Library | File (`outputs/t3_teacher/library/`) | Hash | Records | Success | Failure |
|---|---|---|---|---|---|
| shape (default) | `shape_v2_gemma_train.json` | `65144b79082a6cb5` | 2,600 | 0 | 7 |
| account (E11 ablation) | `account_v2_gemma_train.json` | `4f53cda18c2baa0c` | 2,600 | 0 | 0 |

Both `frozen: false`. Same 248 shape buckets, 16 with >= 20 alerts. The 7 failure notes are all rejected questions in the `1 in, 0 out, 2 banks, <6h` buckets (`n_tx(receiver, in, ...)` x5, `at_least(n_counterparties(...), 3)`, `new_account(receiver, 5)`), gain 0 by construction (no LR, posterior unchanged). The account-keyed library has no group with 20 alerts, as expected.

**Why still no success note.** Gain per accepted question over all shapes (mean, se): `new_account(receiver, 2)` at eps 1, 338 asks, +0.0002 (0.0052); `volume_spike(receiver, 5)` at eps 2, 104, **-0.0348 (0.0069)**; `inflow_bucket(sender)` 30, +0.0066 (0.0121); `n_repeated(receiver, around 168)` 24, -0.0703 (0.0277); `n_banks(receiver, in, before 24)` 21, -0.0362 (0.0189). Only `new_account` reaches 20 alerts in a shape bucket: +0.0141 (0.0087, n 139), -0.0121 (0.0116, 68), +0.0075 (0.0114, 47), none past z = 2. So the planner now spreads its asks, but under naive fusion no accepted question helps reliably, and the ones that clearly hurt (`volume_spike`) are spread over buckets below n_min. The library will stay empty of successes until fusion changes (G2 / T4); the gains are read from the logged posteriors, so after that decision re-fuse these runs' steps from their logged `log_lr` and rebuild (no new LLM calls).
