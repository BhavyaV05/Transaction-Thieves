# T3 first real runs (2026-10-23)

Owner: T3 (Akshat). Machine: 5070 Ti PC, native Linux, vLLM 0.31.0, thinking off (`enable_thinking: false` sent by `run_llm_agent.py`; confirmed by ~120 output tokens per planner call).
**Read these as engineering results, not findings.** Planner/ReAct: **20 alerts, descriptive only**, HI-Medium val `fit` part (the `eval` part, used for E18/G4 selection, was not touched). Fast model: **untuned zero-shot baseline** for E-cascade. Test split untouched, `FTM_FINAL_RUN` unset.

## 1. Planner and ReAct, Qwen3-8B-AWQ, harness toolbox, 20 val-fit alerts

Setup: `run_llm_agent.py --system {planner,react} --mode compose --llm endpoint --toolbox harness --alerts val --val-part fit --data hi_medium --n 20 --backtest data/backtest/hi_medium/7df6bad5337d/lr_table.jsonl` (run ids `planner20`, `react20`, under `outputs/t3_first/runs/`). Same 20 alerts for both (hash `7118abb7f8f5bc1b`); 14 of 20 are positive (alerts must be askable in the harness world, which selects mostly high-p0 alerts). Back-test table: 440 rows, 23 accepted (eps 1: 5, eps 2: 18; none at 0.25/0.5).

| | planner (compose) | ReAct |
|---|---|---|
| JSON parse rate | 1.00 (114/114 outputs) | n/a (text tool calls) |
| valid-DSL rate | 0.84 (96/114) | n/a |
| usable output rate (planner stat) | 0.84 | n/a |
| duplicates refused | 17 | n/a |
| questions asked (reached the gate) / per alert | 79 / 3.95 | 76 / 3.80 |
| eps spent total (mean per alert) | 23.5 (1.18) | 28.5 (1.43) |
| gate refusals | 36 | 23 |
| accountant / system-only refusals | 0 | 1 |
| harness replies / declines / timeouts | 38 / 2 / 3 | 48 / 0 / 5 |
| **accepted-question count** (asked with an accepted back-test LR) | **0** | **0** |
| questions in the table at all (accepted or not) | 33 of 79 | 27 of 76 |
| verdicts without probability | 0 | 10 of 20 |
| Brier (n=20) | 0.0819 | 0.1429 (10 missing verdicts scored as p0) |
| p0 Brier on the same 20 | 0.0819 | 0.0819 |
| seconds per alert, median / p90 / max | 7.5 / 11.5 / 35.2 | 4.8 / 6.7 / 15.8 |
| LLM calls per alert (mean) | 5.7 | 6.1 |
| tokens in / out per call (mean) | 4,479 / 121 | 4,580 / 100 |
| largest prompt (tokens) | 5,515 (cap 6,000) | not logged per call; largest per-alert mean 5,702 |
| `PromptTooLong` | 0 | 0 |

`ftm.eval.runs.score` on both run logs (200 bootstrap resamples, labels from the alert set): planner `brier_end_to_end` = `brier_p0` = 0.0819 (diff 1.5e-13), `no_verdict_rate` 0; ReAct `brier_end_to_end` 0.1429, `no_verdict_rate` 0.5, `brier_e2e_minus_p0` +0.061 (CI 0.002 to 0.125, worse than p0), `brier_questions_only` = p0.

Reading (n=20, nothing here is a result):
- The pipeline works end to end on real data with no code change: vLLM guided decoding (`response_format` json_schema), the qwen3 reasoning parser, the harness gate/accountant, back-test table loading and the run log all behaved. No fix was needed.
- The planner's probability equals p0 by construction here: in compose mode the verdict is the fusion posterior, and fusion only moves on answers whose (canonical, eps) has an accepted back-test LR. None of the 79 asks did (the 8B composes freely; 33 landed in the table, 0 in the 23 accepted rows). So the planner Brier cannot differ from p0 until it asks accepted questions (select mode, library, or the fine-tuned student). This is the main finding, and it is a coverage issue between a free composer and a 23-of-440 table, not a plumbing bug.
- Planner failure modes seen: of 18 invalid programs, 14 are syntax errors (the ones inspected were a bare function name with no arguments, e.g. `n_new_counterparties`) and 4 use invented functions; 17 duplicates; two alerts used all 20 calls looping on rejects. These are prompt/model quality issues (candidates for the grammar-in-prompt and for E16), not changed here.
- ReAct: 10 of 20 runs ended without `finish` (last observation a timeout, gate block or invalid question; one hit 20 calls), which scores as p0; the 10 that finished put 0.4 to 0.95 on alerts with p0 up to 1.0, which hurts Brier. Its own probabilities are uncalibrated and unfused.
- Latency: planner median 7.5 s is at the low edge of the 8 to 20 s target and p90 11.5 s is inside it; the 35 s maximum is a 20-call loop. ReAct is faster because it stops early or runs out of tries. Single user, no batching.

## 2. Untuned fast-model baseline for the cascade

Models: Qwen3-4B-AWQ as the fast tier (single-token logprobs, `LogprobFastModel`, thinking off); Qwen3-8B-AWQ as the large tier, two ways: `JsonDecisionModel` (8B writes `{label, confidence}`, the existing calibrate path) and `LogprobFastModel` pointed at the 8B. Data: `data/decisions/hi_medium/decisions_val.jsonl`, all 13,792 rows, joined to `val_part`. **No subsampling**: the endpoint batches concurrent requests, so the whole set took 80 s (4B, 172 rows/s with 32 workers) and 166 s / 84 s (8B JSON / logprob on cal+eval, the only rows the large tier needs). Because `train_planner.py calibrate` needs both servers up at once (one GPU holds one), the new `scripts/score_fast_baseline.py` caches each tier's outputs and then calls the same `calibrate_cascade`: temperature per decision on `fit`, conformal sets (alpha 0.05, both tiers) on `cal`, everything reported on `eval`. Rows: fit 5,726, cal 3,270, eval 4,796.

Cascade report on `eval` (8B tier = JSON baseline; kept = accepted by the 4B, large = sent on and accepted by the 8B, default = safe default; err = P(accepted and wrong), bound 2 alpha = 0.10; ECE is top-label ECE of the 4B with fitted T / T=1):

| decision | n fit/cal/eval | T | ECE T / T=1 | kept 4B | to 8B accepted | to default | err (marginal) | err given accepted | 4B / 8B argmax acc. |
|---|---|---|---|---|---|---|---|---|---|
| worth_asking | 1146/654/960 | 9.31 | 0.119 / 0.382 | 0.103 | 0.118 | 0.779 | 0.067 | 0.302 | 0.549 / 0.499 |
| typology | 760/436/636 | 4.59 | 0.213 / 0.241 | 0.000 | 0.000 | 1.000 | 0.000 | n/a | 0.403 / 0.124 |
| rule_applies | 1528/872/1280 | 8.12 | 0.061 / 0.219 | 0.276 | 0.014 | 0.710 | 0.061 | 0.210 | 0.766 / 0.612 |
| injection_present | 1146/654/960 | 7.30 | 0.059 / 0.218 | 0.487 | 0.001 | 0.513 | 0.044 | 0.090 | 0.754 / 0.628 |
| stop | 1146/654/960 | 20.0 | 0.155 / 0.514 | 0.057 | 0.405 | 0.538 | 0.059 | 0.128 | 0.454 / 0.695 |

With the 8B tier as single-token logprobs instead of JSON, the 4B columns are identical and the 8B part changes: to-8B-accepted worth_asking 0.063, rule_applies 0.000, injection 0.000, stop 0.121 (to default 0.822), typology 0 (8B argmax accuracy 0.013, 0.518, 0.563, 0.576, 0.442 for typology, worth_asking, rule_applies, injection, stop). Full numbers: `outputs/t3_first/fast/report_{json,lp}.json`.

Reading (untuned, zero-shot; the point is the baseline to beat):
- The marginal error bound holds for every decision (all `within_bound`), but trivially for typology (the 4B never commits: everything goes to the default) and weakly elsewhere: the fast tier handles at most 49 percent (injection) and as little as 0 to 10 percent (typology, stop, worth_asking). H5 ("most routine decisions") is not met untuned; this is what fine-tuning (E10) has to change.
- Temperature scaling helps a lot where the fit rows are informative (ECE 0.22 to 0.06 on rule_applies and injection) but `stop` hit the search bound (T = 20.0): the zero-shot 4B has no usable signal there (accuracy 0.45, near a coin flip). Treat that temperature as a flag, not a fit.
- Zero-shot accuracy is barely above chance on worth_asking and typology (typology has 9 labels, 4B 0.40, 8B 0.01 to 0.12; the 8B's letters are not being produced, probably a prompt-format mismatch rather than ignorance; not investigated).
- No decision type was flagged for a conformal fallback or missing fit rows.

## 3. QLoRA estimate (not run, nothing trained)

`train_planner.py train --config configs/llm/sft_fast.yaml --dry-run` on `decisions_train.jsonl` rendered with `to_sft_example` (59,450 examples, 1,983 alerts): estimated 193 tokens median, 255 max (the repo's rough estimator, not the real tokenizer), 0 over `max_seq_length` 2048; about 11.9 M tokens per epoch, 1 epoch, effective batch 32 = about 1,860 optimizer steps. QLoRA on a 4B model costs about 6 x 4e9 FLOP per token with gradient checkpointing (about 3e17 FLOP per epoch); at an assumed 25 to 40 percent of the 5070 Ti's bf16 throughput that is **about 1.5 to 3.5 h per seed, plan 2.5 h; 3 seeds about 7.5 h** (5 to 10 h). This is an arithmetic estimate, not a timed run.
Missing for a timed dry run: Unsloth, TRL, PEFT, bitsandbytes are not installed in the repo venv; the bf16 base `Qwen/Qwen3-4B` (about 8 GB, revision pinned in `sft_fast.yaml`) is not downloaded (the AWQ copy used for serving cannot be trained); `train()` has never been executed on a GPU. The planner config (8B, 7,168 tokens, 2 epochs) is much larger and needs the teacher traces first (not estimated).

## GPU time

Server up time (wall clock): Qwen3-8B 795 s (load, both planner runs, 8B scoring), Qwen3-4B 136 s (load, 4B scoring) = about 0.26 h; GPU confirmed back at 1.6 GB idle after each server. All of section 2 ran in minutes, so a rerun with a tuned adapter is cheap.

## Blocked / for other tracks

- Planner has nothing to fuse until it asks accepted questions: needs either a table with more accepted rows (T2/T4, more eps levels at 0.25 and 0.5 have 0 accepted) or select mode / library to steer it onto the 23. T4: fusion-params fit on val would change nothing without accepted LRs.
- T2: `askable_ids` reads the harness's private `bindings` (already noted in `run_llm_agent.py`).
