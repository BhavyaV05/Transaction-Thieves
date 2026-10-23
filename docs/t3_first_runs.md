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

## After fixes (2026-10-23, branch `t3/planner-quality`)

Same 20 HI-Medium val-fit alerts (hash `7118abb7f8f5bc1b`), same seed 0, same back-test table, Qwen3-8B-AWQ, thinking off, harness world. n=20, descriptive only. Run dirs: `outputs/t3_first/runs/{planner20_after, planner20_after_nomenu, react20_after}` (the "before" columns are `planner20`, `react20` above). `planner20_after_nomenu` has every fix except the select menu (`--no-select-menu`), to separate the two effects. The typology re-score ran the full report once (one pass, no iteration on it); prompt variants for typology were compared on 240 val-fit rows only.

### Root causes and fixes

| # | Finding | Root cause | Fix |
|---|---|---|---|
| 1 | Planner asked 0 accepted questions | The compose prompt never showed the back-tested questions: `run_llm_agent.py` passed no `questions`, so "select" did not exist and the 8B composed free-form programs (33 of 79 landed in the table, 0 of the 23 accepted rows). tracks.md 5.4: the planner may *select* a library question or *compose* one, and only back-tested ones move the posterior. | `accepted_menu()` (ftm/llm/backtest_table.py): one item per accepted canonical program (accepted eps levels merged, neutral order, ids B01..), shown as "Back-tested questions (select by id, or compose a new one)" with the eps it may be asked at and its LR per answer (per eps when several). Filtered per alert by slot binding (`applicable_to`: only `ALERT.sender/receiver` are bound, set-slot questions are hidden). A selected question is asked at an accepted eps (nearest; logged as `eps_requested`). `no_backtest` strips eps and LRs; `fixed_list` now has a list. The per-mode JSON schema still constrains every field. |
| 2 | 36/79 planner and 23/76 ReAct gate refusals | Harness-world mismatch, not the gate. All 36 planner refusals were `GateBlocked: bank N is not a partner bank`: the prompt listed the 29 banks of the bank table, the harness models 27. The models picked unmodelled banks (e.g. 114) and, since refusals did not count as failures, repeated them (the two 20-call loops). | `run_llm_agent.py` intersects `partner_banks` with the harness banks for every system (`modelled_banks`). Refusals now count toward the 3-in-a-row stop rule, the gate's reason is fed back (and shown in ReAct's observation), a refused (question, bank) is not re-sent, and a "used up" refusal (clock or eps) ends the loop. |
| 3 | 18 invalid DSL (14 bare function names), 17 duplicates | The 8B wrote a bare function name into `program` and ignored the one-line error. | (a) `program` has a regex in the guided-decoding schema (outline only; T2's validator still decides). (b) Retry feedback = the error plus a valid example of the function it tried to use, or the closest functions with examples (`retry_hint`, read from T2's reference text; ReAct uses it too). (c) Output section shows a full-call example; the system prompt says a (bank, question) is asked once and an eps-e question uses e. (d) Duplicate feedback names a bank not yet asked that question. |
| 3b | (new) 7 of 20 first prompts over the cap once the menu was added | `estimate_tokens` (3.2 chars/token) under-counts evidence rows (aliases and amounts tokenise at about 1.7 chars/token): the server counted up to 22 percent more. | Trim loop adds 0.30 tokens per evidence character. Largest prompt 5,905 of 6,000, 0 `PromptTooLong`. |
| 4 | ReAct: 10 of 20 never finished | 5 ended on a timeout/decline that pushed the simulated clock past 5 days (loop stops, no way to answer); 3 looped on gate refusals (cause 2); 2 hit the shared 6,000-token prompt cap as the history grew. `finish[...]` parsing was not the problem. | After the loop, if calls remain, ReAct makes **one** wrap-up call (inside the same 20-call cap, no extra budget) with a compact prompt (system + case + one line per question asked) that asks only for `finish[p]`. Plus causes 2 and 3. |
| 5 | Typology accuracy 0.01 to 0.12 | Not a token collision: the 9 letters A..I are distinct single tokens and are read correctly. The 8B answered `A` (= `fan_in`) for 240 of 240 sampled val-fit rows because the prompt gave only names and no way to say "no pattern". Separately the typology contexts hold no signal (below). | `Decision.hint`: one-line definitions of the 8 patterns and "Choose none when the text shows no clear pattern" (inference and SFT examples alike). |

### Typology after the prompt fix (4B fast tier and 8B large tier re-scored on typology rows only; other rows reuse the cache)

| | before | after |
|---|---|---|
| 4B argmax accuracy (eval) | 0.403 | 0.679 |
| 8B JSON / logprob accuracy | 0.124 / 0.013 | 0.546 / 0.679 |
| fitted T | 4.59 | 7.66 |
| kept by 4B / to 8B / default | 0 / 0 / 1.0 | 0 / 0 / 1.0 |

0.679 is the share of `none` among eval typology rows: both models now answer `none` almost everywhere, so the improvement is the majority-class rate, not signal. The typology contexts in `decisions_*.jsonl` are the alert's own single payment ("1 payments in from 1 home-side accounts ..."), the same shape for all nine labels, and each alert appears 4 times (190 val-fit alerts, 760 rows). No model can recover the pattern from that text; the cascade still defers 100 percent. This is a dataset issue for T2. `stop` (T = 20.0) is not a prompt bug: about 81 percent of labels are yes, driven by "Answers so far (n)", eps left and posterior vs threshold in the context, which the zero-shot 4B does not use; left for fine-tuning. Files: `outputs/t3_first/fast_after/`.

### Before / after (20 alerts)

| | planner before | planner after (menu) | planner after, no menu | ReAct before | ReAct after |
|---|---|---|---|---|---|
| valid-DSL rate | 0.842 | 1.000 | 0.986 | 0.96 (asks not rejected as invalid) | 0.975 |
| questions asked (reached gate) / per alert | 79 / 3.95 | 21 / 1.05 | 58 / 2.90 | 76 / 3.80 | 44 / 2.20 |
| accepted-question count (asked with an accepted LR) | **0** | **19** (19 of 20 alerts) | 0 | 0 | 1 |
| gate refusals | 36 | 0 | 1 | 23 | 0 |
| duplicates | 17 | 1 | 11 | 2 | 5 |
| alerts at the 20-call cap | 2 | 0 | 0 | 1 | 0 |
| verdicts without probability | 0 | 0 | 0 | 10 | 0 |
| Brier (p0 Brier 0.0819) | 0.0819 | 0.1148 | 0.0819 | 0.1429 | 0.2436 |
| seconds per alert, median / p90 / max | 7.5 / 11.5 / 35.2 | 2.0 / 2.1 / 5.9 | 5.7 / 10.6 / 13.5 | 4.8 / 6.7 / 15.8 | 4.5 / 6.4 / 7.2 |
| LLM calls per alert | 5.7 | 1.1 | 3.5 | 6.1 | 5.2 |
| largest prompt (tokens; ReAct: largest per-alert mean) | 5,515 | 5,905 | 5,646 | 5,702 | 5,656 |

Reading (n=20, nothing here is a result):
- The menu fixes coverage (0 to 19 accepted questions), but the 8B then asks the same question on 18 of 20 alerts (`n_repeated(ALERT.receiver, window=around(168))` at eps 2, the widest LR range in the menu), which spends the whole eps 2 budget, so it asks one question and stops. That is a rational read of the displayed statistics, not alert-specific reasoning. With accepted LRs live, the posterior is worse than p0 on these 20 (Brier 0.1148 vs 0.0819; 14 of 20 positive, p0 already informative). Whether the table's LRs help on a larger set is T4's fusion question.
- ReAct now always answers; its Brier is worse because its own probabilities are unfused and uncalibrated (the 10 missing verdicts used to be scored as p0, which flattered it). Fairness kept: same 20-call and prompt caps, the wrap-up call comes out of the same budget, and ReAct does not see the back-test menu (that is the planner's own mode).
- No-menu planner: validity 0.84 to 0.99 and the refusal loops are gone, but it still composes outside the accepted rows (0 accepted of 58 asks).
- Latency: one question per alert puts the menu planner at 2 s, below the 8 to 20 s target; the no-menu planner (5.7 s median, 10.6 s p90) is closer. Not tuned.
- Overfitting guard: every change is general (bank list, menu from the table, token estimate, guided pattern, feedback text); nothing refers to an alert id or a specific question.

### GPU time

Server up: Qwen3-8B 1,493 s (reruns, typology experiments, 8B typology scoring) + Qwen3-4B about 100 s = 0.44 h; GPU back at about 1.6 GB idle.

### For T2 / T4

- T2: (1) `partner_banks` of a real alert come from the bank table (29) but the harness models 27; expose the modelled banks (and bindings) publicly. Repro: hi_medium val alert V000553, `ask_bank(114, "n_tx(ALERT.sender, dir=in, window=before(72))", 0.5)` raises `GateBlocked: bank 114 is not a partner bank` although 114 is in the alert's partner list. (2) Typology decision contexts do not contain the evidence and repeat 4 rows per alert. (3) One timeout of 3.4 days leaves 1.6 days, which ends most sessions after 2 questions; confirm that is intended.
- T4: with live accepted LRs the naive summed fusion moves p0 the wrong way on these 20; check `FusedLR` fitted on val. Should Group D ports see the back-tested question list (plan.md 7.2)? Only our planner does now.
- Group D: Reflexion, ReWOO, LATS, LLMCompiler share `ToolExecutor` (retry hint, refusal reason) but only ReAct got the wrap-up call.
