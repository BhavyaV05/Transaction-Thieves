# Planner prompt budget under `--profile tuned` (worker PROMPTFIT, 2 Nov)

Problem (found by the FLAGS worker, `docs/t3_rag_bench.md` 4a): with the tuned table (`configs/fusion/headroom_v1_table.jsonl`, all 440
hypotheses accepted) the routed menu has about 51 questions, each with a bin -> LR map at four eps levels. The compose prompt is about 10,000
estimator tokens against a cap of 6,000, so the trimmer keeps ONE evidence row and 20 of 51 questions: the planner runs almost without
evidence and the evidence / rules arms cannot differ. This note fixes the rule for the repair BEFORE any number is computed, then records the numbers.

## 1. What changes (profile tuned only; `--profile none` is byte-for-byte the old behaviour)

1. **Menu ranked by expected Brier reduction** (`ftm/graph/menu_rank.py`, gain = `ftm.fusion.gain.expected_gain`, the same function the System 1
   `worth_asking` label uses; moved there from `ftm/orchestrator/decision_set.py`, which imports it). At every planner step, at the current
   fused posterior and the eps left, each routed (question, eps) that is not yet asked and affordable gets
   `gain = p(1-p) - sum_cell [p P(cell|launder)(1-p')^2 + (1-p) P(cell|clean) p'^2]` under the fitted ConditionalFusion
   (shrink 0.5, family weights). Items with gain <= 0 are dropped (in expectation they can only hurt); items with no cell probabilities in the table are dropped and counted;
   the top K by gain are shown, best first. Counts offered / shown / dropped (non-positive, no cells, beyond K) go to the trace (`menu`) and to `summary.json`
   (`planner.menu_budget`). One line per question (the id the schema enumerates) listing the kept eps levels, each with a compact map
   (`eps 2: <1:1.84 1:1.23 2:1.27 4:0.32`: lower bin edge, LR). One ranking serves the planner and the adapted arms (`PlannerView.menu_shown`).
2. **Evidence floor** (`min_evidence_rows`, N): the newest N trail rows are cut last. Trimmer order becomes strategies, menu questions (lowest gain first),
   surplus rules, and only then trail rows below N. `--evidence-min 0` is the old order (oldest rows first, down to one).
3. **Cap**: `max_prompt_tokens` is a profile setting. Rule: `cap = floor250((max_model_len - output_reserve) / (1 + overshoot))`, with max_model_len 8192
   (Gemma 4 12B and every E18 candidate, `configs/experiments/e18_backbone_sweep.yaml`), output_reserve 800 (the largest `output_token_budget`: a wave of 4,
   791 tokens), overshoot = the largest measured server/estimator ratio of Gemma 4 on trimmed prompts (6,308 server tokens for a prompt trimmed to 6,000
   estimated, `docs/t3_e8.md`: 1.0513) plus the 2% refit margin of the planner (`PROMPT_REFIT_MARGIN`) = 0.0713. `ftm.llm.profile.prompt_cap` computes it; `tests/test_profile.py` re-derives the YAML value.
   Profile none keeps `Limits.max_tokens` = 6,000. The cap is a `Limits` field, so under the profile it applies to every system equally (fairness protocol).
4. Evidence row format is NOT changed (aliases, times, amounts, banks, direction are all needed and the legacy text is pinned by tests); the budget is met by the menu.

## 2. Selection rule for K and N (fixed here, before the study runs)

Data: val `fit` + `cal` askable alerts of HI-Medium (tuned stack, harness world, mock LLM, first planner step: no answers, eps left 2.0, which is the largest prompt
of an alert). Never `eval`, never test. CPU only (`scripts/prompt_budget_study.py`).

Candidates: K in {4, 8, 12, 16, 24, 32, all positive-gain}; N in {0, 5, 10, 15, 25, 40}.

Quantities, per candidate K:
- **Coverage(K)** = mean over alerts of V_K / V_all, where V_S is the best total expected Brier reduction a budget of eps 2.0 can buy from the item set S: greedy by gain per eps,
  one eps level per question, gains taken at the first-step posterior (independent-gain approximation). Alerts with V_all = 0 are skipped.
  (Top-1 coverage is 1 by construction, because the ranking is the gain itself; it is not informative and is not used.)
- **Prompt tokens** of the UNTRIMMED prompt (cap 100,000, full trail, 5 rules): p50 / p95 / max of the estimator count times 1.0713 (the server-count factor of rule 3).

Per candidate N: **Recall(N)** = mean over alerts with at least one pattern edge in the 72 h window of (pattern edges among the newest N rows of the scorer_pcst trail) / (pattern edges in the window),
and the retention Recall(N) / Recall(all rows of the trail).

Rule:
1. **K** = the smallest candidate with Coverage(K) >= 0.90 whose p95 untrimmed prompt (estimated server tokens) is <= the cap from 1.3. If no candidate reaches 0.90 inside the cap, the largest K inside the cap.
2. **N** = the smallest candidate with Recall retention >= 0.95; if the full trail of that K's prompt fits the cap for >= 95% of alerts the floor never binds and N only matters in the tail.
3. The result is checked once: prompt tokens p50 / p95 / max with the chosen K, N and the cap, and the share of prompts the trimmer still touches.

GPU arms (cheap to add; the study above only picks the default): `menu_topk_{8,16,all}` and `evidence_min_{0,10,25}`, in `scripts/t3_rag_bench_report.py`.

## 3. Results

Study: `scripts/prompt_budget_study.py prompts|recall|report` on val fit (191 askable alerts) + cal (109), HI-Medium, tuned stack, mock LLM, first planner step.
Cap from 1.3: floor250((8192 - 800) / 1.0713) = **6,750** estimator tokens (= 7,231 server tokens at the 1.0713 factor).

**K (menu).** Untrimmed prompts (server-token estimate = estimator x 1.0713), 300 alerts:

| K | items shown (mean) | coverage of the eps-2 gain | p50 | p95 | max | prompts over the cap |
|---|---|---|---|---|---|---|
| 4 | 4.0 | 0.672 | 5344 | 6712 | 6788 | 0.000 |
| **8** | 8.0 | 1.000 | 5449 | 6819 | 6896 | 0.000 |
| 12 | 12.0 | 1.000 | 5575 | 6937 | 7024 | 0.000 |
| 16 | 16.0 | 1.000 | 5677 | 7044 | 7133 | 0.000 |
| 24 | 24.0 | 1.000 | 5904 | 7270 | 7375 | 0.087 |
| 32 | 32.0 | 1.000 | 6087 | 7469 | 7580 | 0.303 |
| all (positive gain) | 204.0 | 1.000 | 9363 | 10733 | 10820 | 1.000 |

Rule: smallest K with coverage >= 0.90 and p95 inside the cap -> **K = 8**. Reading: K = 4 loses a third of the buyable gain; from K = 8 on the eps-2 budget is
already served by the top 8 (the budget pays for 1 to 4 questions), so a larger K only costs tokens. No item had a non-positive gain at the first step
(posterior of an alert with no answers), the `dropped_nonpositive` counter fires at later steps and in the unit test. The trail has 22.5 rows on average (max 40) in these 300 prompts;
the fixed part of a prompt (system prompt, question language, output spec, rules) (alert, rules, system prompt, language, output spec) is about 3,400 estimator tokens.

**N (evidence floor).** Pattern-edge recall of the newest N rows of the scorer_pcst trail, 263 fit + cal alerts with pattern edges in the window (mean trail 27.2 rows):

| N | recall | retention vs the whole trail |
|---|---|---|
| 5 | 0.424 | 0.506 |
| 10 | 0.563 | 0.673 |
| 15 | 0.671 | 0.803 |
| 25 | 0.773 | 0.924 |
| **40 (= the whole trail, k)** | 0.836 | 1.000 |

Rule: smallest N with retention >= 0.95 -> **N = 40**, i.e. the whole trail is protected. With K = 8 no prompt of the 300 is over the cap (p95 6,819 of 7,231 server tokens), so
the floor costs nothing in the common case and only decides the order of cuts in the tail.

**Chosen:** cap 6,750, K = 8, N = 40 (`configs/llm/tuned.yaml`: `max_prompt_tokens`, `menu_top_k`, `min_evidence_rows`; `--profile none` keeps 6,000 / whole menu / old order; `--max-prompt-tokens`, `--menu-top-k {K,all,off}`, `--evidence-min` override).

### Token breakdown before / after (20 val fit alerts, mock LLM, `--profile tuned`, first planner prompt, estimator tokens; evidence includes the 0.30 per character surcharge)

`scripts/prompt_budget_study.py breakdown --tag before --cap 6000 --k off --evidence-min 0` and `--tag after`. "Other" = system prompt, budget, partner banks, question language, output spec, headings.

| part | before: mean | after: mean | after: p95 | after: max |
|---|---|---|---|---|
| Alert | 75 | 75 | 77 | 78 |
| Evidence | 97 | 1830 | 2531 | 2585 |
| Menu | 2454 | 418 | 427 | 429 |
| Rules | 345 | 345 | 364 | 364 |
| Strategies (no library in this run) | 0 | 0 | 0 | 0 |
| Answers so far | 4 | 4 | 4 | 4 |
| Other | 2951 | 2951 | 2952 | 2952 |
| **total** | 5926 (cap 6000) | 5623 (cap 6750) | 6328 | 6407 |
| evidence rows kept (mean, offered 29.9) | 1.0 | 29.9 | | |
| menu questions in prompt (of 51 programs) | 20 | 6.7 programs / 8 (question, eps) items | | |

Before, the menu took 2,454 tokens for 20 of 51 questions and the trail was cut to the alerted payment; after, the whole trail (29.9 rows) and the 8 best (question, eps) items fit with 340 tokens of the 6,750 to spare at the maximum (6,407).

### What this does not do
- The evidence row text is unchanged (aliases, times, amounts, banks, direction; the legacy text is pinned by `tests/test_evidence_api.py`): the menu was the problem, not the trail.
- Coverage uses the independent-gain approximation at the first-step posterior; the GPU arms `menu_topk_{8,16,all}` and `evidence_min_{0,10,25}` (`scripts/t3_rag_bench_report.py`) measure the end-to-end effect.
- The adapted arms (`*_adapted`) share the ranked top-K menu (`PlannerView.menu_shown`, `MAX_OPTIONS` 30 > K) and keep their own 3,000-character evidence window (about 28 rows, i.e. the whole trail in practice).
- The cap is a `Limits` field: under the tuned profile every system (Group C, D, adapted) gets 6,750; profile none keeps 6,000.
