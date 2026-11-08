# T3 RAG bench: the five retrieval components, with vs without (2 Nov 2026)

Worker RAGB. Files: `scripts/rag_offline_bench.py` (CPU, run), `scripts/t3_rag_bench.sh` + `scripts/t3_rag_bench_report.py` (GPU matrix and report, NOT run), `tests/test_rag_offline_bench.py`.
Hard rules kept: test never read; the offline bench reads TRAIN (time-blocked) and val `fit`+`cal` only, val `eval` is not read by it; the GPU matrix chooses on `fit`+`cal` and scores `eval` once for the frozen winners (`scripts/t3_rag_bench.sh eval` refuses a second run).

## 1. Selection rule (fixed here BEFORE any GPU run; implemented in `select_component`)

Data that choose: val `fit` (191) + `cal` (109) askable alerts, 300 alerts, the same ids in every arm of a group. Every arm differs from its group's default in ONE factor (a flag on top of `--profile tuned` + the frozen shape library). Per component:

1. Eligible arm = verdict rate >= 0.95 AND not collapsed. Collapse checks (all three reported for every arm): (a) `var(p_arm) / var(p0) < 0.5` (shrinking towards a constant), (b) AUC below p0's AUC, (c) a Brier gain over the default that comes with an AUC more than 0.01 below the default's (the gain is the base-rate shrink, not information).
2. **Tier 1**: an eligible arm whose paired cluster-bootstrap 95% CI of `Brier(arm) - Brier(default)` lies entirely below 0 replaces the default; if several, the one with the lowest Brier.
3. **Tier 2** (no tier-1 arm): an eligible arm whose CI upper bound is below +0.005 (non-inferior, margin fixed now) and that cuts mean prompt tokens per alert by at least 10% replaces the default; if several, the cheapest. This is "keep the simpler/cheaper option when nothing is better".
4. Otherwise the default stays. A WITHOUT arm (no evidence, no rules, no library, no memory, zero-shot) is judged by exactly this rule, so a component is dropped only if removing it is significantly better or non-inferior and cheaper.
5. Reported next to the rule, not part of it: all metrics with paired CIs vs the default AND vs p0, Holm-adjusted p within each group (family = the arms of one comparison group), decision metrics at the frozen threshold 0.3227.
6. Interactions: ONE final arm `planner_combined` = all chosen planner winners together (seeds 0 and 1), against the default, with the same CI and collapse checks. If it is not better than the default, the combination is not adopted and the single winners are reported as separate improvements only.
7. Seeds: every LLM call has temperature 0 (the client default), so there is no sampling variance; the remaining randomness is the harness DP-noise seed. The default is run with seeds 0 and 1 (`planner_default_seed1`) and the combined arm with seeds 0 and 1; a difference between arms smaller than the seed-0/seed-1 difference of the default is not read as an effect. No arm samples at temperature > 0.
8. Eval (160 askable val alerts) is scored once, for the frozen defaults and winners, descriptive only. Test: never.

## 2. The arms (`python scripts/t3_rag_bench_report.py plan`)

| arm | group | component | one-factor change vs the group default | est. GPU h (300 alerts) | status |
|---|---|---|---|---|---|
| planner_default (default) | planner | default | - | 0.21 | ready |
| planner_default_seed1 | planner | seed | - seed=1 | 0.21 | ready |
| ev_none | planner | evidence | --evidence-view none | 0.21 | ready |
| ev_amount_1hop | planner | evidence | --evidence-view amount_1hop | 0.21 | ready |
| ev_amount | planner | evidence | --evidence-view amount | 0.21 | ready |
| ev_scorer_v2 | planner | evidence | --evidence-view scorer | 0.21 | ready |
| ev_pcst_v1 | planner | evidence | --edge-scorer v1 | 0.21 | ready |
| rules_none | planner | rules | --rules none | 0.21 | ready |
| rules_fixed | planner | rules | --rules fixed | 0.21 | ready |
| rules_whole | planner | rules | --rules whole | 0.21 | ready |
| lib_none | planner | library | library=none - | 0.21 | ready |
| lib_account | planner | library | library=account - | 0.21 | ready |
| lib_shape_bt | planner | library | library=shape_bt - | 0.21 | ready |
| sharma_native_mem_on (default) | sharma_native | memory | - | 0.14 | ready |
| sharma_native_mem_off | sharma_native | memory | --no-memory | 0.14 | ready |
| sharma_ported_mem_on (default) | sharma_ported | memory | - | 0.21 | ready |
| sharma_ported_mem_off | sharma_ported | memory | --no-memory | 0.21 | ready |
| torres_native_mem_on (default) | torres_native | memory | - | 0.46 | ready |
| torres_native_mem_off | torres_native | memory | --no-memory | 0.46 | ready |
| torres_ported_mem_on (default) | torres_ported | memory | - | 0.59 | ready |
| torres_ported_mem_off | torres_ported | memory | --no-memory | 0.59 | ready |
| co_investigator_native_mem_on (default) | co_investigator_native | memory | - | 0.72 | ready |
| co_investigator_native_mem_off | co_investigator_native | memory | --no-memory | 0.72 | ready |
| co_investigator_ported_mem_on (default) | co_investigator_ported | memory | - | 0.78 | ready |
| co_investigator_ported_mem_off | co_investigator_ported | memory | --no-memory | 0.78 | ready |
| finfre_rag_native_pool_bal (default) | finfre_rag_native | finfre_pool | - | 0.05 | ready |
| finfre_rag_native_pool_nat | finfre_rag_native | finfre_pool | --pool-pos-frac -1 | 0.05 | ready |
| finfre_rag_native_pool_033 | finfre_rag_native | finfre_pool | --pool-pos-frac 0.33 | 0.05 | ready |
| finfre_rag_native_zero_shot | finfre_rag_native | finfre_pool | --no-exemplars | 0.05 | ready |
| finfre_rag_ported_pool_bal (default) | finfre_rag_ported | finfre_pool | - | 0.13 | ready |
| finfre_rag_ported_pool_nat | finfre_rag_ported | finfre_pool | --pool-pos-frac -1 | 0.13 | ready |
| finfre_rag_ported_pool_033 | finfre_rag_ported | finfre_pool | --pool-pos-frac 0.33 | 0.13 | ready |
| finfre_rag_ported_zero_shot | finfre_rag_ported | finfre_pool | --no-exemplars | 0.13 | ready |

Total: 9.3 GPU-h for the 33 arms runnable today (all flags exist since the FLAGS commit), 9.5 GPU-h for all 34 once the library is rebuilt (fit+cal, 300 alerts each; planning figure: old-stack seconds per alert / 2.5 for concurrency 4). Not included: the library teacher runs (1,983 train alerts, about 1.4 GPU-h), the combined arm with two seeds, the once-only eval.

Why one factor at a time and not the full cross product: the factors act on different parts of the prompt; a cross product of the planner factors alone (5 evidence x 4 rules x 4 libraries) is 80 arms (about 17 GPU-h at 0.21 h each), and a 300-alert paired comparison has no power for three-way interactions. The one-factor design answers "does this component help, and which option is best", and the single `planner_combined` arm checks that the winners do not interfere.

## 3. What each component was adapted for (from git history and docs)

### 3.1 Adaptation history, one line per component

| Component | Adapted for our use case (git / docs) |
|---|---|
| C11 evidence views | Built by T1 (`ftm/retrieval/evidence_views.py`): the alert's 72 h bank-view edges, K = 40, time guard, aliased account ids, a time-respecting PCST trail that is one connected, ordered trail ending at the alerted payment. Edge scorer v2 (`edge_scorer_home.py`: per-home-bank features, out-of-fold train scores) replaced v1's global in-sample scores. The planner used a week-2 stub (top 40 by amount) until the tuned profile switched it to `scorer_pcst` with scorer v2 (`configs/llm/tuned.yaml`, rule `pick_evidence_view` on fit+cal). E8 (`docs/t3_e8.md`) could not separate the views: under naive fusion and wrong-bank answers all four were within 0.005 Brier and above p0. |
| C6 regulatory RAG | Plain-python BM25 over 4 legal-basis rules + 44 red flags (`ftm/compliance/rag.py`). Audit finding 15: one fixed sentence gave every alert the same top 5, so `rules_query(alert)` builds the query from jurisdiction, format, currency, amount band. `rules_first` pins the citable rules (the 4 that permit ask_bank) ahead of red flags; hybrid bge-small behind a config switch; only `kind == rule` may be cited. The jurisdiction filter, not the ranking, decides citation accuracy (docs/report 5.4.3). |
| C12 strategy library | Keyed by a coarse situation shape (v2: from ALL 1-hop receiver edges in 72 h, not the evidence rows, so it is the same under every view); search = exact key, then same typology, then nearest within `max_dist` 2.0 (a far shape gets nothing); notes only from back-test-accepted questions with n >= 20 and mean +- 2 se (no tautological "rejected" notes); account keying (E11) builds with n_min 1. Frozen libraries: 9 failure notes, 0 success notes (shape), 0 notes (account); the teacher-only account library has 494 notes. |
| Case memory (Sharma, Torres, Co-Investigator) | Train alerts only (the class refuses anything else), stratified draw 500 at pos frac 0.5 so a nearest-per-class query meets a laundering case (finding 10), `--memory-p0` replaces the inflated out-of-fold train p0 (finding 17); features = p0 + 5 ordinal shape buckets. |
| FinFRE-RAG exemplars | Reimplementation of the paper's pipeline on alerts: RF feature importance, categorical equality filters, cosine on z-scores, top 20; train-only pool, leave-one-out on train queries, pool stratified 1:1 (`--pool-pos-frac 0.5`), score map calibrated on train. |

### 3.2 Offline results (CPU, `outputs/rag_bench/offline.json` / `.md`; 2,000 cluster-bootstrap resamples; train sample 3,000 alerts, val fit+cal 1,432 alerts; val eval and test not read)

All numbers are descriptive properties of the retrievers; the Brier effect of each component on the LLM pipeline is what the GPU matrix measures.

**(a) Evidence views, val fit+cal, 263 alerts with pattern edges, pattern-edge recall among the K = 40** (mean tokens in brackets):

| arm | recall@40 [95% CI] | connected trail | vs `amount_1hop` (paired) |
|---|---|---|---|
| NO evidence (alert summary only) | 0 (0 tokens) | n/a | n/a |
| amount (top-40 by amount, the old stub) | 0.008 [0.000, 0.019] (2,869) | 0 | -0.811 [-0.890, -0.722] |
| amount_1hop | 0.819 [0.732, 0.899] (985) | 1.0 | - |
| scorer, v1 / v2 | 0.451 / 0.411 (2,681 / 2,683) | 0 | -0.368 / -0.408 (CIs exclude 0) |
| scorer_greedy, v1 = v2 | 0.839 [0.749, 0.917] (1,215) | 1.0 | +0.020 [+0.004, +0.044] |
| scorer_pcst, v1 / v2 | 0.836 / 0.836 [0.747, 0.915] (1,214) | 1.0 | +0.017 / +0.017 [0.000 / +0.001, +0.041] |
| ppr (v1 = v2) | 0.707 (1,288) | 0.999, one trail 0.795 | -0.112 [-0.164, -0.067] |
| long_prompt | 0.016 (3,966) | 0 | -0.803 |

Reading: the plain top-40 views carry almost no pattern edge (the laundering attempt's other edges are not the largest ones), the connected 1-hop/PCST trails reach 0.82-0.84 against a ceiling of 0.904 (relevant edges reachable in the window). `scorer_pcst` beats `amount_1hop` by only +0.017 recall (CI lower bound ~0) for 23% more prompt tokens. Edge scorer v2 changes nothing for the PCST view (identical recall) and is WORSE than v1 for the top-40 `scorer` view (0.411 vs 0.451): the scorer adds nothing once a connectivity constraint decides the selection.

**(b) Regulatory RAG** (48 documents, 55 askable expert questions, 100 citation cases; hybrid bge-small measured):

| arm | recall@5 / hit@5 | recall@10 / hit@10 |
|---|---|---|
| bm25 | 0.484 / 0.836 | 0.701 / 0.927 |
| keyword (v0) | 0.489 / 0.800 | 0.631 / 0.909 |
| hybrid (alpha 0.5) | 0.484 / 0.818 | 0.707 / 0.945 |

Citation (top-ranked citable rule, 100 cases): valid = exact = 1.00 with the partner's jurisdiction filter for every retriever and every query wording (fixed or alert-aware, `rules_first` on or off); 0.52 without a jurisdiction filter (the US rule wins every tie); whole corpus with a uniform pick 0.25 (48 documents, ~1,670 tokens in the prompt against ~290 for the top 5); NO rules 0 (nothing to cite). On the 1,432 real fit+cal alerts (BM25, k = 5): the fixed sentence returns ONE top-5 set for all alerts; the alert-aware query returns only 7 distinct sets, and the share of typology alerts (587) whose top 5 holds a red flag of the alert's true typology FALLS from 0.504 (fixed) to 0.307 (alert-aware); keyword retrieval: 0.504 -> 0.588; hybrid: 0.504 -> 0.060. `rules_first` changes nothing here (the fixed base sentence already ranks the rules in the top 5). Proposed typology cue from the shape (many payers in, few out = fan-in, ...) is given for only 14% of the alerts and is right 14.5% of the time (shape buckets of the receiver do not encode the typology), so it does not repair the query: BM25 0.341, keyword 0.574.

**(c) Strategy library** (1,432 fit+cal alerts; realised gain of a note's question = single-question Brier gain from p0 using the train-fitted table LR and the alert's own answer bin; 61,382 alert x question rows):

| library | notes | hit rate (cutoff 0 / 2 / inf) | success notes: realised gain [CI] | failure notes: realised gain [CI] |
|---|---|---|---|---|
| none | 0 | 0 | n/a | n/a |
| shape, frozen (production) | 9 | 0.40 / **0.50** / 1.00 | none exist | +0.0032 [-0.0033, +0.0108] on 39 alerts (a "hurt" note whose question did not hurt) |
| shape, teacher-only v2 | 0 | 0 | n/a | n/a |
| account, frozen | 0 | 0 | n/a | n/a |
| account, teacher-only | 494 | **0.053** (76 alerts) | +0.0291 [-0.0394, +0.0743] on 14 alerts | -0.0054 [-0.0411, +0.0315] on 19 alerts |
| shape, built from TRAIN back-test rows (n_min 20) | 69 | 0.40 / 0.50 / 1.00 | -0.0098 [-0.0245, +0.0044] on 109 alerts | none |
| shape, same, n_min 10 | 131 | 0.44 / 0.70 / 1.00 | -0.0026 [-0.0151, +0.0075] on 206 alerts | none |

The frozen shape library fires on half the alerts but only with failure notes whose sign does not hold on fit+cal; the account library almost never fires (5%); the back-test-built shape library (no LLM, no collapse onto one question) has success notes that do NOT transfer to fit+cal (point estimate negative): the notes are learned at a 13% base rate on train and applied at 42% on val, and a single-question gain depends on the base rate.

**(d) Case memory** (memory = TRAIN only; blocked: first 60% of train by time vs the later 40%; production: whole train sample vs val fit+cal queries; neighbour-label vote over k = 10, and the Group C "2 nearest per class" distance margin; p0 alone has AUC 0.843 [0.816, 0.867] on val, 0.791 on later train):

| memory | train-blocked AUC vote / margin | val AUC vote [CI] | val AUC margin [CI] | AUC gain of p0 + score over p0 alone (val) |
|---|---|---|---|---|
| natural (3,000 rows, 13.7% pos) | 0.615 / 0.700 | 0.632 [0.601, 0.660] | 0.624 [0.593, 0.654] | vote -0.000 [-0.010, +0.010]; margin **+0.018 [+0.011, +0.024]** |
| balanced 0.5 (500 rows) | 0.625 / 0.697 | 0.553 [0.521, 0.583] | 0.588 [0.553, 0.622] | vote +0.003 [-0.006, +0.013]; margin +0.008 [-0.001, +0.017] |
| NO memory | n/a | p0 only 0.843 | | |

Non-blocked leave-one-out on train gives a vote AUC of 0.805 against 0.615 time-blocked: a memory looks informative only when it contains the alert's own ring. Every memory score is far below p0 (differences -0.21 to -0.29 AUC, CIs exclude 0). It carries a small rank signal beyond p0 only in the natural-rate memory's distance margin (+0.018 AUC), and no Brier gain (the refit probabilities are worse in Brier because the val base rate is 42%).

**(e) FinFRE-RAG pool** (500 train exemplars, top 20 retrieved, vote = positive share; val fit+cal):

| pool | positive share of the 20 exemplars | queries with >= 1 positive exemplar | val vote AUC [CI] | train-blocked AUC | AUC gain over p0 |
|---|---|---|---|---|---|
| zero-shot (none) | 0 | 0 | 0.5 (p0 0.843) | n/a | n/a |
| natural (12.6%) | 0.15 | 0.72 | 0.602 [0.567, 0.637] | 0.556 | +0.002 [-0.002, +0.007] |
| pos 0.33 | 0.36 | 0.84 | 0.585 [0.546, 0.621] | 0.599 | +0.005 [-0.001, +0.012] |
| pos 0.5 (default) | 0.52 | 0.91 | 0.589 [0.552, 0.624] | 0.581 | +0.005 [-0.001, +0.011] |

Raising the pool's positive share fixes the exemplar balance (the 20 retrieved exemplars hold a laundering case for 91% of queries instead of 72%) but not the information: every vote AUC is ~0.59-0.60 against p0's 0.843, the differences between pool rates are inside the CIs, and the vote adds at most +0.005 AUC to p0. The pool contains p0 as a feature (`bank_score`), so its nearest-neighbour vote is dominated by p0-similar train alerts.

### 3.3 What still underperforms (offline evidence)

1. Evidence: a top-40 trail of the largest edges (the old stub and `scorer`) misses the pattern edges; only connected trails recall them, and the learned scorer (v1 or v2) does not beat the 1-hop rule on recall (+0.017, CI touching 0). The extra prompt cost of PCST is not supported by recall alone.
2. Regulatory: the alert-aware query is less on-topic than the fixed sentence for the typology (0.307 vs 0.504 under BM25) and varies little across alerts (7 distinct top-5 sets); citation validity is carried by the jurisdiction filter alone. In this corpus all banks map to US, so the cross-border words never fire.
3. Library: no success note exists in the frozen libraries; failure notes do not hold held-out; the account library hits 5% of alerts; train-built notes do not transfer across the 13% to 42% base-rate shift.
4. Case memory and FinFRE: neighbour votes have AUC 0.55-0.63 (0.58-0.60 for FinFRE), 0.2 below p0; at best +0.018 AUC beyond p0 and no Brier gain.

### 3.4 Modifications suggested by these numbers (implemented where contained in this worker's files)

| Suggestion | Status |
|---|---|
| Library built WITHOUT any LLM run from the TRAIN back-test rows (every alert x every accepted question, so it cannot collapse onto the planner's favourite question) | implemented: `build_backtest_library` in `scripts/rag_offline_bench.py`; frozen file `outputs/rag_bench/libs/shape_backtest_train.json` (hash dd79ed5aabe5e760, 69 notes), GPU arm `lib_shape_bt`. Offline it does NOT transfer, so the GPU arm is a test of whether it hurts. |
| Rebuild the libraries from TRAIN teacher runs of the tuned stack (the old frozen ones come from the old stack: naive fusion, stub view) | done by the rerun, not by this bench: `outputs/t3_library/frozen_v3/*_train.json` is built from the teacher_v2 runs (1,983 train alerts, the same tuned stack); the bench stage `library` and the arm `lib_shape_rebuilt` were removed on 7 Nov (they would have repeated those runs, ~1.4 GPU-h, under another name) |
| Condition note gains on the base rate (fit the gain on the logit scale, or build on a val-prevalence reweighted train) | NOT implemented (changes `ftm/library/build.py`, not mine); motivated by the sign flip above |
| Typology cue from the shape in `rules_query` | implemented as a measured arm only (`shape_typology_cue`); rejected by the numbers (cue given 14%, correct 14.5%). A typology cue needs a signal that is not the receiver's 1-hop shape (e.g. the planner's own hypothesis after step 1) |
| Query = fixed sentence + alert words, not alert words alone, and `typology_flag_hit` as the retrieval metric next to citation | measured; the GPU arms `rules_fixed` / `rules_none` / `rules_whole` decide whether the prompt benefit exists at all |
| Memory: use the distance MARGIN (2 nearest per class) rather than a vote, with the natural-rate memory | motivated by +0.018 AUC; a prompt-level change in Group C ports (not mine); GPU arm `*_mem_on/off` tests the end-to-end value |
| FinFRE: keep a 1:1 pool only if the GPU arm shows a Brier effect; offline the pool rate does not change the AUC | decided by arms `finfre_*_pool_*` and `*_zero_shot` |
| Evidence: prefer `amount_1hop` when `scorer_pcst` is not better on the GPU arm (23% fewer tokens, same connectivity) | tier-2 of the selection rule decides it |

## 4. GPU matrix: size and cost

34 arms (+ the combined arm) in `scripts/t3_rag_bench.sh`; table above. Estimated GPU-hours: 9.3 h for the 33 arms runnable today, 9.5 h for all 34 once the library is rebuilt (the earlier 8.2 h excluded the three blocked groups and assumed prompt factors 0.6 / 0.9 / 1.5 for ev_none / rules_none / rules_whole; the measured factors in 4a are 1.01 / 0.99 / 1.01 because the prompt cap binds); plus 1.4 h for the library teacher runs (`library` stage), 0.4 h for the combined arm (2 seeds) and about 3 h for the once-only eval (160 alerts, default + winners of every group); total about 14 GPU-h plus one 12B model load (about 5 min). Planning basis: old-stack seconds per alert (`docs/t3_val_runs.md`) divided by 2.5 for `--concurrency 4`; the first measured arm replaces the factor.

### 4a. The three flags (added by worker FLAGS; the arms are no longer BLOCKED)

| flag | what it does | where |
|---|---|---|
| `--evidence-view none` | the alert summary only: `evidence_views.view(.., "none")` reads nothing and returns the one line `(no evidence trail given)` (`ftm.types.NO_EVIDENCE_TEXT`); the line goes into `summary["evidence"]`, so the planner prompt (section "Evidence", header kept, body that line), the adapted arms, Group D and Group C all print the same thing. Torres skips the placeholder as an evidence row | `ftm/retrieval/evidence_views.py`, `ftm/external/torres.py` |
| `--rules {none,fixed,alert,whole}` | none = no Rules section; fixed = `FIXED_RULES_QUERY` (the old RULES_QUERY) top 5 for every alert; alert = `rules_query(alert)` top 5 (default, unchanged); whole = every rule and red flag of the home jurisdiction (`Gate.whole_corpus`, citable rules first, harness `all_rules()`), then the SHARED trimmer cuts to the 6,000-token prompt cap. Planner and the adapted Group D arms: the pipeline's prepare node (`GraphInvestigator(rules_mode=)`); Group D loops, Torres, Co-Investigator call `lookup_rules(<own query>)` themselves, so `RulesModeToolBox` applies the same four modes to that call (fixed replaces their query by the fixed sentence) | `ftm/compliance/rules_mode.py`, `ftm/graph/pipeline.py`, `ftm/planners/llm.py`, `ftm/graph/agents/registry.py` |
| `--no-exemplars` | FinFRE-RAG zero-shot: the same system prompt and the same current-case sentence (same features, same partner lines in ported mode), but no retrieval and no exemplar block; the instruction line becomes "Assess the following case." (our wording: the repo holds no zero-shot prompt text from the paper). Native and ported (graph and legacy class, parity tests unchanged) | `ftm/external/finfre_rag.py` (`use_exemplars`) |

summary.json records `evidence_view`, `edge_scorer`, `rules_mode` and `exemplars` (`retrieved` / `none` / null for non-FinFRE systems); the planner block also records `mean_rules_offered`, `mean_rules_in_prompt`, `mean_questions_offered`, `mean_questions_in_prompt` (traces.jsonl is unchanged: the graph/legacy parity test compares traces byte for byte). `t3_rag_bench_report.py` checks the recorded settings against each arm's flags (`settings_mismatch`) and excludes an arm that disagrees.

Trimmer: order unchanged (oldest evidence rows, strategies, menu questions); new last step = cut the rules beyond the default top 5 from the tail. So `--rules whole` pays for its extra documents with evidence rows and menu questions, not with other rules, and the default and `fixed`/`alert` (5 documents) never reach the new step.

**Prompt sizes measured through the pipeline** (3 val fit alerts V000438, V000690, V000721, `--profile tuned`, mock LLM, harness world, `--no-cannot-cross`, first planner prompt, estimator tokens; cap 6,000):

| arm | prompt tokens (3 alerts) | rules in prompt / offered | menu questions in prompt (of 51) | evidence rows kept |
|---|---|---|---|---|
| default (`--rules alert`) | 5,862 / 5,923 / 5,861 | 5 / 5 | 20 | 1 |
| `--evidence-view none` | 5,971 / 5,880 / 5,969 | 5 / 5 | 20-21 | 0 (placeholder line) |
| `--rules none` | 5,839 / 5,853 / 5,836 | 0 / 0 | 22 | 1 |
| `--rules fixed` | 5,920 / 5,935 / 5,919 | 5 / 5 | 20 | 1 |
| `--rules whole` | 5,937 / 5,939 / 5,929 | 31 / 31 (all 31 fit) | 12 | 1 |
| default, cap lifted to 20,000 (untrimmed) | 10,327 / 9,745 / 10,420 | 5 / 5 | all 51 | 3,703 / 1,630 / 4,047 chars |
| `--rules whole`, cap 20,000 | 11,583 / 10,954 / 11,676 | 31 / 31 | all 51 | same |

**Superseded (2 Nov, worker PROMPTFIT): `docs/t3_prompt_budget.md` fixes this under the tuned profile (cap 6,750, menu top-8 by expected Brier reduction, whole trail protected); the numbers in this table are the state BEFORE that change, the new token breakdown is in that note and the arms `menu_topk_*` / `evidence_min_*` test it.**

**Reading, important for every arm of the evidence and rules groups:** the compose prompt (alert, 51-item back-tested menu, question-language section) is about 10,000 tokens before trimming and the cap is 6,000, so in production the trimmer keeps ONE evidence row (the alerted payment, "older evidence rows omitted") and 20 of 51 menu questions. The prompt sizes of all arms are therefore within 1% of each other (the cap binds), the evidence views differ only in that one row plus the omitted-rows notice, and `ev_none` removes the last row. Two consequences: (1) the evidence arms are expected to be close to each other by construction at cap 6,000, which is a property of the prompt budget, not of the views (the offline recall numbers in 3.2(a) concern the untrimmed trail); (2) `rules_whole` fits all 31 documents only by cutting the menu from 20 to 12 questions, so its effect is the net of "more rules" and "fewer menu questions". The GPU-hour planning factors are set from these ratios (1.01, 1.01, 0.99, i.e. none of the three changes the time per alert measurably).

**Offline finding to carry into the `rules` group (unchanged, from 3.2(b)):** the alert-aware query LOWERS the red-flag share of the right typology in the top 5 (0.504 fixed -> 0.307 alert-aware under BM25; keyword 0.504 -> 0.588; hybrid 0.504 -> 0.060) and returns only 7 distinct top-5 sets. `alert` stays the default until the GPU arm `rules_fixed` decides (selection rule, section 1); `rules_fixed` is exactly the arm that tests whether the fixed sentence is better.

## 5. How to run

```
scripts/t3_rag_bench.sh plan          # arm table, no GPU
scripts/t3_rag_bench.sh arms          # all ready arms on val fit + cal (resumable), then the report
scripts/t3_rag_bench.sh combined      # after reading outputs/rag_bench/selection.json
scripts/t3_rag_bench.sh eval          # once
python scripts/rag_offline_bench.py   # the CPU bench (about 30 min, contexts cached in outputs/rag_bench/contexts.pkl)
```



## 11. Library fix made before any GPU run of this bench (7 Nov 2026)

**What was wrong.** The fit and cal arms of this bench pointed at `outputs/t3_library/frozen/shape_gemma.json` (default) and `frozen_v2/account_teacher_only.json`. The first was built from the 1,983 train teacher runs **plus the val tuning runs** (`docs/t3_library_draft.md`: 3,091 records), so val fit/cal alert outcomes could shape the notes a fit/cal arm read: a label leak into the very alerts the bench chooses on, and the opposite of the rule `scripts/t3_rerun.sh` follows (teacher-only libraries on fit and cal, the full one on eval). Found while preparing the bench for ada; nothing had run yet, so no result is affected. The offline-bench numbers in sections 2-3 (`outputs/rag_bench/offline*.json`) are descriptive and unchanged: that bench compares the old libraries on val on purpose, and its `shape_teacher_only` arm is the train-only comparison.

**What changed.**
* The part decides the library, in one place (`scripts/t3_rag_bench_report.py`, `lib_path(kind, part)`): fit and cal use `outputs/t3_library/frozen_v3/shape_gemma_train.json` / `account_gemma_train.json` (teacher_v2 runs only); eval uses `frozen_v3/shape_gemma.json` / `account_gemma.json`. The shell script reads both paths from the report (`arms --tsv` now has `lib_fitcal` and `lib_eval` columns) and never names a library file itself. `lib_shape_bt` keeps its file for both parts: it is built from TRAIN back-test rows only (`build_backtest_library(contexts["train"])` in `scripts/rag_offline_bench.py`; no LLM, no val row).
* The stage `library` and the arm `lib_shape_rebuilt` are gone (they would have redone ~1.4 GPU-h of teacher runs that teacher_v2 already covers on the same tuned stack, and `frozen_v3` is built from those). The arm list has 33 arms.
* Tests (`tests/test_rag_offline_bench.py`): no fit/cal arm may point at anything but the three train-only files, eval gets the full file and never the train one, the part must be named, and no script or config on the val path may name a library built with val runs (allow-list with the reason for each remaining hit: T4's `configs/final/jobs.yaml` is the TEST final-run plan and keeps the full library; `scripts/rag_offline_bench.py` is the descriptive comparison; `scripts/t3_qwen_session.sh` is E16 on val eval only; `scripts/t3_val_session.sh` is the old session that built the libraries).

The arms, the selection rule and every threshold are exactly as written in sections 1-2.

## 12. Recorded override: `menu_topk_all` is invalid (8 Nov 2026)

The pre-written selection rule (section 1-2) picked `menu_topk_all` as the menu winner (tier 2, "non-inferior and cheaper": Brier difference -0.0021 [-0.0065, +0.0025], 23 % fewer prompt tokens). The arm failed the call-share validity check of the run (29 % of the alerts made an LLM call on fit/c001, minimum 30 %): the full 204-question menu overflows the prompt, most alerts return p0 with no LLM call (zero-call alerts 60/100 and 63/89 on fit, against 22/100 and 17/89 for `menu_topk_8`; PromptTooLong 0.79 per alert), so its near-p0 Brier and its "saving" are artefacts, not the full menu doing well. By the user's decision the arm is excluded from the selection as **invalid: failed the call-share check; most alerts return p0 because the full menu overflows the prompt**. The exclusion is a recorded override (`--overrides-file`, written to `selection.json` under `overrides` and printed in `report.md`), not an edit of the rule: the arm stays in every table with its numbers, with `excluded` and tier 0 in its candidate row. The menu component therefore keeps the default (`menu_topk_8`, the profile's own setting); no other component changed.
