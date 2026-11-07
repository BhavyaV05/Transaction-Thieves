# T3 S1BENCH: System 1 on the public System One benchmarks and against open System One models (2 Nov 2026)

Owner: T3 (Akshat).  Worker S1BENCH.  Everything in sections 1-6 (design, scorer rules, selection rule, collapse checks) was written
BEFORE any model was run on any of these datasets; the results sections (7 on) are appended afterwards and never change sections 1-6.
Companion docs: `docs/t3_system1.md` (the System 1 design this evaluates), `docs/t3_e10.md` (first version).

System 1 (C13) = Qwen3-4B-AWQ + LoRA (`fast-s*` adapters, `fast3-*` after the System 1 upgrade) answering one-token typed decisions through
`ftm/fastmodel/typed.py` (`TypedDecider.decide`: the full distribution over the enumerated labels from ONE forward pass, two-stage above 26
options), calibrated by temperature, wrapped in a Mondrian conformal cascade to the large tier (Gemma 4 12B, same single-token read).
It is "System-One-style" (TypeSafe's Jev).  Question asked here: how does it do on the public System One benchmarks, and against the open
System One models, on OUR decisions?

## 1. Scope and hard limits

* CPU only for this worker.  Open encoders (bekko, laya, GLiNER2.5-Decide) are run on CPU now.  Anything that needs the GPU (System 1
  and base Qwen3-4B via vLLM, Gemma 4 12B, JevK5, Tev1-4B, Open-Jev-9B, any fine-tune that does not fit on CPU) is an exact command
  in section 9; no number for it exists in this file until the GPU queue has run it.
* No test split of OUR data is read (`decisions_val.jsonl` only; the train rows are used only for the optional fine-tune).  The public
  benchmarks are scored on THEIR own eval split (their test split, or validation for SST-2 whose test labels are hidden); that is not
  our test split.
* S1MB is a test-only benchmark that forbids tuning on its test labels ("fix evaluation settings before running; do not tune parameters or
  select configurations using test labels or measured scores").  So nothing about prompts, temperature or arms is chosen on S1MB scores;
  every setting is fixed in `configs/s1bench/*.yaml` before the run.
* Open-model code and weights are never vendored or committed; they live under `/home/akshat/projects/s1bench_data` (outside the repo) with
  their pinned revisions recorded in `configs/s1bench/models.yaml`.  A separate venv (`venv-open`, torch 2.10 CPU, transformers 5.17) is used
  for the open models because their cards pin those versions; the main venv (torch 2.13) is untouched.

## 2. Part A: external validity

### 2.1 Evaluators (all pinned in `configs/s1bench/datasets.yaml`)

| id | dataset (revision) | eval split | licence (as stated by the source) | decision | metrics |
|---|---|---|---|---|---|
| s1mb | `hotchpotch/S1MB-english-v1` (pinned SHA) | test (the benchmark is test-only) | "other" (mixed-source): each subset keeps its original licence (README table: mostly CC BY / Apache / MIT; some GPL-3.0 `ethos`, non-commercial NanoBEIR subsets, `Unknown` for ag_news / enron_spam / scicite); evaluation use only, no redistribution of the data | the benchmark's own Choice / Noul / Score questions | official S1MB metrics (2.2) |
| banking77 | `PolyAI/banking77` | test (3,080) | CC BY 4.0 | 77-way choice, two-stage | accuracy, macro-F1 |
| clinc150_oos | `clinc/clinc_oos` config `plus` | test (5,500: 4,500 in scope, 1,000 OOS) | CC BY 3.0 | 150 intents + `out_of_scope` = 151-way | in-scope accuracy, OOS recall, OOS precision, accuracy |
| hwu64 | `FastFit/hwu_64` | test (1,076) | CC BY 4.0 (original NLU-Evaluation-Data; the mirror states none) | 64-way choice | accuracy, macro-F1 |
| toxicchat | `lmsys/toxic-chat` `toxicchat0124` | test (5,083) | CC BY-NC 4.0 (research use only) | yes/no "is the user message toxic" | F1 (toxic class), AUROC |
| unfair_tos | `coastalcph/lex_glue` `unfair_tos` | test (1,607) | CC BY 4.0 | 8 yes/no decisions per clause (one per unfair-term type; multi-label) | micro-F1, macro-F1 (LexGLUE's), per-type F1 |
| sst2 | `stanfordnlp/sst2` | validation (872) | none stated upstream (GLUE) | negative / positive | accuracy |

Our own held-out injection families (docs/t3_system1.md, `generate_diverse` val set) are scored by the System 1 integration arms, not here.

### 2.2 S1MB scorer (official definitions, reproduced and validated)

The official evaluator (`hotchpotch/S1MB`, MIT, pinned commit in `models.yaml`) defines, in `evaluator/SCORING.md`: **Noul** primary Brier,
reported balanced accuracy (BA) with soft targets counted fractionally, adjusted skill `2 BA - 1`; **Choice** primary target mass at the argmax
(ties by original order), baseline = best of uniform / fixed option id / fixed position, adjusted skill `(acc - b)/(1 - b)`; **Score**
primary MAE of the normalised expected score, baseline = constant median, adjusted skill `1 - MAE/MAE_base`; each benchmark's adjusted skill is clipped to [0, 1]; the
task score is the equal-weight mean over eligible benchmarks; **Task Avg** is the equal-weight mean of the three task scores.  Borda (a roster-relative
rank) is not reproduced for our rows because it needs the whole roster; it is reported only from the official summary for models in it.
`ftm/fastmodel/s1bench_s1mb.py` re-implements these formulas with numpy only (the official package needs pydantic/datasets and pins) and is
validated three ways (section 7): (a) unit fixtures; (b) against the official `s1mb.metrics.calculate` on identical predictions; (c) by
recomputing a published row of `hotchpotch/s1mb-result` from its stored predictions and comparing to the stored metrics.  Our scorer is trusted
only if (c) matches to 1e-9 on every benchmark of that row.

Rendering of an S1MB question as a typed decision for System 1 / base Qwen / Gemma (fixed now, not tuned): the existing `messages()` template
(system prompt and "Question:" layout of `ftm/fastmodel/decisions.py`, unchanged: it is the format the adapters were trained on); the user
turn is the state (JSON as given), then "Question: <instructions>", then the option list; Noul gets the authored false/true definitions as the two options (letters A=false,
B=true... as the template's `A = <definition>` lines); Choice gets the option descriptions in the stored order with letters; Score gets the levels
in stored order with their numeric values in the description; expected score = sum of value x probability from the returned distribution.  More than 26 options: the
two-stage scorer of `typed.py` (stage 1 lists the whole group, see 2.4).  Inputs whose rendered prompt exceeds the serving
context (2,048 tokens for the fast profile) are NOT truncated: they are counted as unscored, the benchmark is marked incomplete and excluded
from "covered Task Avg" for every model compared (the published models are re-aggregated on the same covered set from their stored predictions).

### 2.3 Metrics for every task (identical code for all external evaluators)

accuracy / F1 as the task defines; multiclass Brier (sum over classes of squared error, mean over items); log loss; ECE (top-label, equal-mass
bins, 15 bins, `ftm/fastmodel/calibration.top_label_ece` style but equal-mass); **option-order bias**: the label order shown to the model is permuted
with 3 fixed random permutations (seeds 11, 12, 13; the canonical order is permutation 0); reported = accuracy under each permutation, the range
(max - min) and the mean total-variation distance between the canonical distribution and the permuted one realigned to the canonical labels; for
tasks with more than 26 options the permutation arms use a fixed 1,000-item subsample (seed 0) to bound the cost, and unfair_tos (8 decisions per clause) a fixed 400-clause subsample; a yes/no task is produced in the production word format ("yes or no") and its order arms use the lettered format (A = yes, B = no, and the swap), so the reference for its order bias is the lettered identity arm; **latency** per decision
(median and p90 wall seconds of the decision, including all passes of a two-stage decision, one request at a time).
Calibration: raw (T = 1) and temperature-scaled with ONE scalar T per task fitted on the benchmark's own validation split (banking77, toxic-chat, sst2: a
500-item subsample of the train split, seed 0; unfair_tos / hwu64 / clinc: their validation split, capped at 500 items per decision).  The
test split never enters the fit.

### 2.4 Models for Part A

`system1` (the `fast3-s0` adapter when it exists on the server, else `fast-s2`; the adapter name is part of every output file), `qwen3-4b` (the
untuned base, same server profile, reference) and `gemma4-12b` (large tier).  All three through `TypedDecider.decide`; no free generation.
Because `typed.py`'s two-stage stage 1 shows each group only as "A = labels first .. last" (it cannot see the labels in between), the
production mechanism is scored as `two_stage_prod` and, as a benchmark-only variant that lists every label of every group in stage 1,
`two_stage_listed`; the difference is reported, not hidden.  The published numbers and our numbers are reported side by side with the published models
from the official results file in section 8 (Jev 1.13 and the open models), never mixed with different row sets.

## 3. Part B: head-to-head on OUR decisions

Data: `data/decisions/hi_medium_s1v1/decisions_val.jsonl`, val parts `fit` / `cal` / `eval` by the alert's `val_part`.  Decisions: `worth_asking`,
`typology`, `rule_applies`, `injection_present` (the four learned decisions; `stop` is an exact function of the state since wave 4 and is not a
learned decision any more, so there is nothing for a model to compete on there; this is the "five decisions" of E10 minus the one that was
removed on purpose).  Train rows (`decisions_train.jsonl`) are used only for the optional fine-tune of an open model.

Open models are used zero-shot through THEIR OWN typed interfaces, with the same authored decision definitions for all of them
(`configs/s1bench/decisions.yaml`: the instruction is the decision's question; the option descriptions are the label meanings: for `typology`
the six pattern definitions of `TYPOLOGY_HINT`, for the yes/no decisions a one-sentence meaning of yes and of no; written once, before any run, and shared by bekko,
laya, GLiNER2.5-Decide, JevK5, Tev1 and Open-Jev).  No per-model prompt hacking: each adapter only translates the shared typed question
(state text, instruction, options with descriptions) into the model's documented input.  Candidates:

| candidate | interface | where |
|---|---|---|
| system1 (`fast3-s0` or `fast-s2`) | `TypedDecider.decide` + frozen temperature | GPU |
| qwen3-4b base | same | GPU |
| bekko-system-one-v0-400m | `BekkoSentenceTransformer.predict` (choice / noul) | CPU |
| laya, laya-typed-decisions | `laya` Router / `laya.load(...).predict` | CPU |
| GLiNER2.5-Decide | `AutoExtractor.classify_text` with full-softmax confidences | CPU |
| JevK5 (4B) | `jevk5.runtime.JevK5.probabilities` (native letter-logit readout) | GPU |
| Tev1-4B-experimental | native decision JSON, A-X letter probabilities | GPU |
| Open-Jev-9B | blocked unless a build <= 15 GB loads (section 9) | GPU |
| best small open model, fine-tuned on our train decisions | its own training recipe | CPU if feasible else GPU |

## 4. Selection rule (fixed before any run; fit and cal only)

Question: which candidate is our System 1.  Reads `fit` and `cal` rows only (pooled; `eval` is scored for every candidate exactly once afterwards and
plays no part in the choice).

1. **Eligibility (collapse checks), per candidate and decision, on fit+cal pooled**: a candidate is *collapsed* on a decision if (i) it predicts one
   single label for every row (constant prediction), or (ii) its argmax accuracy does not exceed the accuracy of always predicting the pooled-fit majority label by more than
   0.01 AND its multiclass Brier is not below the Brier of the fit label prior (majority-only / no skill), or (iii) the share of rows on which the
   predicted label is the majority label is >= 0.99 when the majority share of the truth is < 0.9.  A candidate collapsed on any decision is *ineligible* as the System 1 and is reported as such;
   (the same idea as the cascade's retirement rule of `docs/t3_system1.md` section 3).  A decision on which EVERY candidate is collapsed has no candidate to
   select: it is excluded from eligibility and from the macro criterion below, listed as "no candidate has skill", and the cascade retires the fast tier for it.
2. **Primary criterion: macro Brier.**  For each eligible candidate: the multiclass Brier of its calibrated probabilities (the candidate's own output for
   the open models; its own output after its frozen temperature for System 1), averaged with equal weight over the decisions that survive step 1.  Candidates are ordered by it.  Let S be the lowest.
   Paired cluster bootstrap (`ftm/eval/bootstrap.py`-style, clusters = alert_id, 2,000 resamples, seed 0, 95% percentile CI) of
   `Brier(c) - Brier(S)`: candidate c is *tied with S on Brier* iff the CI contains 0.  The tied set T always contains S.
3. **Within T: lowest macro ECE** (equal-mass, 10 bins).  Candidates whose ECE is within 0.005 of the lowest are tied again.
4. **Then lowest latency per decision** (median seconds, one request at a time; the hardware behind each number is stated in the table, and exact ties only).
5. **Ties -> the smaller model** (fewer parameters).
The selected candidate is written to `outputs/s1bench/selection.json` by `scripts/s1bench_partb.py select` from fit+cal files only, and the eval report is built from that file.
Headline contrast on eval (reported once, per decision, Holm within the family of {chosen vs each other candidate} x {4 decisions}): paired cluster-bootstrap CIs
of the Brier difference, plus accuracy vs the fit majority, ECE, log loss, latency and (for System 1) the cascade kept-fast share.
The fine-tuned open model competes in the same rule; it is created only after the zero-shot fit+cal table exists and its training data is the train split only.

## 5. Option-order and consistency

Option-order bias in Part A as in 2.3.  For Part B, the learned decisions with more than two labels (`typology`) are also scored under the 3 permutations; the
yes/no decisions swap their two options.  The reported numbers are accuracy range and mean TV.

## 6. Output files (not committed)

`outputs/s1bench/<part>/<model>/<task>.jsonl` (one line per item: id, label, per-permutation `probs`, variants, seconds), `.../summary.json`; `models.yaml` / `datasets.yaml`
hold the pins; the report script (`scripts/s1bench_report.py`) regenerates the tables of section 8 from the outputs.

SECTIONS 7 ONWARD (scorer validation, results, GPU commands) FOLLOW BELOW; the text above is frozen.

---
# Results and status (appended after the runs; sections 1-6 above are unchanged in substance)

Two clarifications written before the first Part B run (no result existed): section 4.1 drops a decision on which every candidate is collapsed; the SST-2 / ToxicChat / banking77
calibration split is a 500-item train subsample (sst2's validation is its eval split).

## 7. Scorer validation (done, CPU)

`scripts/s1bench_validate_scorer.py` recomputes every stored metric of a published result file from its stored predictions with our numpy scorer
(`ftm/fastmodel/s1bench_s1mb.py`) and compares to the metrics the official evaluator wrote.  Data: `hotchpotch/S1MB-english-v1` @ `10bcb8c6916b` (prepared:
137 benchmarks, 26,269 judgments, exactly the official active membership of `training-manifest.json`), results @ `426b8308eded`.

| published row (hotchpotch/s1mb-result) | benchmarks compared | match to 1e-9 on every stored numeric metric |
|---|---|---|
| typesafe/jev_1_13 | 125 (it has no result file for 12 of the 137 active benchmarks) | 125 / 125 |
| bekko-system-one-v0-400m, laya, laya-typed-decisions, GLiNER2.5-Decide, JevK5, Tev1-4B-experimental, Open-Jev-9B | 137 each | 137 / 137 each |

So the scorer reproduces 1,122 published benchmark results (8 models) exactly, including baselines, adjusted skills and balanced accuracy with soft targets.
It is trusted for our own runs.  Abstentions (the official max-error penalty) are not implemented; no model of ours abstains (a server failure is an unscored judgment and
makes the benchmark incomplete, as in the official rules).

## 8. Faithfulness table: published numbers re-aggregated by our scorer (S1MB English v1, Task Avg 0-1)

Task score = mean baseline-adjusted score (clipped to [0, 1]) over the eligible benchmarks of the task; Task Avg = mean of Choice, Noul, Score.  "Full" uses every complete benchmark
of the row (Jev 1.13 has 125 of 137, so its Task Avg is over those 125 and is not a full-roster number); "covered" restricts all rows to the 125 benchmarks Jev 1.13 completed.
Published rows were measured by their authors with the official evaluator; Jev is TypeSafe's hosted model, measured by the S1MB maintainer.

| model | Choice | Noul | Score | Task Avg (covered 125) | Task Avg (full) |
|---|---|---|---|---|---|
| Jev 1.13 (TypeSafe) | 0.682 | 0.661 | 0.501 | 0.615 | 0.615 (125 benchmarks) |
| Open-Jev-9B | 0.572 | 0.580 | 0.490 | 0.547 | 0.521 |
| bekko-system-one-v0-400m | 0.615 | 0.525 | 0.417 | 0.519 | 0.506 |
| Tev1-4B-experimental | 0.550 | 0.521 | 0.385 | 0.485 | 0.471 |
| JevK5 (4B) | 0.545 | 0.502 | 0.226 | 0.424 | 0.411 |
| laya-typed-decisions | 0.204 | 0.205 | 0.070 | 0.159 | 0.150 |
| laya | 0.175 | 0.202 | 0.036 | 0.138 | 0.134 |
| GLiNER2.5-Decide | 0.196 | 0.059 | 0.045 | 0.100 | 0.092 |
| **our System 1 / base Qwen3-4B / Gemma 4 12B** | GPU pending (section 10) | | | | |

Caveats that matter: the bekko card states its training data includes dataset families in the evaluation (so its Choice number is not an unseen-task result); laya and GLiNER2.5 are
reported by their cards to be weak zero-shot on typed-decisions style questions, and the table agrees.  Our System 1 was fine-tuned only on AML decisions; its S1MB numbers will measure how much of
the typed-decision format transfers, not a trained-for-S1MB score.

---
# Results (7 Nov 2026; sections 1-8 above are unchanged)

All numbers below are produced by `scripts/s1bench_report.py` and `scripts/s1bench_partb.py select|eval` from the outputs of the ada runs (`docs/t3_ada.md`; RTX 2080 Ti, vLLM, fp16, Qwen3-4B-AWQ + LoRA adapters; S1MB / public tasks scored with the validated scorer of section 7) and from the earlier CPU runs of the open models. Where something has not been run it says PENDING.

## 9. Part A: our System 1 on the public benchmarks (qwen3-4b base, `system1` = fast3-s0, `system1_soft` = fast3soft-s0)

The adapters were fine-tuned only on AML decisions; the numbers measure transfer of the typed-decision format, not a trained-for-the-benchmark score. Prompts that exceed the 2,048-token serving context are not truncated: 1,198 judgments are unscored for every model alike, 12 of the 137 S1MB benchmarks are therefore incomplete, and the S1MB comparison is on the 125 complete ones (same covered set as section 8).

| S1MB English v1 (covered 125 of 137) | Choice | Noul | Score | Task Avg |
|---|---|---|---|---|
| Jev 1.13 (published, section 8) | 0.682 | 0.661 | 0.501 | 0.615 |
| bekko-system-one-v0-400m (published) | 0.615 | 0.525 | 0.417 | 0.519 |
| qwen3-4b base (this work) | 0.287 | 0.211 | 0.068 | **0.189** |
| system1 (fast3-s0) | 0.272 | 0.191 | 0.019 | **0.160** |
| system1_soft (fast3soft-s0) | 0.285 | 0.176 | 0.017 | **0.159** |

Public tasks (test split, raw probabilities; `Brier (T)` / `ECE (T)` are after one temperature fitted on the task's own calibration split; order-bias columns exist only for tasks with at most 26 options; full table `outputs/s1bench/report_ada/ext.md/report.md`):

| task | model | accuracy | macro-F1 | Brier | ECE | ECE (T) | median ms |
|---|---|---|---|---|---|---|---|
| banking77 (77-way) | qwen3-4b | 0.295 | 0.238 | 1.241 | 0.577 | 0.054 | 417 |
| | system1 | 0.386 | 0.409 | 0.831 | 0.218 | 0.148 | 581 |
| | system1_soft | 0.526 | 0.511 | 0.810 | 0.361 | 0.200 | 617 |
| clinc150_oos (151-way) | qwen3-4b | 0.138 | 0.117 | 1.467 | 0.678 | 0.061 | 740 |
| | system1 | 0.246 | 0.263 | 0.892 | 0.120 | 0.040 | 1079 |
| | system1_soft | 0.218 | 0.188 | 0.898 | 0.115 | 0.073 | 1056 |
| hwu64 (64-way) | qwen3-4b | 0.270 | 0.244 | 1.194 | 0.526 | 0.087 | 411 |
| | system1 | 0.325 | 0.349 | 0.867 | 0.207 | 0.153 | 627 |
| | system1_soft | 0.456 | 0.496 | 0.857 | 0.334 | 0.071 | 570 |
| sst2 | qwen3-4b | 0.905 | 0.905 | 0.180 | 0.085 | 0.045 | 126 |
| | system1 | 0.904 | 0.903 | 0.175 | 0.105 | 0.031 | 138 |
| | system1_soft | 0.885 | 0.885 | 0.202 | 0.136 | 0.030 | 145 |
| toxicchat (toxic F1 / AUROC) | qwen3-4b | 0.924 (F1 0.093, AUROC 0.438) | 0.527 | 0.141 | 0.071 | 0.082 | 125 |
| | system1 | 0.951 (F1 0.570, AUROC 0.914) | 0.772 | 0.083 | 0.026 | 0.009 | 140 |
| | system1_soft | 0.949 (F1 0.549, AUROC 0.906) | 0.761 | 0.088 | 0.051 | 0.014 | 142 |
| unfair_tos (8 yes/no per clause; micro-F1 / macro-F1) | qwen3-4b | micro 0.275, macro 0.351 | | 0.112 | 0.048 | 0.008 | 122 |
| | system1 | micro 0.076, macro 0.081 | | 0.520 | 0.199 | 0.025 | 153 |
| | system1_soft | micro 0.068, macro 0.075 | | 0.446 | 0.023 | 0.032 | 143 |

Reading: the fast adapters keep what they were trained for and transfer where the task looks like one of ours (toxicchat: AUROC 0.438 to 0.914; many-class intents: accuracy up and Brier down by about 0.4 to 0.6, with the 150-way tasks still far from usable), do not move on sst2, and lose on unfair_tos, where the base model's "no" prior is right and the fine-tuned yes/no heads over-call (micro-F1 0.076 against 0.275). On S1MB both are below the base model (Task Avg 0.160 and 0.159 against 0.189) and far below the published open System One models; the claim "System-One-style" refers to the mechanism (one forward pass, typed distribution, calibration, cascade), not to S1MB competence. Latencies are 2080 Ti vLLM wall times with 32 workers for the throughput runs and one request at a time for `ms`. The `system1_listed` variant (two-stage with every label listed in stage 1) was run on banking77, clinc150_oos and hwu64 (`outputs/s1bench/ext/system1_listed`); its comparison with the production two-stage mechanism is PENDING (not tabulated here). **Gemma 4 12B on Part A (`s1_gemma_partA`, needs two GPUs) has not been run: PENDING.**

## 10. Part B: head-to-head on our decisions (selection rule of section 4, run once)

Candidates scored on all rows: system1 `fast3-s0` (`system1_tuned_s0`), `fast3soft-s0`, untuned Qwen3-4B, JevK5, Tev1-4B (GPU); on the 500-per-decision subsample of the CPU runs (`--cap 500`, the same subsample for everyone in the selection): bekko-system-one-v0-400m, GLiNER2.5-Decide, laya-typed-decisions. **Not in the table: `laya` base (1,600 rows, incomplete) and the fine-tuned GLiNER (100 rows, the fine-tune did not finish): PENDING; Open-Jev-9B was never loaded (section 3: needs a build under 15 GB).**

Selection on `fit` + `cal` only (300 alert clusters, 2,000 resamples), `outputs/s1bench/selection.json`: macro Brier over the decisions that have a candidate with skill: **system1_tuned_s0 0.1384 (selected)**, system1_soft_s0 0.1765, untuned Qwen3-4B 0.3279, Tev1-4B 0.3762, JevK5 0.4120; bekko, GLiNER2.5-Decide and laya-typed-decisions are ineligible (collapsed on at least one decision). `typology` is excluded from the macro criterion: no candidate beats the fit-majority baseline there (accuracy 0.679 for the majority label and for system1). Brier differences to the best are significant for every other candidate (paired cluster bootstrap, all CI above 0). The tie steps (ECE, latency) did not matter: the best candidate was not tied on Brier.

Eval, scored once (`outputs/s1bench/partb_eval_once/`), 158-159 alert clusters, Brier of each candidate minus Brier of the selected system1 (positive = system1 better), 95 % paired-bootstrap CI, `*` = Holm-significant within the family (candidates x 4 decisions):

| candidate | worth_asking | typology | rule_applies | injection_present |
|---|---|---|---|---|
| system1_soft_s0 | +0.019 [+0.006, +0.037]* | +0.007 [-0.010, +0.022] | +0.045 [+0.024, +0.063]* | +0.064 [+0.037, +0.092]* |
| qwen3-4b_untuned | +0.176 [+0.145, +0.209]* | -0.083 [-0.105, -0.063]* | +0.225 [+0.187, +0.261]* | +0.188 [+0.151, +0.227]* |
| bekko-400m | +0.430 [+0.399, +0.458]* | +0.321 [+0.273, +0.366]* | +0.115 [+0.050, +0.170]* | +0.276 [+0.220, +0.334]* |
| gliner25-decide | +0.428 [+0.364, +0.489]* | +0.334 [+0.284, +0.383]* | +0.412 [+0.368, +0.450]* | +0.272 [+0.228, +0.315]* |
| laya-typed-decisions | +0.496 [+0.463, +0.522]* | +0.293 [+0.246, +0.338]* | +0.296 [+0.261, +0.327]* | +0.328 [+0.289, +0.368]* |
| jevk5 | +0.342 [+0.304, +0.377]* | +0.371 [+0.316, +0.424]* | +0.254 [+0.196, +0.307]* | +0.247 [+0.196, +0.302]* |
| tev1-4b | +0.231 [+0.207, +0.249]* | +0.020 [-0.046, +0.085] | +0.312 [+0.249, +0.370]* | +0.198 [+0.156, +0.242]* |

Eval Brier of the selected system1: worth_asking 0.056 (accuracy 0.960 against the fit majority 0.728), typology 0.659 (0.679 against 0.679), rule_applies 0.215 (0.856 against 0.500), injection_present 0.116 (0.928 against 0.474). Reading: on three of the four decisions the System 1 adapters beat every open System One model by a wide margin, as expected for models trained on these decisions against models used zero-shot; the open models' typology Brier (0.95-1.03 for bekko, GLiNER, laya, JevK5) is worse than guessing the majority label. On `typology` nothing beats the majority baseline and the untuned Qwen3-4B is significantly better than the tuned System 1 on Brier (-0.083): the cascade should keep retiring the fast tier for that decision. `system1_soft_s0` (calibration-as-objective) is worse than the selected hard-label adapter on the three decisions with skill. Caveats: the System 1 latencies in the selection come from the PC's 5070 Ti (vLLM, one request at a time) and the open models' from a CPU or the 2080 Ti, so the latency step is across hardware; the open models were run on a 500-row-per-decision subsample while system1 was scored on that same subsample for the selection and the eval contrast (`--cap 500`).
