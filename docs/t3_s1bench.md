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
