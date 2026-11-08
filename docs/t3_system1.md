# T3 System 1: the fast decision model in the system graph (wave 4, 2 Nov 2026)

Owner: T3 (Akshat).  Replaces the E10 setup of `docs/t3_e10.md` (kept as the record of the first version).  CPU work is done and tested; every
GPU step is a command at the end.  Nothing here has been run on a GPU, so there are no result numbers in this file, only the design, the
rules written BEFORE any run, the dataset statistics and the commands.

System 1 is our open analogue of TypeSafe's System One decision models ("Jev"): a small model that answers typed decisions over an
enumerated option set, returns a calibrated distribution for every option, and is cheap enough to sit in front of the large model as a
"smart if-statement".  Section 7 says, property by property, what we reproduce.

## 1. What changed and why

| E10 finding (docs/t3_e10.md) | Wave-4 change |
|---|---|
| `worth_asking` labelled by the alert label ("is this a real attempt"): leaks the answer, drifted 57% -> 32% yes, model 0.71 vs 0.68 majority | label from the TUNED pipeline (section 2): expected Brier reduction of the best affordable routed question under ConditionalFusion `headroom_v1`; context = the planner's own state |
| `stop` learned (0.98 vs 0.77) although it is a deterministic function of the state | NOT learned: `ftm.fastmodel.decisions.exact_stop` = the pipeline's `cannot_cross` rule (`stop_cannot_cross`, tuned profile, bounds of `configs/decision/headroom_v1.yaml`). Equality with `ftm.decision.threshold.should_stop` is a test |
| `typology` 0.62 vs 0.68 majority, 8 imbalanced classes | 6 merged classes (`fan`, `hub`, `cycle`, `bipartite`, `chain`, `none`), train keeps 20% of the clean-alert rows, val natural; if it still cannot beat the fit-majority on cal the fast tier is RETIRED for it (rule below) |
| `injection_present` / `rule_applies` 100% because val text = train templates | disjoint template families: train = original wording + look-alike set A; val = paraphrases, ES/FR/DE/PT, base64/rot13/hex/leet/spaced encodings, look-alike set B, new plain notes (`ftm/safety/injection.py generate_diverse`); val rules shown in paraphrased wording |
| large tier degenerate (Gemma's stated confidence, q 0.95-0.999999) | large tier read by TOKEN LOGPROBS (`logprobs`/`top_logprobs`, one token, same prompt), own temperature fitted on `fit`, own conformal set on `cal` |
| stored `fast.jsonl` kept one logprob per label | score files keep every token variant per label (`variants`), the logsumexp is recomputed offline (`logps_of`) |
| System 1 not in the planner (H5 detection-loss clause untestable) | node `system1` in the graph (`ftm/graph/system1.py`, `GraphInvestigator.set_system1`), switch `run_llm_agent.py --system1 <frozen json>`; every decision logged |

## 2. Labels and contexts (`ftm/orchestrator/decision_set.py`, `ftm/fastmodel/context.py`)

Contexts are built by ONE set of functions used by the dataset builder and the pipeline node, from PlannerView ingredients only (alert
summary: transfer + compact shape; p0; fused posterior; threshold; eps left; answers so far with their family and LR; the routed menu).

* `worth_asking` (yes/no). For each state (0-3 answers so far, drawn from the simulated banks) take the pool's questions that are
  applicable to the alert direction, **right-bank routed** (`ftm.llm.routing.asked_bank` = the alert's partner bank), affordable
  (eps <= eps left), not yet asked.  For each, with the table's `p_cell = (P(cell|clean), P(cell|launder))` and the fused posterior p' after
  that cell (`ConditionalFusion`: `logit_base(p0) + shrink * sum_f w_f log LR`):
  `gain = p(1-p) - sum_cell [ p P(cell|launder) (1-p')^2 + (1-p) P(cell|clean) p'^2 ]`.
  Label yes iff the best gain >= `GAIN_MIN = 1e-3` (absolute Brier).  A fusion that overshoots a weak answer gets a NEGATIVE gain, so
  "yes" means asking is expected to improve the score, not just to move it.  The alert label is never read by this decision (a test flips it).
* `typology`: pattern-file typology merged by `TYPOLOGY_MERGE`; clean alert = `none`.  Context: `alert_text` (shape + p0).
* `rule_applies`: the Gate's rule logic (`rule_label`), unchanged; it is still exactly computable by the Gate in production, so the
  pipeline does not use the model's answer; it stays in the set as the held-out-wording robustness check.
* `injection_present`: the planted-note label of the generator.
* `stop`: exact, not in the set (section 1).

**Dataset built (CPU, `data/decisions/hi_medium_s1v1`, not committed; `scripts/make_system1_set.py`, p0 v2, headroom_v1 table + fusion, threshold 0.322699, seed 0).**
1,983 askable train alerts and 460 askable val alerts (T1/T2 world; the rest have no modelled partner).  54,896 rows (train 44,318 for SFT; val 10,578 = fit 4,392 / cal 2,507 / eval 3,679 by the alert's `val_part`).

| split / decision | rows | labels |
|---|---|---|
| train worth_asking | 15,864 | yes 10,450 (65.9%), no 5,414 |
| val worth_asking | 3,680 | yes 2,584 (70.2%); by part fit 69.8%, cal 71.2%, eval 70.0% (E10 label drifted 57 -> 43 -> 32%) |
| train typology (merged, none thinned to 20%) | 692 | none 332, chain 112, cycle 56, bipartite 65, fan 65, hub 62 |
| val typology (natural) | 458 | none 252 (55%), chain 57, hub 47, fan 51, cycle 27, bipartite 24; cal has only 97 rows (no bipartite) |
| train / val rule_applies | 15,864 / 3,680 | balanced; val in paraphrased wording |
| train / val injection_present | 11,898 / 2,760 | balanced; disjoint template families |

`worth_asking` best-gain quantiles (absolute Brier, train): p10 0, p25 0.0005, median 0.0024, p75 0.0056, p90 0.0147; val: median 0.0039, p90 0.0193.  `GAIN_MIN = 1e-3`
makes about a third of the states "no" (the posterior is far enough from 0.5 that no affordable answer is expected to pay).  `typology` is thin (cal 97 rows): expect the cal
conformal set to fall back to the full set or the tier to be retired; the report flags both.


## 3. Cascade, retirement, typed API

Per decision type (Mondrian): temperature per tier on val `fit`; conformal LAC sets of BOTH tiers on `cal` (alpha 0.05); everything reported
on `eval`.  Fast tier settles on a singleton set; else the large tier (the backbone server) on its own set; else the safe default (ask /
quarantine / no hint).  P(an answer is accepted and wrong) <= 2 alpha is the marginal bound (union over the tiers), as before.

**Retirement rule (fixed before any run).** A tier is retired for a decision type when its argmax accuracy on `cal` does not exceed the
`fit`-majority label's accuracy on `cal` (`retire_margin` 0): its conformal set can only be a singleton by luck of the base rate, so the
decision goes straight to the next tier.  This is how `typology` is retired "honestly" if the compact description cannot support it.
Eval plays no part.

**Typed API** (`ftm/fastmodel/typed.py`): `TypedDecider.decide(decision, text) -> {label: probability}` over the enumerated label set; one
forward pass reads all options at once (top logprobs of the first token); above 26 options a two-stage score-then-choose
(group letter, then label inside the group).  The node consumes only these distributions and the cascade, never free text.

## 4. Integration (`ftm/graph/system1.py`, `ftm/graph/pipeline.py`)

* `worth_asking` before each `plan` step (node `system1`): settled "no" -> `stop_reason = system1_not_worth_asking`, the planner LLM call is skipped.
* `injection_present` on every reply note in `act`: settled "no" keeps the note, otherwise (yes, unsure, default) the note is removed from the log the prompt is built from
  (the text stays in `system1.jsonl` only; fusion never read notes).
* `typology` once in `prepare`: a settled non-`none` label becomes `system1_typology_hint` in the alert summary and keys the library lookup.
* A server error is "unsure" (next tier, then the safe default).  Both tiers are real served models; there is no mock System 1 (`--system1` needs `--llm endpoint`).
* Logging: `system1.jsonl` (every decision: alert, step, decision, tier, label, conformal sets, seconds, error, quarantined), the same records inside `traces.jsonl`
  (`event: system1`), per-alert counters in `per_alert.jsonl` (`s1_*`), `summary.json["system1"]`.
* Flags: `--system1 FROZEN --system1-endpoint URL --system1-model NAME --system1-functions worth,inject,typology --system1-tiers cascade|fast`.  Off by default.

## 5. Memory: both models on one 16 GB card (`configs/llm/serving.yaml`, `scripts/serve_llm.py --dual system1`)

Card 16,303 MiB; desktop 0.9-1.3 GB (measured 1,259 MiB today) -> 15.0-15.4 GB usable.  Weights on disk: Gemma 4 12B W4A16 9.6 GB, Qwen3-4B AWQ 2.5 GB.
Gemma KV (config: 48 layers, 40 sliding-window (1,024) with 8 KV heads x 256, 8 full with 1 KV head x 512): about 16 KB per token on the full layers plus
a window-capped 335 MB per sequence on the sliding layers = ~0.45 GB per 7k-token sequence at fp16, half at fp8.  Qwen3-4B KV: 36 x 8 x 128 x 2 x 2 B = 147 KB/token (fp16).

| process | weights | activations + CUDA context | KV | total | `--gpu-memory-utilization` |
|---|---|---|---|---|---|
| gemma4-12b-dual (max-model-len 7,680, fp8 KV, 4 seqs) | 9.6 | ~1.2 | ~1.5 | ~12.3 GB | 0.76 |
| qwen3-4b-fast (2,048 ctx, fp8 KV, 8 seqs, `--enforce-eager` = no CUDA graphs) | 2.5 | ~0.3 | ~0.5 | ~3.3 GB | 0.21 |
| sum | | | | ~15.6 GB (0.97 of the card) | 0.97 |

**Verdict: it does not fit with a safe margin** (15.6 GB needed vs 15.0-15.4 GB usable, and the estimates carry about +-0.5 GB).  It may start if the desktop is
below 0.9 GB; the smoke test below decides.  Start order matters (each vLLM sizes itself from the memory free at start-up): Gemma first, then the fast server.
**Measured 8 Nov (vLLM 0.31, desktop 1.1 GiB):** at 0.76 Gemma used 9.04 GiB outside its KV cache (weights 8.6, activations 0.41, CUDA graphs 0.03), well under the ~10.8 GB estimate, and the fast server then found 2.7 GiB free, less than its 3.25 GiB, and failed. The profile now uses 0.68 (10.52 GiB): 1.48 GiB of fp8 KV, enough for the 4 sequences, and about 3.8 GiB left for the fast server.
Fallbacks, in order: (1) close the desktop session (0.9 -> ~0.3 GB); (2) drop the Gemma KV to `--max-num-seqs 2` (the integration runs then use `--concurrency 2`);
(3) the offline path needs no dual serving: fast and large scoring of the decision set, calibration and the cascade report run one server at a time
(`scripts/t3_qwen_session.sh fast3|large3`), only the live integration arms need both.  The fast tier is NOT run as a prompt on the Gemma server: that would be a different model.

## 6. Evaluation protocol (written before any run)

**Arms** (ours compose, tuned profile, val, concurrency 4, same alerts, same seed): `off`; `worth` (gate only); `inject` (screen only); `typology` (hint only);
`all_fast` (all three functions, fast tier only); `all` (all three, fast + large tier + default).  Fit and cal parts are run (choose); eval is run once afterwards with the frozen selection.

**Metrics per arm** (`scripts/system1_report.py`): Brier, AUC, PR-AUC, log loss, ECE, calibration slope, precision / recall / F1 / escalation rate at the frozen threshold 0.3227,
flipped calls vs OFF (right-to-wrong, wrong-to-right), questions / eps / planner LLM calls / tokens per alert, median and p90 seconds per alert and the end-to-end speed-up (median OFF /
median ON, and ratio of means with a CI), per-decision tier shares (fast / large / default) and their latencies, the cascade's realised error on the decision-set eval rows against 2 alpha, and
injection detection precision / recall / false-quarantine rate per held-out family (with the screen; without it recall is 0 by construction).  Paired contrasts against OFF use the
cluster bootstrap of `ftm/eval/bootstrap.py` (attempt+accounts clusters, `outputs/e18/clusters.json`; row-level and labelled if the file is missing), 95% percentile CIs.

**Selection rule (fit+cal only).**  delta = 0.002 absolute Brier (non-inferiority margin; about 2% of the planner's alert-level Brier), alpha = 0.05.
Family of non-inferiority tests, one per arm (`H0: paired Brier(arm) - Brier(off) >= delta`, bootstrap one-sided p), Holm-adjusted (`ftm/eval/stats.holm`).
* `worth` is adopted iff non-inferior (Holm) AND the 95% CI of the planner-LLM-calls-per-alert difference lies below 0.
* `inject` is adopted iff non-inferior AND the live false-quarantine rate on screened notes is <= 5%.
* `typology` is adopted iff non-inferior AND the point estimate of the Brier difference is <= 0.
* tiers: `cascade` is adopted over `fast` iff `all` is non-inferior and the large tier settles >= 5% of the live decisions; otherwise `fast`.
The adopted set is written to `selection.json` by the fit+cal report and applied unchanged to eval.  Nothing in the rule reads eval.

**H5 per clause** (exploratory, uncorrected, as in `ftm/eval/final.py`): (a) bound: realised `error_marginal <= 2 alpha` per decision (decision-set eval rows); (b) most routine decisions:
live share settled by the fast tier > 0.5 (and by any tier); (c) >= 10x faster: decision latency large/fast >= 10 (live medians) and end-to-end seconds per alert OFF/ON >= 10;
(d) no detection loss: the 95% CI upper bound of the paired Brier difference ON - OFF <= delta, and right-to-wrong flips <= wrong-to-right flips.  A clause not met is reported as not met.

**Calibration comparison (offline, on cal/eval rows, per decision):** plain SFT + temperature vs the soft-target arms (`soft_ce`, `brier`; section 8): multiclass Brier, log loss, ECE, reliability bins
(`eval_metrics` in the report), kept-fast share and error vs the bound.  Which fast adapter is frozen: the lowest cal log loss summed over the learned decisions among arms whose cascade is within
2 alpha on cal on every decision; ties -> plain SFT, then the lowest seed.  Eval plays no part.

**Consistency (Jev: similar answers for similar inputs)** (`scripts/system1_consistency.py`): on val `eval` rows, meaning-preserving perturbations (answer lines reordered, shape fields reordered,
numbers written differently, a note's carrier sentences reordered with a re-seeded bank id); per decision the flip rate of the argmax and the mean / p90 total-variation distance between the calibrated
distributions.  **Agreement with the large model** (`eval_metrics.agreement_with_large`): mean TV, mean KL(large || fast), top-1 agreement; **latency** per decision from `latency.json` of the 5070 Ti.

## 7. Faithfulness to TypeSafe's System One / Jev

Source: https://typesafe.ai/blog/introducing-system-one-models-and-jev (published properties only).

| Jev property | Ours | Status |
|---|---|---|
| Typed decisions over an option set defined in advance; never a type error | `Decision` with a fixed label tuple; the answer is a label index read from a logprob distribution, so it cannot be outside the set (`TypedDecider`) | match |
| Cardinality up to 255, two-stage score-then-choose above that | one pass up to 26 single-letter options, two-stage above (`typed.py`); our sets have 2 and 6 options | partial (mechanism yes, the 255 scale not needed or tested live) |
| All option probabilities output in parallel, not autoregressively | all options read from ONE forward pass (first-token top logprobs); no generation beyond one token | match in effect; their non-autoregressive "parallel sampler" architecture is not reproducible (unpublished) |
| Calibrated confidence on every output | temperature scaling on `fit` + Mondrian conformal sets on `cal`; reported as ECE, reliability bins, realised error vs bound | match for the guarantee we state (stronger: a marginal error bound) |
| Calibration learned in training with RL ("RLCD") | soft-target + proper-scoring-rule fine-tune (`soft_ce`, `brier`, targets mixed with the large model's distribution), compared with SFT + T on cal | partial: a publishable stand-in; RLCD itself is unpublished |
| Consistency under similar inputs | flip rate and TV under meaning-preserving perturbations, per decision | measured, not guaranteed |
| 70-500 ms latency | one forward pass of a 4B AWQ+LoRA model on a 16 GB card; measured per decision (E10: ~17 ms median, one request at a time) | match in order of magnitude; their hardware and numbers differ |
| Used as "smart if-statements" for classify / route / score / verify / guardrail / jailbreak detection | worth_asking = route (call the planner or not), injection_present = guardrail / jailbreak screen, typology = classify, stop = verify (exact), rule_applies = verify | match |
| Evaluated against frontier-LLM reference probabilities | agreement with the large model's distribution (TV, KL, top-1) AND against ground truth | match |
| New architecture, training data, training recipe | LoRA on Qwen3-4B, our decision set | not reproducible (unpublished) |

Extensions beyond Jev: a **conformal cascade** (fast -> large model -> safe default) with a stated marginal error bound 2 alpha; **retirement** of a tier that does not beat the base rate on cal;
exact decisions computed instead of learned; an ablation protocol (off / each function / fast-only / cascade) with non-inferiority tests on the end-to-end Brier.

## 8. GPU commands and time (for the lead; the GPU is shared, so each block says what it needs)

Estimates: the E10 fast fine-tune measured 9,230 s for 59,450 examples (2.56 h per seed); the new train set is 44,318 rows, so about 1.9 h per seed.  Scoring runs are 1-token requests at
~170 rows/s (4B, 32 workers) and ~30 rows/s (Gemma, 8 workers).  Integration runs: about 23 s per alert at concurrency 1 (Gemma, STATUS), about 8-10 s of wall time per alert at concurrency 4.

```
cd /home/akshat/projects/tt-sys1   # or the integration checkout after merge; PYTHONPATH must point at it
# 0. data (CPU, done in this worktree: data/decisions/hi_medium_s1v1)
# 1. large tier alone (Gemma 4 12B, solo profile): token-logprob scores of every val row, and of the train rows for the soft targets   ~1.5 h
scripts/t3_qwen_session.sh large3 gemma4-12b
.venv/bin/python scripts/score_fast_baseline.py --decisions data/decisions/hi_medium_s1v1/decisions_train.jsonl score --tier large-logprob --train-rows \
    --model gemma4-12b --workers 8 --out outputs/s1v1/large/large_lp_train.jsonl
# 2. fast fine-tune, SFT arm, 1 seed first (QLoRA takes the whole GPU)                                                              ~1.9 h per seed
.venv/bin/python scripts/train_planner.py fast-data --out outputs/sft/fast_train_s1v1.jsonl
.venv/bin/python scripts/train_planner.py train --config configs/llm/sft_fast.yaml --dataset outputs/sft/fast_train_s1v1.jsonl --seed 0
#    then seeds 1 and 2 (same command); soft-target arm (needs step 1):
.venv/bin/python scripts/train_planner.py fast-data --soft outputs/s1v1/large/large_lp_train.jsonl --mix 0.5 --smooth 0.05 --out outputs/sft/fast_train_s1v1_soft.jsonl
.venv/bin/python scripts/train_planner.py train --config configs/llm/sft_fast_soft.yaml --dataset outputs/sft/fast_train_s1v1_soft.jsonl --seed 0   # smoke first: add --max-steps 5
# 3. fast scoring of every val row: base + adapters (one server, qwen3-4b-fast profile, port 8000 solo)                              ~0.5 h
scripts/t3_qwen_session.sh fast3 qwen3-4b-fast
# 4. consistency (perturbed twins scored by the same adapter) and the offline report + frozen file (CPU)
.venv/bin/python scripts/system1_consistency.py prepare --out outputs/s1v1/perturbed.jsonl
.venv/bin/python scripts/score_fast_baseline.py --decisions outputs/s1v1/perturbed.jsonl score --tier fast --model fast3-s0 --out outputs/s1v1/tuned_s0/perturbed_fast.jsonl
.venv/bin/python scripts/score_fast_baseline.py report --fast outputs/s1v1/tuned_s0/fast.jsonl --large outputs/s1v1/large/large_lp.jsonl \
    --fast-model fast3-s0 --large-model gemma4-12b --latency outputs/s1v1/tuned_s0/latency.json --freeze outputs/s1v1/system1_frozen_s0.json --out outputs/s1v1/tuned_s0/report.json
.venv/bin/python scripts/system1_consistency.py report --perturbed outputs/s1v1/perturbed.jsonl --orig-scores outputs/s1v1/tuned_s0/fast.jsonl \
    --pert-scores outputs/s1v1/tuned_s0/perturbed_fast.jsonl --temperatures outputs/s1v1/system1_frozen_s0.json --out outputs/s1v1/tuned_s0/consistency.json
# 5. DUAL-SERVE SMOKE TEST (lead): both servers on the card; must answer both models and leave >= 300 MiB free                      ~0.2 h
export FTM_LLM_ENDPOINT=http://127.0.0.1:8000/v1 FTM_FAST_ENDPOINT=http://127.0.0.1:8001/v1
.venv/bin/python scripts/serve_llm.py --dual system1 &      # starts gemma4-12b-dual (8000) then qwen3-4b-fast (8001); watch for OOM
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
.venv/bin/python scripts/run_llm_agent.py --system planner --alerts val --val-part fit --data hi_medium --n 8 --llm endpoint --model gemma4-12b-dual \
    --system1 outputs/s1v1/system1_frozen_s0.json --system1-model fast3-s0 --system1-endpoint $FTM_FAST_ENDPOINT --run-dir outputs/s1/smoke
#    pass = per_alert.jsonl has s1_decisions > 0, system1.jsonl has fast and (some) large tiers, no OOM in either server log
# 6. INTEGRATION ARMS (dual server up; tuned profile = concurrency 4). One command per arm and part; fit 191 alerts, cal 109 (askable), eval 160     ~1 h per arm for fit+cal
for ARM in off worth inject typology all_fast all; do
  case $ARM in
    off)      S1="" ;;
    worth)    S1="--system1 $FROZEN --system1-functions worth" ;;
    inject)   S1="--system1 $FROZEN --system1-functions inject" ;;
    typology) S1="--system1 $FROZEN --system1-functions typology" ;;
    all_fast) S1="--system1 $FROZEN --system1-tiers fast" ;;
    all)      S1="--system1 $FROZEN" ;;
  esac
  for PART in "fit 191" "cal 109"; do set -- $PART
    .venv/bin/python scripts/run_llm_agent.py --system planner --alerts val --val-part $1 --n $2 --data hi_medium --llm endpoint --model gemma4-12b-dual \
        --system1-model fast3-s0 --system1-endpoint $FTM_FAST_ENDPOINT $S1 --run-dir outputs/s1/fitcal/$ARM/$1
  done
done      # FROZEN=outputs/s1v1/system1_frozen_s0.json; the "off" arm runs on the SAME dual server (same Gemma settings) so only System 1 differs
# 7. report on fit+cal -> selection.json; then eval once with the frozen selection (same loop with --val-part eval --n 160 and --run-dir outputs/s1/eval/$ARM)   (CPU)
.venv/bin/python scripts/system1_report.py report --part fitcal --off off=outputs/s1/fitcal/off/fit,outputs/s1/fitcal/off/cal \
    --arm worth=outputs/s1/fitcal/worth/fit,outputs/s1/fitcal/worth/cal --arm inject=... --arm typology=... --arm all_fast=... --arm all=... \
    --offline outputs/s1v1/tuned_s0/report.json --out outputs/s1/fitcal_report
# 8. E15 with and without the screen on the held-out injection families (planner, harness; the existing script, fast model as the screen)
.venv/bin/python scripts/run_injection_tests.py --system planner --alerts val --val-part fit --data hi_medium --llm endpoint --model gemma4-12b-dual \
    --screen fastmodel --screen-endpoint $FTM_FAST_ENDPOINT --screen-model fast3-s0 --arms nogate,gate,gate+screen --set <val-family set from ftm.safety.injection.generate_diverse(...,'val')>
```

Total GPU time, planning figures: step 1 ~1.5 h; step 2 SFT 3 seeds ~3 x 1.9 h, soft arm another 3 x 1.9 h (run the soft arm on 1 seed first, 2 more only if it wins on cal); step 3 ~0.5 h;
step 5 0.2 h; step 6 about 6 arms x 300 alerts x ~9 s = ~4.5 h for fit+cal, plus 160 alerts x 6 arms = ~2.4 h for eval; step 8 ~1 h.

## 9. Not done / caveats

* No GPU run: no accuracy, speed-up or Brier number exists yet.  Every claim above about behaviour is a design statement or a unit test.
* `train_soft` (the soft-target fine-tune) is untested on a GPU (the loss, targets and data path are unit-tested; run it with `--max-steps 5` first).
* The dual-serving memory budget is an estimate; the verdict (does not fit with margin) should be confirmed by the smoke test.
* Multi-token options (teacher-forced option scoring) are not implemented: no decision needs them.
* `scripts/e10_final.py` (the blind test-row scorer) was moved to the v3 frozen format but never run on test; `configs/final/jobs.yaml` still names `outputs/e10/fast_v2.json` and adapter `fast-s2`: change both to the frozen file and adapter chosen above when the final jobs are regenerated.
* The E15 script's `fastmodel_screen` now sends the note in the training format with an unknown bank id ("?"); a small domain shift, noted.
