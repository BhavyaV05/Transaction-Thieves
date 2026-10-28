# T3: planner fine-tuning (C10) and the System 1 fast model + cascade (C13)

Owner: T3 (Akshat). Spec: `tracks.md` §5.4 (fine-tuning steps 1–4), §5.6, `plan.md` §6.11, §6.14. Experiments: E16 (fine-tuned planner), E10 (fast model, H5).

**Status (28 Oct):** distillation is done; QLoRA training, the E16 val runs and the E10 fast-model runs are queued on the 5070 Ti (`scripts/gpu_queue.sh`, queue file `outputs/t3_queue/queue.txt`; results below when they finish).

## 0. Environment (28 Oct)

Training venv `~/projects/train-venv` (Python 3.12), separate from `.venv` and from the vLLM venv (vLLM 0.31.0 is untouched):

| package | version |
|---|---|
| torch | 2.8.0+cu128 (sm_120 works) |
| transformers | 5.21.0 |
| trl | 1.16.0 |
| peft | 0.21.2 |
| bitsandbytes | 0.50.2 |
| accelerate | 1.15.0 |
| datasets | 5.1.0 |

Full freeze: `~/projects/train-venv/freeze.txt` (not committed). **Unsloth is not used**: pip resolves `unsloth` against torch 2.8.0 to 2025.7.2, a mid-2025 release that does not run with transformers 5. `scripts/train_planner.py train` therefore uses plain PEFT: 4-bit NF4 base with double quantisation and bf16 compute (bitsandbytes), LoRA r 16 / alpha 32 on all seven linear projections, gradient checkpointing, TRL `SFTTrainer` with prompt/completion pairs so that the loss covers the completion only (`responses_only`). The prompt is rendered exactly as inference sees it: Qwen3 chat template with `enable_thinking=False`, i.e. the generation prompt ends with the empty think block, and the completion is the assistant JSON (or the one label token) plus `<|im_end|>`.

Base models (pinned, `models--Qwen--*` in `~/hf_cache`): Qwen3-8B `b968826d9c46dd6066d109eabc6255188de91218` (planner), Qwen3-4B `1cfa9a7208912126459214e8b04321603b3df60c` (fast model). **Serving the students (changed 28 Oct).** vLLM 0.31.0 has no bitsandbytes quantisation ("Unknown quantization method: bitsandbytes"; vLLM is not to be upgraded), so the bf16 base cannot be served 4-bit. Profiles `qwen3-8b-lora` and `qwen3-4b-lora` (`configs/llm/serving.yaml`) therefore serve the already-cached AWQ checkpoints (`Qwen/Qwen3-8B-AWQ` `4da05a8e...`, `Qwen/Qwen3-4B-AWQ` `74d4bd2b...`, the E18 Qwen3-8B candidate) with `--enable-lora`; one server serves the base ("same model prompted", the parity arm) and the adapters by name (`planner-sft-s0..2`, `fast-s0..2`). **Train/serve mismatch:** the adapters are trained on the NF4-quantised bf16 base and served on the AWQ-quantised base of the same model. The two quantisations differ, so a student can lose a little against what it was trained for; the prompted arm runs on the same AWQ base, so the comparison student vs prompted is like for like on the serving side. `scripts/t3_lora_check.sh` proves the adapters load (first queue step after the Gemma session). Fallback if LoRA on AWQ does not load: llama.cpp (`llama-server --lora`, Qwen3-8B GGUF Q4_K_M plus the adapters converted to GGUF LoRA); this page says which one was used.

## 1. Planner distillation (E16)

| Step | How | Where |
|---|---|---|
| 1. Teacher runs | The E18 backbone, **Gemma 4 12B** (revision `1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee`), on the 1,983 askable HI-Medium **train** alerts (`outputs/t3_teacher/runs`, 26 Oct), with `--keep-prompts` so every step keeps its exact prompt | `scripts/run_llm_agent.py --system planner --alerts train --keep-prompts --record ...` |
| 2. Filter | Keep a question step only if it got a reply, its (canonical, ε) is **accepted** in back-testing, and it moved the posterior **toward the true label** (\|after − y\| < \|before − y\|). Keep that alert's stop step only if at least one of its questions was kept. Train split only; test is refused | `ftm/planners/llm_distill.py`, `train_planner.py distill` |
| 3. SFT | QLoRA via PEFT (section 0), loss on the assistant JSON only, Qwen3 chat template with `enable_thinking=False` (as at inference), 3 seeds | `configs/llm/sft_planner.yaml`, `train_planner.py train` |
| 4. Compare | Tuned student vs the same model prompted, vs the teacher, vs the E18 backbone, all through `run_llm_agent.py` with the adapter served by vLLM (LoRA) | E16 |

**Evidence view for training.** The default is top-40 by amount, the view the stub produces today. The week-7 decision with Abhishek (`tracks.md` §5.10) can change it; the student must be trained on the view used in the final run.

**Distillation result (28 Oct).** `scripts/train_planner.py distill --runs outputs/t3_teacher/runs/c0?? --data hi_medium --backtest data/backtest/hi_medium/7df6bad5337d/lr_table.jsonl --teacher-model gemma4-12b --teacher-revision 1d2c2d7f... --out outputs/sft/planner_train.jsonl`, naive summed fusion (`--fusion-params` omitted: the posteriors are recomputed with the fusion the system runs on, and the final run's fusion is the naive default until the G2 decision; the filter refuses the naive fusion unless it is asked for by leaving the flag out). 3,470 teacher steps; 393 had no saved prompt (the prompt-overflow alerts of the teacher runs); **2,817 question steps, of which 252 are kept** (accepted in the realistic back-test table, answered, and moved the posterior toward the true label); **0 stop steps** (the Gemma teacher never chose `stop`: it ends when the eps or the calls run out, so the student sees no stop example). So the SFT set is **252 examples (252 alerts)**, a very small set: the student is trained anyway (2 epochs, 3 seeds) and its result is read with that in mind. Adapters go to `models/` (gitignored); trained weights are never released (`plan.md` §10).

## 2. Fast model (E10)

**Decisions** (`ftm/fastmodel/decisions.py`). Each label is answered with **one token** (yes/no, or letters A–I for the typology plus none):

| Decision | Labels | Safe default when neither tier is sure |
|---|---|---|
| `worth_asking` | yes / no | yes (keep investigating) |
| `typology` | 8 typologies + none | none: the planner decides |
| `rule_applies` | yes / no | no (do not ask without a permitting rule) |
| `injection_present` | yes / no | yes (treat an unclear note as an injection) |
| `stop` | yes / no | no (keep investigating) |

**Expected dataset format from T2 (hand-off #13; please confirm):** rows of `decision`, `context` (aliased text), `label` (a label name above), `alert_id`, `split`. `to_sft_example` turns a row into a chat example for `configs/llm/sft_fast.yaml`.

**Models.**
- `LogprobFastModel`: our tuned Qwen3-4B, or any served model, e.g. Qwen3-1.7B as a comparison. One forward pass with `max_tokens=1`. It reads the top-20 logprobs of the first token, keeps the label tokens (case and leading space ignored; missing ones floored at 1e-6), renormalises, and divides by temperature T.
- `JsonDecisionModel`: the "8B writes JSON" baseline. The stated confidence is P(label), and the remainder is spread evenly over the other labels.
- ModernBERT/DeBERTa classifiers and RouteLLM-style cascades come from T2. SemIf and Laya (stretch) are not started.

**Calibration** (`calibration.py`): temperature scaling fitted by minimising NLL on validation, with ECE reported.

## 3. Conformal cascade (stretch)

`cascade.py` uses Mondrian (per decision type) split conformal with the LAC score 1 − p(true), fitted on a calibration part of validation:
- If the fast model's set has one label, accept it.
- Otherwise ask the 8B; if its set has one label, accept that.
- Otherwise take the safe default.

`evaluate` reports, per decision type:
- the share kept by the fast tier, the share sent to the 8B and the share sent to the default;
- **`error_marginal` = P(accepted and wrong)**, which is the quantity bounded by 2α;
- `error_given_accepted`, which is *not* bounded and can exceed 2α when few answers are accepted.

H5 ("handles most routine decisions within its bound, ≥10× faster, no detection loss") is read off this report plus the latency lists on `LogprobFastModel.latencies` and `JsonDecisionModel.latencies`, measured on the reference machine.

**Fallback** (`tracks.md` §5.6): if the 4B fails, the 8B makes every small decision (`JsonDecisionModel`, or `LogprobFastModel` pointed at the 8B).
