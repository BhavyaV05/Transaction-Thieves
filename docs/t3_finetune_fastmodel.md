# T3: planner fine-tuning (C10) and the System 1 fast model + cascade (C13)

Owner: T3 (Akshat). Spec: `tracks.md` §5.4 (fine-tuning steps 1–4), §5.6, `plan.md` §6.11, §6.14. Experiments: E16 (fine-tuned planner), E10 (fast model, H5).

**Status:** the code, configs and tests are done. **Nothing has been trained.** Training needs:
- teacher runs on train alerts (the endpoint and harness v0);
- the back-test table (hand-off #9);
- the small-decision dataset (hand-off #13, T2, week 6);
- GPU windows blocked in `GPU_CALENDAR.md`.

## 1. Planner distillation (E16)

| Step | How | Where |
|---|---|---|
| 1. Teacher runs | The strongest E18 candidate (default Qwen3.8-27B, or gpt-oss-20b if it scores better on val) on **train** alerts, with `--keep-prompts` so every step keeps its exact prompt | `scripts/run_llm_agent.py --system planner --alerts train --keep-prompts --record ...` |
| 2. Filter | Keep a question step only if it got a reply, its (canonical, ε) is **accepted** in back-testing, and it moved the posterior **toward the true label** (\|after − y\| < \|before − y\|). Keep that alert's stop step only if at least one of its questions was kept. Train split only; test is refused | `ftm/planners/llm_distill.py`, `train_planner.py distill` |
| 3. SFT | QLoRA via Unsloth, loss on the assistant JSON only, Qwen3 chat template with `enable_thinking=False` (as at inference), 3 seeds | `configs/llm/sft_planner.yaml`, `train_planner.py train` |
| 4. Compare | Tuned student vs the same model prompted, vs the teacher, vs the E18 backbone, all through `run_llm_agent.py` with the adapter served by vLLM (LoRA) | E16 |

**Evidence view for training.** The default is top-40 by amount, the view the stub produces today. The week-7 decision with Abhishek (`tracks.md` §5.10) can change it; the student must be trained on the view used in the final run.

**Not yet run on a GPU.** `train()` follows the documented Unsloth and TRL APIs (`FastLanguageModel`, `SFTTrainer(processing_class=...)`, `train_on_responses_only`), but has not been executed here. Check it on the first real run with `--dry-run` first. Adapters go to `models/` (gitignored); trained weights are never released (`plan.md` §10).

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
