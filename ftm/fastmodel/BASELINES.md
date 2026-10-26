# Fast-model component baselines: ModernBERT / DeBERTa classifiers and RouteLLM-style routing (E10)

Owner: T2 (Bhavya). Spec: `plan.md` 7.3 and 8 (E10), `tracks.md` 4.9, week 10 (secondary). Files: `encoder.py`, `router.py`, `scripts/encoder_baseline.py`. T3 owns the rest of this package (the LLM fast model, calibration, the conformal cascade); these files only add tiers that plug into it.

## What is compared

| Row | What | Where |
|---|---|---|
| Our fine-tuned 4B | single-token logprobs, temperature-scaled | T3, `model.py` |
| 8B JSON baseline | the 8B writes label and confidence | T3, `model.py` |
| **ModernBERT-base**, **DeBERTa-v3-base / -small** | one fine-tuned classifier per decision, trained on the TRAIN rows of the decision set | `encoder.py` |
| **TF-IDF + logistic regression** | the floor: a linear model that needs no GPU | `encoder.py` |
| Conformal cascade | fast tier, then the 8B, else the safe default; marginal error of an accepted answer <= 2 alpha | T3, `cascade.py` |
| **RouteLLM / FrugalGPT-style router** | a score for "the fast answer is right", a threshold set to send a chosen share to the 8B | `router.py` |

Models are pinned in `ENCODERS` (hub id and revision): ModernBERT-base (Apache-2.0, 149M), DeBERTa-v3-base and -small (MIT). Max length 256, batch 16, bf16 autocast on CUDA, optional gradient checkpointing for the 4 GB card.

## Plugging in

`EncoderDecisionModel.raw_logps(ex)` and `TfidfDecisionModel.raw_logps(ex)` return the label log-probabilities in `DECISIONS` order, the same as `LogprobFastModel.raw_logps`. `scripts/encoder_baseline.py score` writes `{"i", "logps"}` rows, the format of `score_fast_baseline.py score --tier fast`, and selects rows with T3's own `load_rows` (same `--cap`, same seed), so `score_fast_baseline.py report --fast <file> --large <8B file>` gives the conformal cascade with a classifier as the fast tier, and `encoder_baseline.py route` gives the routers on the same rows. A test runs T3's `report` on the file this script writes.

```
python scripts/encoder_baseline.py train --model modernbert-base --out outputs/fast/modernbert        # 4 GB GPU, per decision
python scripts/encoder_baseline.py score --model-dir outputs/fast/modernbert --out outputs/fast/fast_modernbert.jsonl
python scripts/encoder_baseline.py tfidf --out outputs/fast/fast_tfidf.jsonl                          # the floor, CPU
python scripts/encoder_baseline.py route --fast outputs/fast/fast_modernbert.jsonl --large outputs/fast/large_8b_json.jsonl --out outputs/fast/route_modernbert.json
```

## The router

Two kinds. `learned`: a ridge logistic regression on the fast tier's confidence (top probability, margin to the second label, normalised entropy) and the decision type, fitted on the validation `fit` part to predict whether the fast argmax is right. `maxprob`: no training, the score is the top probability (FrugalGPT's confidence rule). The threshold is the quantile of the score on `cal` that sends a chosen share of rows to the 8B (RouteLLM sets the cost the same way). Same split discipline as `splits.py`: temperature and router on `fit`, thresholds and conformal sets on `cal`, every number on `eval`.

`route_report` gives, per router, the error-versus-large-share curve (0 to 100% in steps of 10) from the fast tier alone to the 8B alone, the AUC of its score for separating right from wrong fast answers, and a **matched** row: the router at the share of `cal` rows the conformal cascade did not settle with the fast tier.

**Read the comparison with its caveats.** The router always answers and has no coverage guarantee; the conformal cascade bounds the marginal error of an accepted answer and abstains to the decision's safe default when neither tier is sure. So the report keeps both error numbers for the conformal cascade: `error_accepted_and_wrong` (the quantity the 2 alpha bound is about; a default is not "wrong") and `error_default_as_answer` (the default taken as the answer, the like-for-like number against a router, and the harsher one for it). A router that wins on the second and loses on the first has bought accuracy by giving up the bound; that is the trade the plan wants shown (H5), not a verdict.

## Status

Built and tested on toy and synthetic decisions (`tests/test_encoder_router.py`): a tiny random encoder trains, saves and reloads; the pinned ModernBERT-base fine-tunes, saves and scores on CPU (smoke run on 64 rows). **No real numbers**: this laptop has no CUDA build of torch and no `data/` (no real decision set, no 8B scores, no `val_part` alert set). To run on the 3050 Ti: install a CUDA build of torch, then the commands above. Install the extra with `pip install -e ".[enc]"`.
