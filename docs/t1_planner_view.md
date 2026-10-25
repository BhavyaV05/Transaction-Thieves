# Planner evidence view: decision memo (week 7, T1)

Owner: T1 (Abhishek). **Confirmed by Akshat (T3) on 2026-10-25:** the planner is trained on the `amount_1hop` top-40 stub; `scorer_pcst` stays the E8 inference-time arm (run zero-shot on the fine-tuned planner, and said so). `top_features` is hidden from every prompt except Sharma's (`_SUMMARY_HIDDEN` in `ftm/planners/llm.py`, `ftm/external/text_agent.py`, `ftm/external/_infoseek.py`). Rule (tracks.md 3.8): the fine-tuned planner is trained on top-40 evidence **unless scorer + PCST is ready and better on validation**. Numbers: `ftm/retrieval/README.md` (views tables, H6), `data/runs/evidence_views_<ds>_scorer_cal_eval.json`, `data/runs/h6_<ds>.json`. Val `cal`+`eval` alerts only, no test row.

## Recommendation

**Train the planner on the top-40 view it already uses (T3's week-2 stub, `view(alert, "amount_1hop")`), not on scorer + PCST.** Scorer + PCST is ready (one call, tested, 0.4 s per alert) but it is **not shown to be better on validation**: against the stub it finds 1.2 to 3.0 points more of the pattern edges (CI [0.0, 6.9] and [0.0, 3.1] points, both include 0), and the pre-registered H6 statistic is far from its margin. Keep it as the **E8 inference-time arm** and as the candidate for an upgrade if T3's planner runs show a Brier gain; per the plan only a subset of teacher traces is then regenerated. If Akshat prefers one evidence format for the whole pipeline, the cost of using scorer + PCST for training is the table at the end; the retrieval numbers give no reason to expect a better planner from it.

## The evidence

**H6, pre-registered** (`prereg/hypotheses.yaml`): `pattern_edge_share(scorer_pcst) - pattern_edge_share(top40_by_amount)`, reject only if the CI lower bound is >= 0.15 **and** the Brier-difference CI upper bound is < 0. Share = pattern edges in the view / edges counted. Paired bootstrap over attempts, 10,000 resamples; alerts with at least one pattern edge in the 72 h window.

| | HI-Small (52 alerts, 26 attempts) | HI-Medium (163 alerts, 91 attempts) |
|---|---|---|
| share, scorer + PCST (denominator 40) | 0.064 | 0.062 |
| share, plain top-40 by amount | 0.001 | 0.001 |
| **difference, denominator 40** [95 % CI] | **+0.064** [0.030, 0.104] | **+0.061** [0.044, 0.079] |
| difference, denominator = edges selected (15.7 / 17.2) | +0.164 [0.050, 0.353] | +0.162 [0.121, 0.208] |
| share half of H6 rejected? (lower bound >= 0.15) | no, either denominator | no, either denominator |
| Brier half | **pending** (T3, below) | **pending** |

Two things to confirm with T4: (a) **the denominator.** The trail views return fewer than 40 edges (15.7 and 17.2 on average). Read literally ("among the 40") the share is divided by 40, so a view that returns 15 edges is not rewarded for being short; that is the reading used for the verdict here. The alternative divides by the edges returned and gives 0.16, still with a lower bound under 0.15. (b) **the margin itself is unreachable on this data with denominator 40**: an alert has 3.7 (HI-Small) and 2.8 (HI-Medium) pattern edges in its window on average, so even a perfect retrieval has a share of at most about 0.07 to 0.09. If the margin is meant as an absolute share, H6 cannot be rejected by any retriever; T4 may prefer a margin on recall or on the selected-edges share before test.

**Secondary context (not the pre-registered statistic).** Recall of the pattern edges in the window, alerts with pattern edges (`recall_secondary`): scorer + PCST 0.651 / 0.843 (HI-Small / HI-Medium), the stub 0.621 / 0.831, scorer only 0.591 / 0.447, plain top-40 by amount 0.006 / 0.006, long prompt 0.003 / 0.012. Differences of scorer + PCST: +64.5 / +83.7 points over the plain top-40, +3.0 [0.0, 6.9] / +1.2 [0.0, 3.1] over the stub, +6.1 [-9.9, 22.8] / +39.7 [31.2, 47.8] over the scorer alone, 0.0 over the greedy time-respecting trail (PCST adds nothing). Where the stub's 1-hop pool has at least 40 edges (the only place a better choice can matter; 19 and 21 alerts) the gain over the stub is +8.3 [0.0, 17.9] and +4.8 [0.0, 14.3] points. 72 % (HI-Small) and 89 % (HI-Medium) of the pattern edges in the window are reachable by a time-respecting walk from the alerted edge, so 0.65 and 0.84 are close to what that construction can reach. What the time-respecting constraint buys is structure: both trail views, and the stub, are always one connected time-respecting trail; the scorer-only and plain top-40 views never are.

**The long prompt is not a competitor under the harness cap.** All candidate edges (12,000 to 35,000 per alert) cannot be shown: at 4,000 estimated tokens (the harness prompt cap of 6,000 minus about 2,000 for instructions, alert, question menu and history) it holds the newest 58 edges, 0.3 % of the window, and finds 0.3 to 1.2 % of the pattern edges. Cap check against the Qwen3 tokenizer (9 alerts): the 4,000-token estimate corresponds to 3,680 real tokens on average (max 3,782).

## What T3 would call and what it costs per alert

```python
from ftm.retrieval.evidence_views import view
v = view(alert_context, "scorer_pcst", k=40, data="hi_medium")   # or "amount_1hop" (today's stub), "amount", "scorer", "long_prompt"
prompt_evidence, ids = v.text, v.tx_ids       # aliased, time-ordered, no labels; ids are for bookkeeping only
```

| view | edges (mean, HI-Medium) | real Qwen3 tokens (mean over 9 alerts) | CPU per call |
|---|---|---|---|
| `amount_1hop` (today's stub) | 13.8 | 600 | 0.07 s |
| `scorer_pcst` | 17.2 | 957 (max 2,530) | 0.38 s |
| `scorer` | 40 | 2,586 | 0.40 s |
| `amount` | 40 | 2,773 | 0.09 s |
| `long_prompt` | 58 | 3,680 | 0.10 s |

Scorer + PCST costs +350 tokens per alert and +0.3 s CPU over the stub, one-off load of the scorer scores (200 MB parquet for HI-Medium, 37 MB HI-Small), no GPU. The stub's token cost is low because the 1-hop pool is small; both fit the 6,000-token cap with room (the trail's maximum, 2,530 tokens, is a hub alert).

## What T3 must run for the Brier half of H6 (E8)

The same planner, fusion, seeds, back-test table and alerts (val `eval` part), four runs that differ only in the evidence text: `view(alert, "amount")` (the pre-registered baseline), `"scorer_pcst"`, and, as context, `"amount_1hop"` (the current default) and `"long_prompt"`. Per-alert Brier of the final probability for each, paired difference `Brier(scorer_pcst) - Brier(amount)`, cluster bootstrap by attempt+accounts, hub cap on HI-Medium rows as in `docs/t1_handoffs.md`. H6 is rejected only if that CI's upper bound is below 0 **and** the share half holds (it does not, above). If the planner is fine-tuned on `amount_1hop` evidence, run the `scorer_pcst` arm zero-shot at inference and say so.
