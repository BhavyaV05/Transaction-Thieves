# ftm/retrieval: edge scorer, time-respecting PCST, evidence views (C11, hand-off #11)

Owner: T1 (Abhishek). Spec: `plan.md` 6.12, `tracks.md` 3.8. T3's week-2 stub (`views.py`, PR #6) is untouched; this code lives in `edge_scorer.py`, `pcst.py`, `evidence_views.py`.

```
python3.12 -m ftm.retrieval.edge_scorer --data hi_small --train           # M3: trains on train, selects on val fit, writes models/ + data/retrieval/
python3.12 -m ftm.retrieval.evidence_views --data hi_small --parts cal eval   # M4: the views table below (~4 min, CPU)
python3.12 -m ftm.retrieval.edge_scorer --data hi_medium --train --streaming  # HI-Medium scorer: day-by-day features, streamed negative subsampling (16 min, peak RSS 8.3 GB)
python3.12 -m ftm.retrieval.evidence_views --data hi_medium --parts cal eval  # the HI-Medium views table (15 min)
python scripts/retrieval_demo.py --data hi_medium --alert V000993 --with-labels   # G4 demo, see the last section
```

## The stable API for T3: `ftm.retrieval.evidence_views.view` (week 7, E8 / H6)

```python
from ftm.retrieval.evidence_views import view
v = view(alert, "scorer_pcst", k=40, data="hi_medium")   # alert: an AlertContext, an alert id ("V000553") or {tx_id, bank, timestamp}
v.text      # the aliased trail, one line per edge, oldest first, the alerted edge last and marked "<- ALERT"
v.tx_ids    # the same edges' transaction ids, same order (server side bookkeeping, never put them in a prompt)
v.n_candidates, v.tokens_est
```

`name` is one of `amount` (top-40 by amount over the whole 72 h window), `scorer` (top-40 by edge-scorer score), `scorer_pcst` (the scorer's edges as a time-respecting PCST trail connected to the alerted edge, at most k), `long_prompt` (all candidate edges, newest kept until the evidence token cap), and the ablations `amount_1hop` (**T3's week-2 stub: the 40 largest edges that touch the alerted sender or receiver**, which is what the planner is trained on today) and `scorer_greedy`. Guarantees, each with a test (`tests/test_evidence_api.py`): candidates come from the home bank's view through `ftm.data.io` (time guard, train and val only); nothing after the alert time is read, scored or returned (`assert_not_future`); accounts are aliased with `alias_seed` (the run seed, default 0); the text is built by `views.format_trail`, which refuses raw ids and labels; an `AlertContext` whose `as_of` is not the alert's time raises. `data` is the dataset name (an alert id is only unique within one). Cost per call on one CPU core (9 alerts of both datasets): `amount` 0.09 s, `amount_1hop` 0.07 s, `long_prompt` 0.10 s, `scorer` 0.40 s, `scorer_pcst` 0.38 s, plus the one-off load of `data/retrieval/<ds>/edge_scores.parquet`; the views table below computes all six views for 1,303 alerts in 1,005 s.

**The long-prompt view and its cap.** "Everything in a long prompt" = every candidate edge of the window serialised like the other views, **cut at 4,000 estimated tokens** (`EVIDENCE_TOKENS`): the harness prompt cap is `Limits.max_tokens` = 6,000 (`configs/llm/default.yaml`; the server context is 8,192), and T3's planner prompt (instructions, alert, question menu, answer history) uses about 2,000 of them. Tokens are estimated at 1.5 characters per token (`CHARS_PER_TOKEN`); the Qwen3 tokenizer gives 1.54 to 1.62 characters per token on these lines (9 alerts, 5 views, `docs/t1_planner_view.md`), so the estimate is 3 to 8 % high and a view that the estimate puts at 4,000 tokens has about 3,700 to 3,900 real ones. When the window does not fit the **newest** edges are kept, the alerted edge always: with 12,000 to 35,000 candidate edges per alert the cap holds about 58 of them (0.3 % of the window). `max_evidence_tokens=` changes the cap; the estimate is exact for the line length (a test compares it with the formatted lines).

**HI-Medium edge scorer** (`train_scorer_streaming`, the same MLP, loss and epoch rule as the HI-Small one). HI-Medium does not fit the whole-frame feature code (56 M node-events), so features are computed one day at a time from a two-day slab with the prior counts carried in two dense counters (`chunked_timeline_features`, equal to the whole-frame values: `tests/test_retrieval_scorer.py`), negatives are subsampled while streaming (all 8,744 attempt edges and 10 % of the other train edges: 1.96 M of 19.5 M train rows, plus the 1.28 M val `fit` rows for the epoch choice), and every train+val edge is then scored from the cached day chunks. **Peak RSS 8.3 GB** (day 2 slab; 5.2 GB after day 1), 16 min in total, seed 0, 120 epochs, best epoch 59. Val `eval` part (1.40 M edges, 1,205 attempt edges, base rate 0.00086): **ROC AUC 0.9953, average precision 0.3935** (HI-Small: 0.9929 and 0.228). Check of the streaming path on HI-Small: AUC 0.9929, AP 0.2267 (whole-frame model 0.9929 and 0.2281), peak RSS 2.8 GB.

## pcst_fast needs a patched build (blocking if skipped)

The PyPI wheel `pcst_fast` 1.0.10 declares its result arrays as `py::array_t<int>` but fills them with `int64_t`, so the returned node and edge index arrays are garbage (the algorithm itself is right; the Python side reads half-width integers). No error is raised. `ftm/retrieval/pcst_fast_binding.patch` (one line) fixes the return type. Build:

```
python3.12 -m pip download pcst_fast==1.0.10 --no-binary :all: --no-deps -d /tmp/pcst && cd /tmp/pcst && tar xf pcst_fast-1.0.10.tar.gz
cd pcst_fast-1.0.10 && patch -p1 < <repo>/ftm/retrieval/pcst_fast_binding.patch
python3.12 -m pip install --user --no-cache-dir --no-deps --no-build-isolation .      # needs pybind11
```

`pcst.check_binding()` solves a 4-node path once per process and raises `RuntimeError` naming the patch if the indices are wrong; `pcst_edges` calls it first, so an unpatched wheel can never silently produce wrong trails. CI installs the unpatched wheel through the `gnn` extra, so the PCST tests skip there with the guard's message (`tests/test_retrieval_pcst.py`); the pure-numpy trail code is tested everywhere.

## Edge scorer (M3)

Label: the transaction has a row in `patterns.parquet` (it belongs to a pattern-file attempt). Laundering transactions that are in no attempt get label 0, so the scorer predicts attempt membership. Model: MLP 2 x 128 on `FEATURES2` = the edge's own attributes, past-only prior counts, and past-24 h aggregates of the sender's and receiver's flows (count and amount in and out, pass-through share); every feature uses only edges strictly before the edge's own timestamp, so it is the same at any `as_of`. Training uses train edges only (all attempt edges + 10 % of the rest, loss weights restore the base rate); the epoch is picked by AP on the val `fit` part; metrics are on the val `eval` part. Seed 0, 120 epochs, CPU, about 3.5 min.

| val `eval` part (234,572 edges, 193 attempt edges, base rate 0.00082) | value |
|---|---|
| ROC AUC | **0.9929** |
| average precision | **0.228** (val `fit` AP at the chosen epoch 0.287, epoch 111 of 120) |

The earlier 30-epoch model had AUC 0.991 and AP 0.154; the curve was still rising at 30 epochs. Pattern-edge recall at 40 per alert neighbourhood is in the next section. **`data/retrieval/<ds>/edge_scores.parquet` has a `split` column: scores of `train` edges are in-sample (the model was fit on them); only `val` scores are out-of-sample, and only `val` `eval`/`cal` rows are used for any reported number.** The model file is `models/edge_scorer_<ds>.pt` (gitignored).

## Time-respecting PCST and the evidence views (M4)

For an alert (home bank b, transaction, `as_of` = alert time) the candidates are the edges of view(b) in `[as_of - 72 h, as_of]`, read through `ftm.data.io` with `as_of` (train/val only, nothing after `as_of`). Prizes are the edge scorer's scores (log amount if no scores). Views, each returning at most K = 40 transaction ids that always include the alerted one:

| view | definition |
|---|---|
| `amount` | the 40 largest candidates by amount (own implementation of the baseline; T3's stub is not touched) |
| `amount_1hop` | the 40 largest candidates that touch the alerted edge's sender or receiver (the stub's definition, own code) |
| `scorer` | the 40 candidates with the highest scorer score |
| `scorer_greedy` | ablation: starting from the alerted edge, repeatedly add the highest-score edge that is time-respecting-eligible |
| `scorer_pcst` | G-Retriever construction: edge prizes become virtual nodes, `pcst_fast` (gw pruning) is rooted at the alerted edge's sender, and the PCST edges are boosted so the time-respecting greedy fill takes them first |

**Time-respecting:** an edge e = (u, v, t) may be added when an already selected edge f touches u or v with t_f >= t (money moves forward in time along e -> f -> ... -> alert; edge direction is ignored). `reachable_to_alert` computes the pool of such edges in one backward sweep over time, so the trail views are by construction connected, time-ordered, and contain nothing after `as_of`. Trail edges carry aliased ids (`ftm.data.alias`, `format_trail`): no raw account id, no label.

Evaluation on the HI-Small val alerts (`data/alerts/hi_small/val.parquet`, a few hundred): a **pattern edge** is a candidate edge of the same laundering attempt (`patterns.parquet`) as the alerted transaction. Labels are used only to score the views. Recall = pattern edges in the 40 / pattern edges among the candidates, averaged over the alerts that have any (52 of 636 on `cal`+`eval`; 30 of 332 on `eval` alone). "One trail" = every selected edge reaches the alerted edge by a time-respecting walk inside the selection (the H6 check); "connected" = one component of accounts.

| view (val `cal`+`eval`, 636 alerts, 52 with pattern edges) | pattern-edge recall in the 40 | pattern edges in the 40 (mean) | connected | one time-respecting trail | mean edges selected |
|---|---|---|---|---|---|
| top-40 by amount | 0.006 | 0.04 | 0.00 | 0.00 | 40.0 |
| top-40 by amount, 1 hop (stub definition) | 0.621 | 2.40 | 1.00 | 1.00 | 12.0 |
| scorer only | 0.591 | 2.52 | 0.00 | 0.00 | 40.0 |
| scorer + time-respecting greedy | 0.651 | 2.58 | 1.00 | 1.00 | 15.7 |
| **scorer + PCST** | **0.651** | 2.58 | 1.00 | 1.00 | 15.7 |

On `eval` alone (30 alerts with pattern edges): 0.010 / 0.685 / 0.534 / 0.702 / 0.702. The scorer was selected on val `fit`, so `cal` and `eval` are out-of-sample for it.

How to read this (limitations first):

- **PCST adds nothing over the greedy trail here.** The two columns are identical alert by alert (paired bootstrap difference 0.000, 95 % CI [0, 0]). The time-respecting pool around an alert is small (the trail views select 15.7 edges on average; only 141 of 636 alerts have a pool of 40 or more, and on those PCST and greedy take the same pattern edges, 58 each). The time-respecting constraint, not the Steiner tree, is what shapes the output. PCST is kept because it is cheap and it is the specified method; the honest claim for the report is "scorer + time-respecting trail", with PCST as a non-contributing ablation on HI-Small.
- **Trail vs amount-1hop:** +0.03 recall, paired bootstrap 95 % CI [0.00, 0.07] over 52 alerts. Not distinguishable from the stub's definition at this sample size.
- **Scorer only vs trail:** +0.06, CI [-0.10, 0.22]. The scorer-only 40 are never one connected trail (0 of 636); the trail views are always one.
- Ceiling: only 72 % of the pattern edges inside the window are time-respecting-reachable from the alerted edge (`relevant_reachable_share` 0.72 on `cal`+`eval`), so recall of 0.65 is near the best this construction can do; the rest are later or unrelated legs.
- Only 52 of 636 alerts (8 %) have any pattern edge inside the 72 h window, mean 3.7 such edges; most alerts are false alerts or have their attempt outside the window. All differences between trail-like views are within sampling noise.
- Mean candidates per alert 17,100 (median 12,750) because bank 70's view is large; one alert costs about 0.25 s end to end.

Raw output: `data/runs/evidence_views_hi_small_scorer_{eval,cal_eval}.{json,parquet}` (gitignored). The table above predates the long-prompt column; the rerun (week 7, `cal`+`eval`, same 636 alerts, 52 with pattern edges) gives the same numbers plus `long_prompt`: recall 0.003, 66 edges (3,970 tokens).

### Week 7: the views on HI-Medium and H6

HI-Medium val `cal`+`eval` (1,303 alerts, 372 in an attempt, 163 with pattern edges in the window; mean 19,800 candidate edges per alert), HI-Medium scorer, same definitions. Tokens are the estimate above (1.5 characters per token).

| view | pattern-edge recall in the selection | pattern edges (mean) | connected | one time-respecting trail | edges selected | prompt tokens (mean, est.) |
|---|---|---|---|---|---|---|
| top-40 by amount (`amount`) | 0.006 | 0.04 | 0.00 | 0.00 | 40.0 | 2,861 |
| top-40 by amount, 1 hop (`amount_1hop`, the stub) | 0.831 | 2.45 | 1.00 | 1.00 | 13.8 | 946 |
| scorer only | 0.447 | 1.04 | 0.00 | 0.00 | 40.0 | 2,685 |
| scorer + time-respecting greedy | 0.843 | 2.47 | 1.00 | 1.00 | 17.2 | 1,172 |
| **scorer + PCST** | **0.843** | 2.47 | 1.00 | 1.00 | 17.2 | 1,172 |
| everything in a long prompt (cap 4,000 tokens) | 0.012 | 0.01 | 0.00 | 0.00 | 57.6 | 3,967 |

HI-Small (`cal`+`eval`, 52 alerts with pattern edges): recall 0.006 / 0.621 / 0.591 / 0.651 / 0.651 / 0.003 in the same order, tokens 2,823 / 816 / 2,655 / 1,058 / 1,056 / 3,966. 89 % of the pattern edges inside the window are time-respecting-reachable from the alerted edge on HI-Medium (72 % on HI-Small), so 0.843 is close to the construction's ceiling.

**H6 as pre-registered** (`prereg/hypotheses.yaml`): statistic `pattern_edge_share(scorer_pcst) - pattern_edge_share(top40_by_amount)`, reject the null only if the share-difference CI lower bound is >= 0.15 **and** the Brier-difference CI upper bound is < 0. `pattern_edge_share` is the share of the view's edges that are pattern edges (precision, not recall); the baseline is the plain top-40 by amount (`amount`). Paired, 95 % percentile CI, 10,000 bootstrap resamples over attempts (`data/runs/h6_<ds>.json`; alert-level CIs are narrower and also in the json). "Among the 40" is ambiguous when the trail returns fewer than 40 edges, so both denominators are reported; **the verdict uses denominator 40** (a view is not rewarded for being short): **flag for T4 to confirm**.

| share of pattern edges in the view | scorer + PCST | top-40 by amount | difference [95 % CI over attempts] | lower bound >= 0.15 |
|---|---|---|---|---|
| HI-Small, alerts with pattern edges (52 alerts, 26 attempts), denominator 40 | 0.064 | 0.001 | **+0.064** [0.030, 0.104] | no |
| same, denominator = edges selected (15.7) | 0.165 | 0.001 | +0.164 [0.050, 0.353] | no |
| HI-Medium, alerts with pattern edges (163, 91 attempts), denominator 40 | 0.062 | 0.001 | **+0.061** [0.044, 0.079] | no |
| same, denominator = edges selected (17.2) | 0.163 | 0.001 | +0.162 [0.121, 0.208] | no |
| HI-Small, all 636 alerts, denominator 40 / selected | 0.005 / 0.014 | 0.0001 | +0.005 [0.002, 0.010] / +0.013 [0.003, 0.031] | no |
| HI-Medium, all 1,303 alerts, denominator 40 / selected | 0.008 / 0.020 | 0.0001 | +0.008 [0.005, 0.011] / +0.020 [0.014, 0.028] | no |

**Reading.** (1) The null of H6 cannot be rejected on the share half: with denominator 40 the difference is 0.06 on both datasets, and **0.15 is out of reach by construction on this data**: an alert has only 2.8 to 3.7 pattern edges in its window (HI-Medium 2.77, HI-Small 3.65), so even a perfect retrieval has a share of at most 0.07 to 0.09 among 40 edges. With the selected-edges denominator the point estimate is 0.16 on both datasets, but the lower bounds (0.05, 0.12) stay under 0.15. (2) The Brier half is **pending**: it needs planner runs with each view on the same val alerts (T3's E8: the same planner and fusion with `view(alert, "scorer_pcst")` and with `"amount"` as the evidence text, paired per alert, the CI of the Brier difference needs its upper bound below 0). (3) Secondary context, recall of the pattern edges in the window on the alerts that have any (`h6_<ds>.json`, `recall_secondary`): scorer + PCST minus the plain top-40 by amount +64.5 points on HI-Small, +83.7 on HI-Medium; minus the week-2 stub (top-40 by amount, 1 hop) only +3.0 [0.0, 6.9] and +1.2 [0.0, 3.1]: the time-respecting 1-hop neighbourhood already holds almost every reachable pattern edge and the scorer adds 1 to 3 points; minus scorer only +6.1 [-9.9, 22.8] and +39.7 [31.2, 47.8]; minus the greedy trail exactly 0 on both (PCST adds nothing). Decision memo: `docs/t1_planner_view.md`.

## Retrieval demo (G4, week 9)

```
python scripts/retrieval_demo.py --data hi_medium --alert V000993 [--view scorer_pcst] [--with-labels] [--full]
```

Prints the four views of E8 for one train or val alert (aliased accounts, time-ordered, nothing after the alert time; the long prompt shows its first and last three lines unless `--full`). `--with-labels` (off by default) also scores each view against the pattern file: the pattern edges of the alert's attempt inside the window, how many the view holds, and the one-time-respecting-trail check; labels are never printed next to an edge. Three HI-Medium val alerts (copied from the script; accounts are the seed-0 aliases, banks are public ids):

**V000993**: an attempt with 4 pattern edges in the window of 12,860 candidate edges. The top-40 by amount holds none of them (40 edges, about 2,900 tokens: the bank's largest payments); the scorer alone holds 3 of 4 but not as one trail; the scorer + PCST trail holds all 4 in 9 edges (about 590 tokens): one account paying itself and one counterparty at each step, ending at the alerted payment.

```
=== scorer_pcst: 9 of 12860 candidate edges in the 72 h window, about 585 tokens ===
2022-09-08 14:24 | A011a5a3 -> A011a5a3 | 15,404.69 US Dollar | ACH | bank 1601 -> bank 1601
2022-09-08 14:24 | A011a5a3 -> A01ca922 | 13,146.36 Euro | ACH | bank 1601 -> bank 20538 (outgoing)
2022-09-09 08:31 | A011a5a3 -> A011a5a3 | 16,526.98 US Dollar | ACH | bank 1601 -> bank 1601
2022-09-09 08:31 | A011a5a3 -> A01e6d48 | 14,104.12 Euro | ACH | bank 1601 -> bank 4424 (outgoing)
2022-09-09 15:57 | A011a5a3 -> A0155bf7 | 14,467.94 US Dollar | ACH | bank 1601 -> bank 16809 (outgoing)
2022-09-10 12:10 | A011a5a3 -> A011a5a3 | 1,311.86 US Dollar | ACH | bank 1601 -> bank 1601
2022-09-10 12:10 | A011a5a3 -> A00bb4a0 | 8,786.34 Yuan | ACH | bank 1601 -> bank 25 (outgoing)
2022-09-11 13:09 | A011a5a3 -> A011a5a3 | 3,033.90 US Dollar | ACH | bank 1601 -> bank 1601
2022-09-11 13:09 | A011a5a3 -> A01c7c26 | 2,589.13 Euro | ACH | bank 1601 -> bank 21514 (outgoing) <- ALERT
```

The long prompt of the same alert holds only the newest 57 of the 12,860 edges (about 3,700 real tokens, the last half hour of the bank's traffic) and none of the 4 pattern edges:

```
=== long_prompt: 57 of 12860 candidate edges in the 72 h window, about 3971 tokens ===
2022-09-11 12:46 | A001111e -> A015fe37 | 927.00 US Dollar | Cheque | bank 292109 -> bank 1601 (incoming)
2022-09-11 12:47 | A005e16a -> A0199f15 | 305.18 US Dollar | Credit Card | bank 1601 -> bank 11 (outgoing)
2022-09-11 12:47 | A01f2e79 -> A01f378e | 257.30 US Dollar | Credit Card | bank 1601 -> bank 7770 (outgoing)
... (51 more edges, the newest 57 of 12860 fit the cap) ...
2022-09-11 13:09 | A01eb61c -> A01be450 | 67.12 US Dollar | Cheque | bank 114976 -> bank 1601 (incoming)
2022-09-11 13:09 | A0122be2 -> A0122be2 | 56.72 US Dollar | ACH | bank 1601 -> bank 1601
2022-09-11 13:09 | A0122be2 -> A0160322 | 48.41 Euro | Wire | bank 1601 -> bank 142848 (outgoing)
```

**V000940**: 3 pattern edges among 31,550 candidates; the top-40 by amount and the scorer alone hold none of them, scorer + PCST holds all 3 inside a 40-edge trail (about 2,700 tokens). **V000935**: a false alert with no pattern edge; the trail is 2 edges (about 130 tokens) against 40 large unrelated payments (about 2,800 tokens): "nothing connected to this payment in the window" is also information. Rerun the command for the full output.

## Files

- `edge_scorer.py`: features (`edge_features`, `timeline_features`), `train_scorer` (M3), a tiny sklearn baseline.
- `pcst.py`: `reachable_to_alert`, `greedy_trail`, `pcst_edges`, `select_pcst_trail`, `is_one_trail`, `is_connected`, `check_binding`. Pure numpy; `pcst_fast` is imported inside the function.
- `evidence_views.py`: `view` (the stable API), `candidates` (time-guarded read), `select_views`, `long_prompt_mask`, `line_chars`, `format_trail`, `evaluate`, `h6_report`.
- `views.py`: T3's week-2 stub; T1 owns it from week 4 and added `direction_keys` and the `alert_summary` extras (`direction`, `counterparty_bank`, `top_features`).
- `pcst_fast_binding.patch`: the one-line fix for the pcst_fast wheel.
