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

`name` is one of `amount` (top-40 by amount over the whole 72 h window), `scorer` (top-40 by edge-scorer score), `scorer_pcst` (the scorer's edges as a time-respecting PCST trail connected to the alerted edge, at most k), `long_prompt` (all candidate edges, newest kept until the evidence token cap), and the ablations `amount_1hop` (**T3's week-2 stub: the 40 largest edges that touch the alerted sender or receiver**, which is what the planner is trained on today) and `scorer_greedy`. Guarantees, each with a test (`tests/test_evidence_api.py`): candidates come from the home bank's view through `ftm.data.io` (time guard, train and val only); nothing after the alert time is read, scored or returned (`assert_not_future`); accounts are aliased with `alias_seed` (the run seed, default 0); the text is built by `views.format_trail`, which refuses raw ids and labels; an `AlertContext` whose `as_of` is not the alert's time raises. `data` is the dataset name (an alert id is only unique within one). Cost: about 0.25 s per alert on HI-Small and 0.7 s on HI-Medium on one CPU core (the candidate read dominates; the views table below took 228 s for 636 and 897 s for 1,303 alerts, all six views each), plus the one-off load of `data/retrieval/<ds>/edge_scores.parquet`.

**The long-prompt view and its cap.** "Everything in a long prompt" = every candidate edge of the window serialised like the other views, **cut at 4,000 estimated tokens** (`EVIDENCE_TOKENS`): the harness prompt cap is `Limits.max_tokens` = 6,000 (`configs/llm/default.yaml`; the server context is 8,192), and T3's planner prompt (instructions, alert, question menu, answer history) uses about 2,000 of them. Tokens are estimated at 1.7 characters per token (`CHARS_PER_TOKEN`; see `docs/t1_planner_view.md` for the check against the Qwen3 tokenizer). When the window does not fit the **newest** edges are kept, the alerted edge always: with 12,000 to 35,000 candidate edges per alert the cap holds about 65 of them (0.3 % of the window). `max_evidence_tokens=` changes the cap; the estimate is exact for the line length (a test compares it with the formatted lines).

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

HI-Medium val `cal`+`eval` (1,303 alerts, 372 in an attempt, 163 with pattern edges in the window; mean 19,800 candidate edges per alert), HI-Medium scorer, same definitions. Tokens are the estimate above.

| view | pattern-edge recall in the selection | pattern edges (mean) | connected | one time-respecting trail | edges selected | prompt tokens (mean) |
|---|---|---|---|---|---|---|
| top-40 by amount (`amount`) | 0.006 | 0.04 | 0.00 | 0.00 | 40.0 | 2,524 |
| top-40 by amount, 1 hop (`amount_1hop`, the stub) | 0.831 | 2.45 | 1.00 | 1.00 | 13.8 | 835 |
| scorer only | 0.447 | 1.04 | 0.00 | 0.00 | 40.0 | 2,369 |
| scorer + time-respecting greedy | 0.843 | 2.47 | 1.00 | 1.00 | 17.2 | 1,034 |
| **scorer + PCST** | **0.843** | 2.47 | 1.00 | 1.00 | 17.2 | 1,034 |
| everything in a long prompt (cap 4,000 tokens) | 0.012 | 0.01 | 0.00 | 0.00 | 65.3 | 3,969 |

HI-Small (`cal`+`eval`, 52 alerts with pattern edges): 0.006 / 0.621 / 0.591 / 0.651 / 0.651 / 0.003 in the same order, tokens 2,491 / 720 / 2,343 / 934 / 932 / 3,970. 89 % of the pattern edges inside the window are time-respecting-reachable from the alerted edge on HI-Medium (72 % on HI-Small), so 0.843 is close to the construction's ceiling.

**H6 (at least 15 points more real pattern edges than the top-40-by-amount baseline), paired over the alerts with pattern edges, 95 % percentile CI from 10,000 bootstrap resamples over alerts and over attempts** (`data/runs/h6_<ds>.json`; positive = scorer + PCST better):

| comparison (recall, in points) | HI-Small (52 alerts, 26 attempts) | HI-Medium (163 alerts, 91 attempts) |
|---|---|---|
| scorer + PCST minus plain top-40 by amount | **+64.5** [52.5, 76.1] alerts, [43.7, 88.1] attempts: met | **+83.7** [78.1, 88.7], [73.8, 92.9]: met |
| scorer + PCST minus the stub (top-40 by amount, 1 hop) | **+3.0** [0.0, 6.9], [0.0, 10.2]: **not met** | **+1.2** [0.0, 3.1], [0.0, 3.2]: **not met** |
| scorer + PCST minus scorer only | +6.1 [-9.9, 22.8], [-17.2, 35.3] | +39.7 [31.2, 47.8], [26.7, 52.3] |
| scorer + PCST minus scorer + greedy trail | 0.0 [0, 0] | 0.0 [0, 0] |
| scorer + PCST minus the long prompt | +64.9 | +83.1 |

H6 is met against the plain top-40 by amount (which finds almost no pattern edges, 0.6 %) and **not met against the stub**, the baseline the planner actually trains on: the time-respecting 1-hop neighbourhood of the alerted edge already holds almost every reachable pattern edge, and the scorer adds 1 to 3 points on top. In the stratum where the stub's pool has at least 40 edges (so that the choice matters; 19 alerts on HI-Small, 21 on HI-Medium) the difference is +8.3 [0.0, 17.9] and +4.8 [0.0, 14.3] points, still under the margin. PCST adds nothing over the greedy time-respecting trail on either dataset (identical alert by alert). Decision memo: `docs/t1_planner_view.md`.

## Files

- `edge_scorer.py`: features (`edge_features`, `timeline_features`), `train_scorer` (M3), a tiny sklearn baseline.
- `pcst.py`: `reachable_to_alert`, `greedy_trail`, `pcst_edges`, `select_pcst_trail`, `is_one_trail`, `is_connected`, `check_binding`. Pure numpy; `pcst_fast` is imported inside the function.
- `evidence_views.py`: `candidates` (time-guarded read), `select_views`, `format_trail`, `evaluate`.
- `pcst_fast_binding.patch`: the one-line fix for the pcst_fast wheel.
