# ftm/retrieval: edge scorer, time-respecting PCST, evidence views (C11, hand-off #11)

Owner: T1 (Abhishek). Spec: `plan.md` 6.12, `tracks.md` 3.8. T3's week-2 stub (`views.py`, PR #6) is untouched; this code lives in `edge_scorer.py`, `pcst.py`, `evidence_views.py`.

```
python3.12 -m ftm.retrieval.edge_scorer --data hi_small --train           # M3: trains on train, selects on val fit, writes models/ + data/retrieval/
python3.12 -m ftm.retrieval.evidence_views --data hi_small --parts cal eval   # M4: the views table below (~3 min, CPU)
```

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

Raw output: `data/runs/evidence_views_hi_small_scorer_{eval,cal_eval}.{json,parquet}` (gitignored).

## Files

- `edge_scorer.py`: features (`edge_features`, `timeline_features`), `train_scorer` (M3), a tiny sklearn baseline.
- `pcst.py`: `reachable_to_alert`, `greedy_trail`, `pcst_edges`, `select_pcst_trail`, `is_one_trail`, `is_connected`, `check_binding`. Pure numpy; `pcst_fast` is imported inside the function.
- `evidence_views.py`: `candidates` (time-guarded read), `select_views`, `format_trail`, `evaluate`.
- `pcst_fast_binding.patch`: the one-line fix for the pcst_fast wheel.
