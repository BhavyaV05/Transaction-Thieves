# ftm/baselines: GFP+LightGBM p0 scores (hand-off #5 and #17), reproduction checks

Owner: T1 (Abhishek). Spec: `plan.md` 6.1-6.2, `tracks.md` 3.4-3.5. Everything here is rebuilt by:

```
PYTHONPATH=. python scripts/train_gfp.py --repro                                  # E1/G1 reproduction (raw CSV, paper split)
python scripts/train_gfp.py --data hi_small --mode one_bank                       # 30 bank views, ~9 min
python scripts/train_gfp.py --data hi_small --mode see_everything --threads 6     # ~17 min, 4.7 GB RSS
python scripts/train_gfp.py --data hi_small --report                              # one-bank vs see-everything table
python scripts/train_gfp.py --data hi_medium --mode one_bank --neg-frac 0.3       # HI-Medium
python scripts/train_gfp.py --data hi_medium --mode see_everything --neg-frac 0.1 # needs >= 10 GB free RAM
```

## What is in the p0 table (hand-off #5 one-bank, #17 see-everything)

`data/scores/<ds>/p0.parquet` (gitignored; also `p0_one_bank.parquet`, `p0_see_everything.parquet`). Train and val rows only: **there are no test scores** (`read_p0(..., split="test")` raises `TestSplitLocked`).

| column | type | note |
|---|---|---|
| tx_id | int64 | as in `transactions.parquet` |
| bank | int64, null for see-everything | one-bank: a transaction between two top-30 banks appears twice (once per bank, each scored by that bank's own model) |
| mode | `one_bank` / `see_everything` | |
| model | `gfp_lgbm` | |
| split | `train` / `val` | |
| val_part | `fit` / `cal` / `eval`, null for train | first half of the val time span / next quarter / last quarter (cut by time of day, same for every bank). `fit` = early stopping and the calibrator were fit here (**in-sample**); `cal` = conformal calibration part (not used for the calibrator); `eval` = every reported val metric |
| p0 | float64 | calibrated P(laundering). Null where the bank was skipped (see below) |
| score_raw | float64 | LightGBM output before calibration |
| label | bool | `is_laundering` (train/val only, so no test label is exposed) |
| seed, commit | | seed 0; git short hash of the run (also in `data/runs/p0_<ds>_<mode>.json`) |

**Train scores are out-of-fold**: the train period (by time) is cut into 3 contiguous blocks and each block is scored by a model fit on the other two (with the hyper-parameters chosen on val), then mapped through the same calibrator as val. This uses later train days to score earlier ones (no val or test data), so OOF scores are slightly more optimistic than a forward-only split but are not in-sample. OOF quality is lower than val quality for one-bank (PR-AUC 0.05 train OOF vs 0.20 val on HI-Small: the OOF models see only 2/3 of an already short train period, and train positives are fewer). T2 back-testing should use `split == "val"` and `val_part in {cal, eval}`; `fit` rows are in-sample for the calibrator.

Load with `ftm.baselines.gfp_gbt.read_p0(ds, mode=..., split=...)`. **Status:** `hi_small/p0.parquet` has both modes; `hi_medium` has only `p0_one_bank.parquet` (see-everything is blocked, below), so `read_p0("hi_medium")` needs `root`/the one-bank file directly until then.

## Features, and the leakage rule

`GraphFeaturePreprocessor` (snapml 1.17.2) with the settings of the GFP paper (Blanusa et al., KDD 2024, arXiv:2402.08593, section 3): one-day window for vertex statistics, fan, degree, temporal-cycle and simple-cycle features, six hours for scatter-gather, cycle length constraint 10, vertex statistics on Timestamp and Amount. Bins (2, 3) are the library example's; the paper does not list them. Plus 7 basic features (amounts, three category codes, same-bank flag, hour of day). Account ids are not features. One-bank models also get `out` (does the bank send or receive the transaction).

**Window semantics (checked on a toy graph and in `tests/test_baselines_gfp.py`):** edges of earlier `transform` calls are history (past only); edges inside **one** call are inserted together, so an edge also sees the other edges of its own batch, including later ones. We therefore feed time-sorted batches of 128 edges (the paper's inference batch size) and never put two splits in one batch. An edge sees all earlier batches plus **at most 127 later edges of the same split**; train features never touch val, val features use train history (earlier split), test is never passed in (`build_features` raises on a test row without `FTM_FINAL_RUN=1`). The 127-edge look-ahead is the same inside train and inside val, so there is no train/val feature shift; it is a limitation, not cross-split leakage. GFP columns that depend on absolute time (mean and sum of the raw timestamp) are dropped (found empirically: they change when all timestamps are shifted), leaving 48 of 66 GFP columns, 55 features in total.

## Reproduction (M1, E1/G1): paper split, raw HI-Small CSV, reproduction only

`scripts/train_gfp.py --repro` runs on the **published** 60/20/20 split by transaction order (it overlaps our test days, so it is gated behind `--repro`, reads the raw CSV, writes nothing to `data/scores` or `data/alerts`, and nothing from it feeds any later model or decision). Hyper-parameters are chosen by validation AP from a tiny fixed grid (the paper's tuned values are not published); the decision threshold is the best-F1 threshold on the paper's val part.

| model (paper split, HI-Small) | minority-class F1 | precision | recall | PR-AUC | published (batch 128) |
|---|---|---|---|---|---|
| GFP + LightGBM | **62.95** | 0.785 | 0.525 | 0.603 | 62.86 +- 0.25 |
| GFP + XGBoost | **62.51** | 0.800 | 0.513 | 0.591 | **63.23 +- 0.17** |

Reproduced within 0.8 points. The 63.23 quoted in tracks.md is the **XGBoost** row (AMLworld paper, arXiv:2306.16424, Table 2 and appendix D for the GFP configuration; the same numbers are in Table 4 of the GFP paper); the GFP+LightGBM row is 62.86. We ran both. Runtime: features 518 s (5.08 M transactions, 8-10 k rows/s, 4-6 threads), LightGBM grid 462 s, XGBoost grid 359 s. At the fixed 0.5 threshold F1 is 51.4 / 50.9 (the val-tuned threshold matters). Json: `data/runs/gfp_repro_hi_small.json` (commit `b26e990`, seed 0).

Two things that cost F1 before they were fixed, so reviewers know where to look: (1) with LightGBM's default `min_child_samples=20, reg_lambda=0` the validation AP *falls* from 0.30 (1 tree) to 0.08 (100 trees); early stopping on the logloss (the wrapper's default metric next to `average_precision`) then stopped after one tree and gave 50.9 F1. `LGBM_GRID` now uses `min_child_samples=100, reg_lambda=10` and `metric="average_precision"` only. (2) Float32 overflow in kurtosis features (inf) is clipped at 1e30.

## Our split: one-bank vs see-everything (M3), HI-Small

Splits: train days 1-6, val 7-8 (`configs/data/hi_small.yaml`). LightGBM, 4-point grid (num_leaves 31 or 127 x scale_pos_weight 1 or 8, lr 0.05, min_child_samples 100, lambda 10, early stopping on val `fit` AP, colsample/subsample 0.8), logged in `data/runs/p0_<ds>_<mode>.json`. Calibration is fit on the val `fit` part; all numbers below are on val `eval` (the last quarter of the val time span, 93,240 bank-view rows with 122 laundering rows one-bank; the same rows scored by the whole-graph model for the comparison). **Isotonic is used only where the `fit` part has >= 30 laundering rows AND the out-of-fold train AP is at least 3x the train prevalence (`SIGNAL_LIFT`); otherwise a two-parameter Platt scaling.** On HI-Small that is Platt for all 28 scored one-bank models (bank 70 had enough positives but no signal) and isotonic for see-everything. An isotonic step function fit on a handful of positives, or on a score with no signal, puts `p0 = 1` on its top plateau: HI-Medium bank 70 did that (170,000 false alerts) before the rule was added. `python scripts/train_gfp.py --data <ds> --recalibrate` refits the calibrators from the stored raw scores without refitting a model. Two banks (19, 1674) have fewer than 5 train positives and are skipped (`p0` null).

**Headline (pooled over the 28 bank views, val eval part):**

| | Brier | Brier skill vs constant | PR-AUC | F1 at the best `fit`-part threshold |
|---|---|---|---|---|
| constant (train/val-fit prevalence) | 0.001307 | 0 | | |
| one-bank (30 views, own edges only) | 0.001167 | 0.107 | 0.202 | 0.283 |
| **see-everything** (whole graph) | **0.000906** | **0.307** | **0.419** | **0.474** |

Paired bootstrap over rows of (one-bank Brier - see-everything Brier) = 0.000261, 95% CI [0.000139, 0.000386]: the gap is real and large relative to the headroom (see-everything removes 2.9x the Brier skill of one-bank). **The see-everything model is not barely better than one-bank: the gap, which is the project's denominator, exists on HI-Small.** The caveat is statistical power: 122 laundering rows in the eval part, 0-28 per bank (the per-bank table below is noisy; read the pooled row).

Two things to know: bank 70 (the largest, 28 eval positives) is at base rate under both models (PR-AUC 0.001 vs prevalence 0.001); this is a hub with 450 k transactions and no pattern the GFP features pick up, and it dominates the pooled Brier weight but not the PR-AUC. The macro-average Brier over banks is 0.001178 (one-bank) vs 0.000834 (see-everything). Per-bank table: `data/runs/gap_hi_small_per_bank.md` (copied below).

| bank | laundering (train+val) | eval rows | eval pos | Brier const | Brier one-bank | Brier see-all | PR-AUC one-bank | PR-AUC see-all | F1 one-bank | F1 see-all |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 59 | 4504 | 3 | 0.666e-3 | 0.569e-3 | 0.628e-3 | 0.3650 | 0.1963 | 0.4000 | 0.0000 |
| 3 | 16 | 2807 | 0 | 0.000e-3 | 0.011e-3 | 0.147e-3 | n/a | n/a | n/a | n/a |
| 4 | 26 | 1730 | 2 | 1.155e-3 | 1.184e-3 | 0.640e-3 | 0.0335 | 0.5833 | 0.0690 | 0.8000 |
| 6 | 8 | 1791 | 0 | 0.000e-3 | 0.000e-3 | 0.229e-3 | n/a | n/a | n/a | n/a |
| 7 | 18 | 2538 | 2 | 0.788e-3 | 0.787e-3 | 0.394e-3 | 0.2750 | 0.5625 | 0.5000 | 0.6667 |
| 10 | 69 | 5846 | 7 | 1.196e-3 | 1.150e-3 | 0.469e-3 | 0.2917 | 0.7536 | 0.4000 | 0.4444 |
| 11 | 56 | 2051 | 6 | 2.918e-3 | 2.846e-3 | 2.265e-3 | 0.3702 | 0.4663 | 0.2500 | 0.2857 |
| 12 | 109 | 5848 | 11 | 1.878e-3 | 1.762e-3 | 1.155e-3 | 0.1873 | 0.5278 | 0.1111 | 0.5882 |
| 14 | 15 | 1809 | 0 | 0.001e-3 | 0.136e-3 | 0.006e-3 | n/a | n/a | n/a | n/a |
| 15 | 43 | 3939 | 4 | 1.015e-3 | 1.014e-3 | 1.113e-3 | 0.0428 | 0.0535 | 0.1053 | 0.0000 |
| 20 | 67 | 3007 | 10 | 3.318e-3 | 2.081e-3 | 2.410e-3 | 0.5932 | 0.5930 | 0.4615 | 0.5556 |
| 22 | 45 | 2128 | 5 | 2.344e-3 | 1.717e-3 | 1.897e-3 | 0.5648 | 0.5594 | 0.3333 | 0.5455 |
| 23 | 41 | 2022 | 7 | 3.454e-3 | 2.824e-3 | 2.239e-3 | 0.4799 | 0.6114 | 0.4000 | 0.6667 |
| 28 | 23 | 2318 | 2 | 0.862e-3 | 0.742e-3 | 0.063e-3 | 0.4167 | 1.0000 | 0.6667 | 0.6667 |
| 70 | 523 | 24916 | 28 | 1.123e-3 | 1.123e-3 | 1.123e-3 | 0.0015 | 0.0015 | 0.0000 | 0.0000 |
| 116 | 32 | 2283 | 2 | 0.875e-3 | 0.937e-3 | 0.860e-3 | 0.0606 | 0.0611 | 0.0000 | 0.0000 |
| 118 | 21 | 1853 | 0 | 0.001e-3 | 0.001e-3 | 0.179e-3 | n/a | n/a | n/a | n/a |
| 119 | 86 | 1584 | 10 | 6.276e-3 | 3.038e-3 | 2.118e-3 | 0.7808 | 0.8111 | 0.7826 | 0.8000 |
| 211 | 48 | 2319 | 1 | 0.432e-3 | 0.979e-3 | 0.009e-3 | 0.0385 | 1.0000 | 0.0000 | 1.0000 |
| 213 | 26 | 1578 | 0 | 0.001e-3 | 0.043e-3 | 0.005e-3 | n/a | n/a | n/a | n/a |
| 214 | 22 | 1677 | 1 | 0.596e-3 | 0.636e-3 | 0.572e-3 | 0.0833 | 0.2500 | 0.0000 | 0.0000 |
| 220 | 29 | 3813 | 4 | 1.048e-3 | 1.064e-3 | 0.775e-3 | 0.0250 | 0.3541 | 0.0000 | 0.2857 |
| 410 | 40 | 1708 | 5 | 2.920e-3 | 1.992e-3 | 1.004e-3 | 0.7417 | 0.8385 | 0.5000 | 0.7500 |
| 1292 | 19 | 2032 | 3 | 1.474e-3 | 1.449e-3 | 0.029e-3 | 0.1058 | 1.0000 | 0.0000 | 1.0000 |
| 1362 | 30 | 1819 | 0 | 0.001e-3 | 0.024e-3 | 0.471e-3 | n/a | n/a | n/a | n/a |
| 1420 | 31 | 1618 | 1 | 0.618e-3 | 0.617e-3 | 0.023e-3 | 0.0667 | 1.0000 | 0.0000 | 0.1053 |
| 1588 | 21 | 1438 | 3 | 2.083e-3 | 2.073e-3 | 2.059e-3 | 0.0605 | 0.0699 | 0.0000 | 0.0000 |
| 1665 | 29 | 2264 | 5 | 2.207e-3 | 2.197e-3 | 0.462e-3 | 0.1365 | 0.8208 | 0.2500 | 0.3333 |


## HI-Medium

Splits: train days 1-9, val 10-12. **One-bank: done for all 30 bank views**, `neg_frac 0.3` (all laundering rows plus 30% of clean train rows for fitting; the calibrator on val removes the bias), 6 isotonic and 24 Platt calibrators. Wall time: the 29 smaller banks took 200 s in total; **bank 70 alone took 7,213 s**, almost all of it GFP (a hub view where the streaming graph feature extraction slows to a few hundred rows/s; 2.09 M train+val rows). Bank 70 has 2,942 laundering transactions in train+val and is **at base rate under the model** (PR-AUC 0.0015 vs prevalence 0.0015), so a quarter of the HI-Medium one-bank laundering is invisible to GFP features on that view.

**One-bank on val eval part, pooled over 30 views (320,372 rows, 527 laundering):** Brier 0.001451 vs constant 0.001644 (skill 0.117), PR-AUC 0.233, F1 at the best `fit`-part threshold 0.350; macro-average Brier over banks 0.001370. Per bank (the see-everything columns are n/a, see below):

| bank | laundering (train+val) | eval rows | eval pos | Brier const | Brier one-bank | Brier see-all | PR-AUC one-bank | PR-AUC see-all | F1 one-bank | F1 see-all |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 323 | 10131 | 28 | 2.760e-3 | 2.377e-3 | n/a | 0.3530 | n/a | 0.4364 | n/a |
| 3 | 166 | 5540 | 14 | 2.531e-3 | 1.838e-3 | n/a | 0.4695 | n/a | 0.4211 | n/a |
| 4 | 91 | 5826 | 7 | 1.201e-3 | 1.009e-3 | n/a | 0.4381 | n/a | 0.5000 | n/a |
| 6 | 178 | 5344 | 15 | 2.823e-3 | 2.432e-3 | n/a | 0.4274 | n/a | 0.3774 | n/a |
| 11 | 314 | 10919 | 25 | 2.295e-3 | 1.542e-3 | n/a | 0.5185 | n/a | 0.5172 | n/a |
| 12 | 247 | 10928 | 22 | 2.018e-3 | 1.620e-3 | n/a | 0.3642 | n/a | 0.4815 | n/a |
| 14 | 153 | 6282 | 18 | 2.857e-3 | 1.435e-3 | n/a | 0.7023 | n/a | 0.5714 | n/a |
| 15 | 101 | 6026 | 9 | 1.495e-3 | 1.518e-3 | n/a | 0.5121 | n/a | 0.2400 | n/a |
| 20 | 368 | 9728 | 25 | 2.578e-3 | 1.991e-3 | n/a | 0.5995 | n/a | 0.6296 | n/a |
| 25 | 112 | 5763 | 11 | 1.913e-3 | 1.562e-3 | n/a | 0.4950 | n/a | 0.3429 | n/a |
| 26 | 57 | 3872 | 6 | 1.547e-3 | 1.411e-3 | n/a | 0.2285 | n/a | 0.1429 | n/a |
| 27 | 156 | 10301 | 14 | 1.361e-3 | 2.055e-3 | n/a | 0.1170 | n/a | 0.2424 | n/a |
| 28 | 31 | 4189 | 5 | 1.193e-3 | 1.110e-3 | n/a | 0.4579 | n/a | 0.2308 | n/a |
| 29 | 92 | 6297 | 11 | 1.744e-3 | 1.367e-3 | n/a | 0.4803 | n/a | 0.0000 | n/a |
| 70 | 2942 | 149662 | 218 | 1.454e-3 | 1.455e-3 | n/a | 0.0015 | n/a | 0.0000 | n/a |
| 111 | 51 | 4910 | 4 | 0.814e-3 | 0.783e-3 | n/a | 0.1063 | n/a | 0.0000 | n/a |
| 112 | 33 | 6690 | 4 | 0.598e-3 | 0.597e-3 | n/a | 0.0214 | n/a | 0.0299 | n/a |
| 113 | 20 | 4814 | 1 | 0.208e-3 | 0.250e-3 | n/a | 0.0038 | n/a | 0.0000 | n/a |
| 114 | 27 | 5627 | 1 | 0.179e-3 | 0.240e-3 | n/a | 0.0101 | n/a | 0.0000 | n/a |
| 210 | 25 | 4294 | 3 | 0.701e-3 | 0.734e-3 | n/a | 0.0201 | n/a | 0.0290 | n/a |
| 214 | 134 | 4892 | 12 | 2.449e-3 | 1.724e-3 | n/a | 0.5279 | n/a | 0.5000 | n/a |
| 544 | 108 | 4359 | 10 | 2.298e-3 | 1.643e-3 | n/a | 0.6150 | n/a | 0.5217 | n/a |
| 741 | 123 | 4103 | 11 | 2.680e-3 | 1.506e-3 | n/a | 0.6575 | n/a | 0.3750 | n/a |
| 1046 | 73 | 3890 | 6 | 1.547e-3 | 1.731e-3 | n/a | 0.4781 | n/a | 0.3636 | n/a |
| 1208 | 148 | 4780 | 3 | 0.650e-3 | 0.631e-3 | n/a | 0.4802 | n/a | 0.3333 | n/a |
| 1601 | 112 | 3950 | 5 | 1.281e-3 | 0.908e-3 | n/a | 0.4897 | n/a | 0.5000 | n/a |
| 1818 | 89 | 3983 | 4 | 1.006e-3 | 1.029e-3 | n/a | 0.3283 | n/a | 0.1667 | n/a |
| 2310 | 175 | 4954 | 20 | 4.025e-3 | 1.927e-3 | n/a | 0.7266 | n/a | 0.6939 | n/a |
| 2597 | 114 | 4449 | 6 | 1.366e-3 | 0.587e-3 | n/a | 0.8774 | n/a | 0.7059 | n/a |
| 2776 | 98 | 3869 | 9 | 2.328e-3 | 2.102e-3 | n/a | 0.5523 | n/a | 0.4348 | n/a |


**See-everything on HI-Medium: BLOCKED-COMPUTE.** The job (GFP over the full train+val graph, 23.1 M rows, then LightGBM with `--neg-frac 0.1`) fits in RAM (about 10 GB peak RSS with the Int16-encoded frame) but had not finished after more than 2 hours of wall time on the laptop (4-6 busy threads; HI-Small's 4.2 M rows took 17 minutes in total) and was stopped at the end of the session, so there is no `p0_see_everything.parquet` for HI-Medium and no HI-Medium one-bank vs see-everything gap yet. Ready to run on the 5070 Ti machine (needs >= 12 GB free RAM and many CPU threads; GFP is CPU-bound; expect several hours):

```
python scripts/train_gfp.py --data hi_medium --mode see_everything --neg-frac 0.1 --threads 16
python scripts/train_gfp.py --data hi_medium --recalibrate     # refit calibrators with the signal rule
python scripts/train_gfp.py --data hi_medium --report          # writes data/runs/gap_hi_medium_*.{md,csv,json}
```

Because of that, the "gap" (the project's denominator) is only measured on HI-Small (see above), where it is large.



## Multi-GNN reproduction (M2)

`scripts/train_gnn.py --repro` (launcher in `multi_gnn.py`) runs IBM's Multi-GNN, Apache-2.0, kept **outside the repo** in `~/Documents/t1_external/Multi-GNN` at commit `252b0252afca109d1d216c411c59ff70753b25fc`. **Reproduction only**: it uses the code's own 60/20/20 split by transaction index (overlaps our test days), logs test F1 every epoch, writes nothing to `data/scores`/`data/alerts`, and feeds nothing downstream.

- Data: `format_kaggle_files.py` (needs `datatable`) on `HI-Small_Trans.csv` -> `data/multignn/Small_HI/formatted_transactions.csv`.
- Environment: python 3.12, torch 2.5.1+cu121, PyG 2.7. Extra installs: `torch_sparse` 0.6.18 and `pyg_lib` 0.4.0 (wheels from data.pyg.org for torch 2.5.0+cu121, `--no-deps`; needed by `LinkNeighborLoader`), `datatable`, `wandb` (disabled with `--testing`). `torch_scatter` was already there; torch itself untouched.
- Patch (`multi_gnn_t1.patch`, 93 lines): (1) `AddEgoIds.__call__` -> `forward` (PyG 2.7's `BaseTransform` is abstract); (2) time budget (`T1_BUDGET_S`) and a per-epoch metrics line (`T1_METRICS`) in `training.py`; (3) local paths in `data_config.json`.
- Model: Multi-PNA with edge updates, reverse message passing, ego IDs, port numbering (`--model pna --emlps --reverse_mp --ego --ports`), the repo's `model_settings.json` PNA parameters (2 GNN layers, 20 hidden, lr 6.1e-4), seed 1. **Deviations from the paper's runs:** batch size 2048 and neighbour sampling [100, 100] (batch 8192 ran out of memory on the 8 GB GPU), and only **5 of 100 epochs** because of the 2.5 h time box (epochs took 21-26 min on the 3070 Ti laptop with the CPU shared with other jobs), one seed. 8,932 s in total.

| epoch | train F1 | val F1 | test F1 (own split) |
|---|---|---|---|
| 1 | 0.1 | 1.9 | 4.8 |
| 2 | 26.9 | 29.1 | 36.5 |
| 3 | 40.9 | 47.6 | 57.2 |
| 4 | 47.2 | 55.8 | 65.4 |
| **5** | 48.9 | **60.1** | **67.6** |
| 6 | 49.1 | 56.9 | 65.5 |

Model selection by best val F1 (epoch 5): **test F1 67.64 vs the paper's Multi-PNA+EU 68.16 +- 2.65 on AML Small HI** (Table 2 of arXiv:2306.11586; Multi-GIN+EU is 64.79; Multi-PNA without EU 64.59). Reproduced (inside one paper standard deviation) even though it is a short, single-seed run; the curve was still rising at epoch 5 on val and fell at epoch 6, so longer training or more seeds could move it a point or two either way. Metrics: `data/multignn/pna_metrics.jsonl`, log `data/multignn/pna_train.log` (gitignored). Our-split Multi-GNN (one-bank, see-everything) is in the next section.


## Multi-GNN on OUR split (M2, #17): one-bank vs see-everything, HI-Small

```
python scripts/train_gnn.py --our-split --data hi_small --mode see_everything --budget-s 4200 [--batch-size 1024] [--resume]   # GPU, 7.1 GB at batch 2048
python scripts/train_gnn.py --our-split --data hi_small --mode one_bank --max-epochs 100 --budget-s 7200                       # all bank views, ~100 min
python scripts/train_gnn.py --compare --data hi_small                                                                           # table below -> data/runs/multi_gnn_vs_gfp_hi_small.json
```

Multi-PNA+EU (the reproduced config: edge updates, reverse message passing, ego ids, port numbering; `model_settings.json` PNA parameters, seed 1), batch 2048, neighbours [100, 100] (the reproduction config; VRAM was free enough, no change was needed). Input is built from our Parquet (`ftm.data.io`, train and val only, our split; node ids from the dense ids of `src_id`/`dst_id`, kept as **int64**; the four edge features of IBM's `format_kaggle_files.py`: seconds since the first row, amount received, receiving currency, payment format). The train graph holds train edges only, validation and scoring use the train+val graph, no test edge is ever read (`edge_arrays` raises on a test row). Output goes to **`data/scores/hi_small/p0_multi_gnn.parquet`** (GFP's files are untouched), same schema as the p0 table above with `model = multi_gnn` and **val rows only** (no train rows: the GNN has no out-of-fold train scores). Calibration is the GFP protocol: the val `fit` part fits the calibrator (isotonic only with >= 30 positives and signal, else Platt; a bank view with no positive in the `fit` part gets `p0 = null`, as GFP skips such views), `cal` is left for conformal, `eval` is for metrics. The epoch is picked by a weighted AP on the `fit` part (all positives + 10 % of negatives). `p0` is also null for the few val seed edges that were not in their own sampled subgraph (68 of 965,524 in see-everything).

**Fixes from the review of the paused run.** (1) **Scoring is always in `eval()` mode** (no dropout, batch-norm running statistics); the paused version also tried `train()` mode and kept the better one, which was removed. (2) Edge ids used to locate the seed edges passed through a float32 column (exact only to 16.7 M); they are now an int64 `eid` attribute, so HI-Medium (31.9 M edges) does not break. (3) The val look-ahead is documented below, not removed.

**Limitation: val look-ahead.** A val edge is scored on the graph of train + the whole val split, so its sampled neighbourhood can contain later val edges (as in Multi-GNN's own setup and GFP's 127-edge batches). Size on HI-Small (val = days 7-8): for a val edge, 12.6 % of its incident edges (both endpoints, train + val) are later than it on average (median 12.2 %), 94 % of val edges have at least one later neighbour edge, median 11 later neighbour edges. Train edges have the same look-ahead inside the train split, so there is no train/val feature shift. It is not cross-split leakage (nothing after the val end, no test), but it flatters the see-everything and one-bank GNN scores a little compared with a strictly causal deployment; a time-restricted neighbourhood (`time_attr` in `LinkNeighborLoader`) would remove it at the cost of a train/score mismatch and was not done.

**Training length.** The time boxes were removed during the run, so: **see-everything** was first trained 4 epochs (4,589 s, best val-`fit` AP 0.528 at the last, partial, epoch: still rising; epochs took 17-20 min on the 3070 Ti with the CPU-bound sampler); **one-bank** runs up to 100 epochs per bank view (74-100 reached; the time share of a view is proportional to its train size), batch 2048, [100, 100]. Seed 1, one run per graph; no seed variance is available. SEE_EXTENSION_PLACEHOLDER

**Results on val `eval` (HI-Small).** One-bank rows are the 28 scored bank views (93,240 rows, 122 laundering), the same rows for both models; two views (banks 19 and 1674) have no positive in the val `fit` part and are null for both. See-everything is the whole graph on the whole val `eval` part (GFP 234,572 rows, Multi-GNN 234,539: 33 seed edges unsampled). F1 at the best-F1 threshold of the same model's val `fit` part.

| | Brier | PR-AUC | F1 | constant Brier |
|---|---|---|---|---|
| GFP+LightGBM one-bank | 0.001167 | 0.202 | 0.283 | 0.001307 |
| Multi-GNN one-bank (28 views) | **0.001110** | **0.243** | **0.341** | 0.001307 |
| GFP+LightGBM see-everything | 0.000830 | 0.481 | 0.541 | 0.001294 |
| Multi-GNN see-everything (4 epochs) | **0.000810** | 0.482 | **0.581** | 0.001290 |

Paired cluster bootstrap (attempt + shared-account clusters, 2,000 resamples, `ftm.eval`): **see-everything, GFP minus Multi-GNN Brier = +0.000016, 95 % CI [-0.000051, +0.000084]: no detectable difference**. **The gap (one-bank minus see-everything Brier, same rows): Multi-GNN 0.000333 [0.000162, 0.000511]; GFP 0.000261 [0.000115, 0.000394].** One-bank Multi-GNN is not significantly better than one-bank GFP either (0.001110 vs 0.001167, no CI computed; PR-AUC 0.243 vs 0.202).

**Headline:** on our split Multi-GNN see-everything is **not a clearly stronger upper limit than GFP see-everything** (Brier 0.000810 vs 0.000830, PR-AUC 0.482 vs 0.481, F1 0.581 vs 0.541; the Brier difference CI contains 0). The project's denominator, the one-bank to see-everything gap, exists under both models and is a little larger with Multi-GNN (0.000333 vs 0.000261), so the gap does not depend on the choice of the strong model. Caveats: the GNN had 4 of 100 epochs on the full graph (EXTENSION_CAVEAT), 303 eval positives, one seed. Per-view numbers, epochs, calibrator and best fit AP of every bank: `data/runs/p0_hi_small_multi_gnn_one_bank.json`; compare json: `data/runs/multi_gnn_vs_gfp_hi_small.json`; logs `data/runs/logs/mg_*.log` (gitignored).

**HI-Medium** (not run here; 5070 Ti machine; needs about 24 GB RAM for the 23 M-row train+val graph and 8+ GB VRAM; the int64 ids make 31.9 M edges safe):

```
python scripts/train_gnn.py --our-split --data hi_medium --mode see_everything --budget-s 28800 --batch-size 2048
python scripts/train_gnn.py --our-split --data hi_medium --mode one_bank --max-epochs 100 --budget-s 28800
python scripts/train_gnn.py --compare --data hi_medium        # needs p0 of GFP see-everything too (blocked, see HI-Medium above)
```

## DP-FedAvg over the bank views (Group B, M5)

`ftm/baselines/federated.py` + `scripts/train_federated.py`; Flower 1.42 (clip and noise helpers from `flwr.supercore.differential_privacy`) and Opacus 1.5.4 (`GradSampleModule`, `DPOptimizer`, RDP accounting). 30 clients = the 30 HI-Small bank views, **train split only** for training, val for evaluation, no test row anywhere. Rounds run in-process and sequentially; Flower's own simulation engine needs Ray, which was not installed (the 1.42 strategy classes are message-based and need a grid), so only its DP helper functions are used and the FedAvg loop is ours. The model is a tiny MLP (hidden 16, about 1,000 parameters), CPU only (the GPU was busy with Multi-GNN; a run takes 10-120 s).

```
python scripts/train_federated.py --data hi_small --unit edge   --eps 1 2 4 8        # ~10 min, 3 seeds each
python scripts/train_federated.py --data hi_small --unit client --eps 1 2 4 8 16     # ~35 min
```

Two privacy units, **never on one axis**. delta = 1e-5 for both.

| | client-level DP-FedAvg (McMahan et al. 2018) | edge-level DP-SGD (Opacus) |
|---|---|---|
| privacy unit | one bank's whole dataset | one transaction |
| mechanism | each client trains locally (no per-example mechanism); its update is clipped to S = 1; the 30 updates are averaged with equal weight; Gaussian noise of std sigma S / 30 is added (Flower `compute_stdv`, `add_gaussian_noise_inplace`) | per client and round: Poisson-sampled batches (rate q = 0.02, the same for every client), per-example clipping to C = 1, Gaussian noise sigma C, 50 local steps; server FedAvg weighted by client train size (sizes treated as public) |
| accountant | Gaussian mechanism over R = 10 rounds, all clients every round (q = 1), RDP | subsampled Gaussian over R x 50 = 500 steps, RDP, **composed twice** because a transaction between two top-30 banks sits in two clients' data |
| features | `ctx`: the transaction's own attributes plus past-only 24 h aggregates of the two endpoints inside the bank's own view (56 features) | `tx`: the transaction's own attributes only (42 features: log amount, log paid/received, same bank, hour sin/cos, one-hot currencies and format) |
| any model? | yes (local training is non-private; only the update is protected) | only a model whose per-example gradient depends on **one** transaction |

**Why the edge-level arm cannot use a message-passing GNN.** The edge-level guarantee needs the summed clipped gradient to change by at most C when one transaction is added or removed. In a GNN the loss of every training edge in the k-hop neighbourhood depends on the neighbours' attributes, so one transaction changes the gradients of many other examples (a hub edge, of thousands), and per-sample clipping bounds none of that. A valid guarantee would need degree-bounded neighbour sampling plus node-level clipping. `tests/test_federated.py` checks the premise on the features: dropping other transactions leaves `tx` features unchanged and changes the `ctx` ones.

Training hyper-parameters are fixed, not tuned on private data (SGD lr 0.5, positive weight 20 in the loss, no search over sigma schedules). The noise multiplier for a target epsilon comes from bisection on the RDP bound. Every number is on the val `eval` part (96,318 bank-view rows, 127 laundering rows, 30 views); one Platt calibrator (2 parameters) is fit on the pooled val `fit` rows (post-processing with val labels, the same protocol as the other models), F1 at the best `fit`-part threshold. Mean of 3 seeds (0, 1, 2); the Brier column shows the seed standard deviation. Constant-predictor Brier on these rows: 0.001317. Skill = 1 - Brier / constant.

**Client-level (unit = one bank's dataset), `ctx` features, R = 10 rounds x 600 local steps** (local steps cost no privacy; rounds do, so few rounds and many local steps; R = 30 x 300 reached 0.056 skill without noise but needs 3x the noise):

| epsilon (client) | noise multiplier | Brier | seed sd | Brier skill | PR-AUC | F1 |
|---|---|---|---|---|---|---|
| 1 | 12.79 | 0.001315 | 2e-6 | 0.002 | 0.0085 | 0.034 |
| 2 | 6.80 | 0.001306 | 2e-6 | 0.008 | 0.0213 | 0.065 |
| 4 | 3.66 | 0.001291 | 2e-6 | 0.020 | 0.0533 | 0.124 |
| 8 | 2.02 | 0.001281 | 7e-6 | 0.027 | 0.0695 | 0.144 |
| 16 | 1.15 | 0.001277 | 3e-6 | 0.030 | 0.0817 | 0.159 |
| inf (no clipping, no noise) | 0 | 0.001267 | 4e-6 | 0.037 | 0.0949 | 0.183 |

**Edge-level (unit = one transaction), `tx` features, R = 10 x 50 steps:**

| epsilon (per transaction, incl. the 2-view composition) | noise multiplier | Brier | Brier skill | PR-AUC | F1 |
|---|---|---|---|---|---|
| 1 | 2.71 | 0.001317 | 0.000 | 0.0015 | 0.003 |
| 2 | 1.58 | 0.001317 | 0.000 | 0.0015 | 0.002 |
| 4 | 1.04 | 0.001317 | 0.000 | 0.0015 | 0.002 |
| 8 | 0.77 | 0.001317 | 0.000 | 0.0015 | 0.002 |
| inf (no clipping, no noise) | 0 | 0.001305 | 0.009 | 0.0216 | 0.066 |

**Reading.** (1) Client-level: the Brier-vs-epsilon curve is monotone and recovers about 75-80 % of the non-private federated skill at epsilon 8-16, but it is below 1 % under epsilon 2; with 30 clients the noise on the averaged update (std sigma S / 30 per coordinate) swamps the signal at small epsilon. (2) Edge-level: the transaction's own attributes (amount, currency, format, hour) carry almost no laundering signal on HI-Small (non-private skill 0.009), so the DP curve is flat at the constant predictor for every epsilon; nothing is lost and nothing is gained. The edge-level guarantee is therefore only available for a model that cannot see the structure that makes laundering detectable; getting that signal back needs neighbour features, which break the unit. (3) The federated models are far weaker than GFP (one-bank Brier skill 0.107, 28 views, `0.001167` vs `0.001307`) because they are a 1,000-parameter MLP on 14 aggregates, not a GBT on 55 graph features; the curve shows the privacy cost relative to the same non-private federated model, not to the project's strongest one-bank model. The 30 views here include banks 19 and 1674, which GFP skipped.

**Caveat for the report (tracks.md 3.7).** The training-time epsilon above is a guarantee about the released federated model for the stated unit. It is **not** the per-question epsilon the project reports for its answers: the values come from different mechanisms and units (a bank's dataset or a transaction vs a query answer), and epsilons of different units or different accountants must not be compared or added. The calibrator and the epoch/hyper-parameter choices use val labels outside the DP accounting. Server weights in the edge arm depend on client sizes (not protected). Results are for HI-Small only.

Json (settings, per-seed rows, commit): `data/runs/federated_hi_small_{edge,client}.json` (gitignored). The runs used the code of commit `4c13ffe` (the json records the parent commit `f5fb9ae` because the code was uncommitted when they ran).

## Files

- `gfp_gbt.py`: GFP wrapper, features, LightGBM fit, calibrators, metrics, `score_graph`, `p0_table`, `read_p0`.
- `multi_gnn.py`, `multi_gnn_t1.patch`: launcher and compatibility patch for the IBM code (kept outside the repo).
- `federated.py`, `scripts/train_federated.py`: DP-FedAvg (client-level and edge-level arms).
- `fraudgt_t1.patch`: patch for the FraudGT checkout (reproduction only, section above).
- Not done in this stage: MEGA-GNN, the AML-specific federated method.
