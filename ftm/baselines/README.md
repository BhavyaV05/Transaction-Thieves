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

Load with `ftm.baselines.gfp_gbt.read_p0(ds, mode=..., split=...)`.

## Features, and the leakage rule

`GraphFeaturePreprocessor` (snapml 1.17.2) with the settings of the GFP paper (Blanusa et al., KDD 2024, arXiv:2402.08593, section 3): one-day window for vertex statistics, fan, degree, temporal-cycle and simple-cycle features, six hours for scatter-gather, cycle length constraint 10, vertex statistics on Timestamp and Amount. Bins (2, 3) are the library example's; the paper does not list them. Plus 7 basic features (amounts, three category codes, same-bank flag, hour of day). Account ids are not features. One-bank models also get `out` (does the bank send or receive the transaction).

**Window semantics (checked on a toy graph and in `tests/test_baselines_gfp.py`):** edges of earlier `transform` calls are history (past only); edges inside **one** call are inserted together, so an edge also sees the other edges of its own batch, including later ones. We therefore feed time-sorted batches of 128 edges (the paper's inference batch size) and never put two splits in one batch. An edge sees all earlier batches plus **at most 127 later edges of the same split**; train features never touch val, val features use train history (earlier split), test is never passed in (`build_features` raises on a test row without `FTM_FINAL_RUN=1`). The 127-edge look-ahead is the same inside train and inside val, so there is no train/val feature shift; it is a limitation, not cross-split leakage. GFP columns that depend on absolute time (mean and sum of the raw timestamp) are dropped (found empirically: they change when all timestamps are shifted), leaving 48 of 66 GFP columns, 55 features in total.

## Reproduction (M1, E1/G1): paper split, raw HI-Small CSV, reproduction only

`scripts/train_gfp.py --repro` runs on the **published** 60/20/20 split by transaction order (it overlaps our test days, so it is gated behind `--repro`, reads the raw CSV, writes nothing to `data/scores` or `data/alerts`, and nothing from it feeds any later model or decision). Hyper-parameters are chosen by validation AP from a tiny fixed grid (the paper's tuned values are not published); the decision threshold is the best-F1 threshold on the paper's val part.

| model (paper split, HI-Small) | minority-class F1 | precision | recall | PR-AUC | published (batch 128) |
|---|---|---|---|---|---|
| GFP + LightGBM | **62.95** | 0.785 | 0.525 | 0.603 | 62.86 +- 0.25 |
| GFP + XGBoost | **62.51** | 0.800 | 0.513 | 0.591 | **63.23 +- 0.17** |

Reproduced within 0.8 points. The 63.23 quoted in tracks.md is the **XGBoost** row of Table 4 of the GFP paper (the AMLworld paper does not give GFP+LightGBM); we ran both. Runtime: features 518 s (5.08 M transactions, 8-10 k rows/s, 4-6 threads), LightGBM grid 462 s, XGBoost grid 359 s. At the fixed 0.5 threshold F1 is 51.4 / 50.9 (the val-tuned threshold matters). Json: `data/runs/gfp_repro_hi_small.json` (commit `b26e990`, seed 0).

Two things that cost F1 before they were fixed, so reviewers know where to look: (1) with LightGBM's default `min_child_samples=20, reg_lambda=0` the validation AP *falls* from 0.30 (1 tree) to 0.08 (100 trees); early stopping on the logloss (the wrapper's default metric next to `average_precision`) then stopped after one tree and gave 50.9 F1. `LGBM_GRID` now uses `min_child_samples=100, reg_lambda=10` and `metric="average_precision"` only. (2) Float32 overflow in kurtosis features (inf) is clipped at 1e30.

## Our split: one-bank vs see-everything (M3), HI-Small

Splits: train days 1-6, val 7-8 (`configs/data/hi_small.yaml`). LightGBM, 4-point grid (num_leaves 31 or 127 x scale_pos_weight 1 or 8, lr 0.05, min_child_samples 100, lambda 10, early stopping on val `fit` AP, colsample/subsample 0.8), logged in `data/runs/p0_<ds>_<mode>.json`. Calibration is fit on the val `fit` part; all numbers below are on val `eval` (the last quarter of the val time span, 93,240 bank-view rows with 122 laundering rows one-bank; the same rows scored by the whole-graph model for the comparison). **Isotonic is used only where the `fit` part has >= 30 laundering rows (1 of 28 banks: bank 70); the other 27 banks use a two-parameter Platt scaling** because an isotonic step function on a handful of positives is mostly plateaus. Two banks (19, 1674) have fewer than 5 train positives and are skipped (`p0` null).

**Headline (pooled over the 28 bank views, val eval part):**

| | Brier | Brier skill vs constant | PR-AUC | F1 at the best `fit`-part threshold |
|---|---|---|---|---|
| constant (train/val-fit prevalence) | 0.001307 | 0 | | |
| one-bank (30 views, own edges only) | 0.001167 | 0.107 | 0.201 | 0.283 |
| **see-everything** (whole graph) | **0.000906** | **0.307** | **0.419** | **0.474** |

Paired bootstrap over rows of (one-bank Brier - see-everything Brier) = 0.000261, 95% CI [0.000139, 0.000386]: the gap is real and large relative to the headroom (see-everything removes 2.9x the Brier skill of one-bank). **The see-everything model is not barely better than one-bank: the gap, which is the project's denominator, exists on HI-Small.** The caveat is statistical power: 122 laundering rows in the eval part, 0-28 per bank (the per-bank table below is noisy; read the pooled row).

Two things to know: bank 70 (the largest, 28 eval positives) is at base rate under both models (PR-AUC 0.001 vs prevalence 0.001); this is a hub with 450 k transactions and no pattern the GFP features pick up, and it dominates the pooled Brier weight but not the PR-AUC. The macro-average Brier over banks is 0.001178 (one-bank) vs 0.000834 (see-everything). Per-bank table: `data/runs/gap_hi_small_per_bank.md` (copied below).

PER_BANK_HI_SMALL

## HI-Medium

HI_MEDIUM_SECTION

## Multi-GNN reproduction (M2)

MULTI_GNN_SECTION

## Files

- `gfp_gbt.py`: GFP wrapper, features, LightGBM fit, calibrators, metrics, `score_graph`, `p0_table`, `read_p0`.
- `multi_gnn.py`, `multi_gnn_t1.patch`: launcher and compatibility patch for the IBM code (kept outside the repo).
- Not done in this stage: our-split Multi-GNN (one-bank, see-everything), FraudGT, MEGA-GNN, federated baselines.
