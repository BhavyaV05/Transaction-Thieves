# Every baseline on the validation set (week 9, T1)

Owner: T1 (Abhishek). Val `eval` part only (the last quarter of the val time span; nothing on test, `FTM_FINAL_RUN` never set). Produced by `python scripts/baselines_val_table.py --data <ds>` (JSON in `data/runs/baselines_val_<ds>.json`), p0 of every model from its own file under `data/scores/<ds>/`.

**How to read the tables.** Rows are (transaction, bank view) pairs of the val `eval` part, restricted to the **common rows** (every listed model has a p0 there: 28 of the 30 views on HI-Small because GFP and Multi-GNN skip banks 19 and 1674, 30 on HI-Medium); see-everything models are matched to those rows by transaction. Brier, PR-AUC, and F1 at the threshold that maximises F1 on that model's own val `fit` rows. Multi-seed models (DP-FedAvg, layer-wise exchange: 3 seeds) show the mean over seeds and the seed sd, with the Brier CI computed on the seed-mean squared error; their PR-AUC and F1 CIs are for seed 0. **CIs are cluster bootstrap by attempt + shared accounts with a hub cap of 20** (the plan's definition without the cap puts 16 % of the HI-Small and 58 % of the HI-Medium rows in one cluster, above the 0.2 cut of decision 6; with the cap the largest cluster is 0.26 % and 0.17 %), 2,000 resamples for Brier and 500 for PR-AUC and F1, paired for the "minus GFP one-bank" column. The constant predictor is the base rate of the val `fit` rows. Calibration is the project's: isotonic or Platt on the val `fit` part for GFP and the GNNs, a pooled Platt for the federated models. Val look-ahead of the GNNs (a val edge is scored on the graph of train + all of val) is documented in `ftm/baselines/README.md`.

## HI-Small

### hi_small: 93,239 common val eval rows, 122 positives, hub-capped clusters 23,841 (largest 0.0026)

| model | seeds | Brier [95 % CI] | skill vs constant | Brier minus GFP one-bank [CI] | PR-AUC [CI] | F1 [CI] |
|---|---|---|---|---|---|---|
| Constant (fit base rate) | 1 | 0.001307 [0.000987, 0.001651] | 0.000 | +0.000140 [+0.000025, +0.000277] | 0.0013 | n/a |
| GFP one-bank | 1 | 0.001167 [0.000903, 0.001439] | 0.107 | +0.000000 [+0.000000, +0.000000] | 0.2021 [0.1162, 0.3112] | 0.283 [0.190, 0.379] |
| Multi-GNN one-bank | 1 | 0.001110 [0.000869, 0.001376] | 0.150 | -0.000057 [-0.000159, +0.000032] | 0.2435 [0.1365, 0.3779] | 0.341 [0.228, 0.450] |
| DP-FedAvg client-level (unit: a bank), eps 16 | 3 | 0.001271 (sd 0.000003) [0.000964, 0.001603] | 0.028 | +0.000104 [-0.000004, +0.000233] | 0.0780 [0.0462, 0.1217] | 0.177 [0.120, 0.229] |
| DP-FedAvg client-level (unit: a bank), eps 8 | 3 | 0.001275 (sd 0.000002) [0.000968, 0.001610] | 0.025 | +0.000108 [-0.000001, +0.000237] | 0.0640 [0.0317, 0.1090] | 0.129 [0.069, 0.151] |
| DP-FedAvg client-level (unit: a bank), eps inf (no noise) | 3 | 0.001259 (sd 0.000004) [0.000958, 0.001588] | 0.036 | +0.000092 [-0.000012, +0.000221] | 0.0833 [0.0514, 0.1426] | 0.175 [0.108, 0.239] |
| DP-FedAvg edge-level (unit: a transaction), eps 8 | 3 | 0.001307 (sd 0.000001) [0.000987, 0.001651] | -0.000 | +0.000140 [+0.000025, +0.000278] | 0.0014 [0.0012, 0.0021] | 0.002 [0.003, 0.005] |
| DP-FedAvg edge-level (unit: a transaction), eps inf (no noise) | 3 | 0.001295 (sd 0.000000) [0.000980, 0.001634] | 0.009 | +0.000128 [+0.000017, +0.000260] | 0.0211 [0.0123, 0.0346] | 0.065 [0.035, 0.086] |
| Layer-wise exchange (Ceydeli), fedavg, exchange bwd | 3 | 0.001248 (sd 0.000017) [0.000949, 0.001569] | 0.045 | +0.000081 [-0.000025, +0.000210] | 0.1466 [0.1204, 0.3416] | 0.200 [0.189, 0.416] |
| Layer-wise exchange (Ceydeli), fedavg, exchange none | 3 | 0.001229 (sd 0.000005) [0.000933, 0.001545] | 0.059 | +0.000062 [-0.000041, +0.000191] | 0.1416 [0.0684, 0.1978] | 0.187 [0.103, 0.254] |
| Layer-wise exchange (Ceydeli), fedprox, exchange bwd | 3 | 0.001280 (sd 0.000006) [0.000968, 0.001616] | 0.020 | +0.000113 [+0.000003, +0.000245] | 0.1035 [0.0445, 0.1734] | 0.163 [0.082, 0.242] |
| Layer-wise exchange (Ceydeli), fedprox, exchange none | 3 | 0.001279 (sd 0.000005) [0.000970, 0.001615] | 0.021 | +0.000112 [+0.000003, +0.000243] | 0.1173 [0.0716, 0.1905] | 0.147 [0.085, 0.230] |
| Layer-wise exchange (Ceydeli), local, exchange none | 3 | 0.001290 (sd 0.000006) [0.000973, 0.001628] | 0.013 | +0.000123 [+0.000017, +0.000249] | 0.1107 [0.0482, 0.2130] | 0.143 [0.041, 0.269] |
| Layer-wise exchange (Ceydeli), sync_sgd, exchange bwd | 3 | 0.001116 (sd 0.000054) [0.000870, 0.001378] | 0.146 | -0.000051 [-0.000154, +0.000050] | 0.2685 [0.1470, 0.3822] | 0.328 [0.185, 0.432] |
| Layer-wise exchange (Ceydeli), sync_sgd, exchange fwd | 3 | 0.001202 (sd 0.000084) [0.000925, 0.001508] | 0.080 | +0.000035 [-0.000059, +0.000142] | 0.1838 [0.0530, 0.1723] | 0.248 [0.065, 0.248] |
| Layer-wise exchange (Ceydeli), sync_sgd, exchange none | 3 | 0.001200 (sd 0.000031) [0.000926, 0.001500] | 0.082 | +0.000033 [-0.000053, +0.000122] | 0.1361 [0.0518, 0.1581] | 0.186 [0.103, 0.238] |
| GFP see-everything | 1 | 0.000906 [0.000675, 0.001167] | 0.307 | -0.000261 [-0.000412, -0.000124] | 0.4189 [0.2966, 0.5247] | 0.474 [0.340, 0.580] |
| Multi-GNN see-everything | 1 | 0.000719 [0.000547, 0.000905] | 0.450 | -0.000448 [-0.000646, -0.000264] | 0.5484 [0.4269, 0.6550] | 0.638 [0.527, 0.726] |


**FraudGT (partial).** Reproduction on our split, 111 of 500 epochs in 3 h (a 5070 Ti job to finish, command in `ftm/baselines/README.md`): best val F1 **0.648** (precision 0.741, recall 0.575, val AUC 0.988; the same-condition Multi-GNN val F1 values are 0.58 to 0.64), measured on a random subset of val edges per epoch with a threshold-free argmax, so the best epoch is optimistic by about 0.03. **No predictions were saved, so Brier, PR-AUC and calibrated F1 on the val `eval` rows are not available** (not in the table; they need the full run with a prediction dump). Not comparable to the other rows' F1, which uses calibrated probabilities and a `fit`-part threshold.

**HI-Small reading.** (1) Multi-GNN one-bank (0.001110) is nominally better than GFP one-bank (0.001167) but the paired CI covers 0 (-0.000057 [-0.000160, +0.000033]); both beat the constant (CI of the difference excludes 0 for GFP). (2) The see-everything upper limits are clearly better (GFP -0.000261, Multi-GNN -0.000448 Brier, CIs exclude 0), the one-bank to see-everything gap is what the agent has to close. (3) DP-FedAvg is far weaker than both one-bank models (client-level 0.00126 to 0.00127; its improvement over the constant is small but its paired CI excludes 0 at eps 8 (-3.2e-5 [-4.8e-5, -1.8e-5]) and eps 16 (-3.6e-5 [-5.4e-5, -2.1e-5]); the edge-level arm at eps 8 is the constant predictor, and only the non-private edge-level model is marginally better than it (-1.1e-5 [-1.7e-5, -6e-6]); the `Brier minus constant` CIs are in `data/runs/baselines_val_hi_small.json`). Epsilons are training-time guarantees of different units (a bank for client-level, a transaction for edge-level) and not comparable to the per-question eps.

## HI-Medium

### hi_medium: 320,367 common val eval rows, 527 positives, hub-capped clusters 98,478 (largest 0.0017)

| model | seeds | Brier [95 % CI] | skill vs constant | Brier minus GFP one-bank [CI] | PR-AUC [CI] | F1 [CI] |
|---|---|---|---|---|---|---|
| Constant (fit base rate) | 1 | 0.001644 [0.001447, 0.001851] | 0.000 | +0.000193 [+0.000073, +0.000322] | 0.0016 | n/a |
| GFP one-bank | 1 | 0.001451 [0.001302, 0.001597] | 0.117 | +0.000000 [+0.000000, +0.000000] | 0.2327 [0.1743, 0.2890] | 0.350 [0.292, 0.401] |
| Multi-GNN one-bank | 1 | 0.001330 [0.001191, 0.001472] | 0.191 | -0.000121 [-0.000192, -0.000046] | 0.2837 [0.2309, 0.3403] | 0.369 [0.312, 0.424] |
| DP-FedAvg client-level (unit: a bank), eps 16 | 3 | 0.001596 (sd 0.000008) [0.001432, 0.001778] | 0.029 | +0.000146 [+0.000054, +0.000242] | 0.0770 [0.0577, 0.1069] | 0.167 [0.140, 0.199] |
| DP-FedAvg client-level (unit: a bank), eps 8 | 3 | 0.001607 (sd 0.000012) [0.001438, 0.001789] | 0.023 | +0.000156 [+0.000062, +0.000256] | 0.0635 [0.0430, 0.0811] | 0.149 [0.109, 0.157] |
| DP-FedAvg client-level (unit: a bank), eps inf (no noise) | 3 | 0.001569 (sd 0.000007) [0.001409, 0.001748] | 0.045 | +0.000118 [+0.000032, +0.000210] | 0.0938 [0.0724, 0.1309] | 0.195 [0.158, 0.229] |
| DP-FedAvg edge-level (unit: a transaction), eps 8 | 3 | 0.001644 (sd 0.000000) [0.001447, 0.001851] | 0.000 | +0.000193 [+0.000073, +0.000322] | 0.0017 [0.0016, 0.0022] | 0.004 [0.004, 0.005] |
| DP-FedAvg edge-level (unit: a transaction), eps inf (no noise) | 3 | 0.001646 (sd 0.000000) [0.001461, 0.001843] | -0.002 | +0.000195 [+0.000083, +0.000313] | 0.0209 [0.0164, 0.0294] | 0.074 [0.058, 0.091] |

Not run on HI-Medium: layer-wise exchange (BLOCKED-COMPUTE: full-batch message passing over each client's whole extended subgraph; HI-Small's bank 70 view with 0.37 M edges already needs most of the 8 GB with activation recomputation, HI-Medium's has 2.1 M; it would need the neighbour-sampled variant of the paper or a larger GPU: `python scripts/train_layerwise.py --data hi_medium --steps 480 --lr 6e-3 --arms sync_sgd:bwd --seeds 0` on the 5070 Ti), FraudGT (partial on HI-Small only).

**BLOCKED-COMPUTE (5070 Ti commands).** HI-Medium see-everything for GFP and for Multi-GNN, needing at least 12 GB of free RAM (GFP) and 24 GB RAM plus 8 GB VRAM (Multi-GNN), several hours each:

```
python scripts/train_gfp.py --data hi_medium --mode see_everything --neg-frac 0.1 --threads 16
python scripts/train_gfp.py --data hi_medium --recalibrate
python scripts/train_gfp.py --data hi_medium --report            # one-bank vs see-everything gap
python scripts/train_gnn.py --our-split --data hi_medium --mode see_everything --budget-s 28800 --batch-size 2048
python scripts/train_gnn.py --compare --data hi_medium
```

**HI-Medium reading.** Multi-GNN one-bank (Brier 0.001330, PR-AUC 0.284) beats GFP one-bank (0.001451, 0.233): paired difference -0.000121 [-0.000192, -0.000046], so on HI-Medium the stronger one-bank starting model is significantly better, unlike on HI-Small. Bank 70 (a quarter of the laundering) is invisible to both. Client-level DP-FedAvg stays at 2 to 3 % skill (non-private 4.5 %, 0.001569; the improvement over the constant has a CI that excludes 0 at eps 8 and 16: -3.7e-5 [-7.2e-5, -2e-6] and -4.7e-5 [-8.7e-5, -7e-6]), edge-level is the constant predictor.

## Layer-wise exchange (Ceydeli et al.)

Ceydeli, Wang and Atasu, arXiv:2605.06433v2, implemented from the paper (no code released; Tian et al.'s unlicensed code was not run); method, deviations and tests in `ftm/baselines/layerwise.py` and the "Layer-wise exchange" section of `ftm/baselines/README.md`. HI-Small only, 30 clients = the 30 bank views, train split only, scored on val `eval`; 8 arms x 3 seeds = 24 runs of 22 to 27 min on the 3070 Ti, 480 lock-step steps each, one hyper-parameter set for every arm. Brier, mean +- sd over 3 seeds (rows in the HI-Small table above; constant 0.001307, GFP one-bank 0.001167, Multi-GNN one-bank 0.001110, DP-FedAvg client-level 0.00126 to 0.00127):

| optimisation | no exchange | forward exchange | forward + backward exchange |
|---|---|---|---|
| Fully Local | 0.001290 +- 0.000006 | | |
| FedAvg | 0.001229 +- 0.000005 | | 0.001248 +- 0.000017 |
| FedProx (mu 0.01) | 0.001279 +- 0.000005 | | 0.001280 +- 0.000006 |
| Sync-SGD | 0.001200 +- 0.000031 | 0.001202 +- 0.000084 | **0.001116 +- 0.000054** |

**Reading.** (1) **Sync-SGD with forward and backward exchange is the best arm** (0.001116 Brier, PR-AUC 0.269, F1 0.328) and it is at the level of Multi-GNN one-bank (0.001110) and better than GFP one-bank (0.001167), though the paired CIs cover 0 (-0.000051 [-0.000154, +0.000050] against GFP one-bank). With no exchange the same optimisation gives 0.001200 and with forward exchange only 0.001202: **the backward exchange is what carries the gain** here (-0.000084 Brier against no exchange, about 2.7 seed sd of the no-exchange arm), forward exchange alone does nothing. That is the paper's ordering for the exact-equivalence setting (Theorems 1 and 2, verified in `tests/test_layerwise.py`: with shared parameters the distributed forward and gradient equal the centralised ones to 1e-5). (2) **FedAvg and FedProx do not profit**: with exchange they are within a seed sd of no exchange (FedAvg 0.001248 vs 0.001229, FedProx 0.001280 vs 0.001279; the paper also finds FedAvg with exchange unstable), and parameter drift keeps them far from Sync-SGD (FedAvg skill 4 to 6 %, FedProx 2 %). Fully Local is close to the constant (skill 1 %). (3) Sync-SGD needs a per-step parameter exchange between the 30 banks, so its privacy cost is not that of a few FedAvg rounds; the layer-wise exchange sends node representations and adjoints of boundary accounts at every layer and step (not accounted here, no DP). It is therefore a non-private upper reference for the federated family, not a privacy competitor to DP-FedAvg. (4) Limits: the 30 clients own 30 of 30,470 banks, so most remote neighbours have no owner (zero in every mode) and the exchange has little to carry; full-batch message passing instead of neighbour sampling; no ego ids or port numbers; 3 seeds; the seed sd of Sync-SGD (5e-5) is as large as most gaps; nothing tuned on val besides the step choice (the best step is 200 to 480 of 480 for the synchronised arms, 120 to 200 for Fully Local: the synchronised arms were still improving at the end, so more steps would probably help). The pilot (1,000 steps, one seed, before the bug fix below) had Brier 0.001113.

**A bug found on the way.** The first grid (24 runs) used an edge update from node representations that had not been exchanged yet; the Theorem 1/2 equivalence test (`tests/test_layerwise.py`) caught it (logits off by 1e-3), the node outputs are now exchanged before the edge update, and the whole grid was rerun. Only the rerun is reported.

## Open items

- Layer-wise exchange: only HI-Small, 3 seeds, 480 steps (still improving); 5 seeds, longer runs, neighbour sampling and a partition of all banks as in the paper are open.
- HI-Medium see-everything (GFP, Multi-GNN): BLOCKED-COMPUTE, commands above.
- FraudGT: partial; the full 500-epoch run with a prediction dump is a 5070 Ti job.
- MEGA-GNN (week 10) is not in the table.
