# Results

This page collects every result in one place and links the files that hold the detail. Test results come from the one
frozen final run; everything else is validation (val) only.

## 1. The final test run (g5)

| | |
|---|---|
| Run id | `g5` (`outputs/final/g5/`, not in git) |
| Freeze | `prereg/FREEZE.json`, combined hash `b094561b57d2f8e3da6bd6945e8146904c8baec4b44910c633fce0b04722b97f`, no TA (waived) |
| Data | HI-Medium test split. Headline: the 771 askable alerts of the full test set (alerts with a partner bank the home bank can ask); ablations and some arms on the frozen 1,000-alert subset (208 askable alerts) |
| Backbone | Gemma 4 12B (QAT W4A16) on vLLM 0.31.0, thinking off, temperature 0, same caps for every LLM system |
| Compute | 44 GPU-hours on the RTX 5070 Ti; T1's `--final` steps on the RTX 3070 Ti laptop |
| What ran | the reduced arm list of [`t4_reduced_final.md`](t4_reduced_final.md); arms that did not run are "not run" in every table |
| Tables | archived in [`results/test_analysis/`](results/test_analysis/) (copied from `outputs/final/g5/analysis/`) |
| Regenerate | `FTM_FINAL_RUN=1 python scripts/final_analysis.py --split test --run outputs/final/g5 --inputs configs/analysis/inputs_test.yaml --fixed outputs/analysis/val/fixed_on_val.json` |

### Main comparison (771 askable alerts, ε = 2 per alert unless stated)

Brier: lower is better. Gap closed = (B_one_bank − B_system) / (B_one_bank − B_see_everything). 95% cluster-bootstrap CIs.

| System | Brier | Gap closed [95% CI] |
|---|---|---|
| One bank only (GFP + LightGBM, p0) | 0.1668 | 0 (reference) |
| GFP see-everything (upper limit) | 0.0966 | 1 |
| Ours, compose, ε = 2 | 0.1673 | −0.007 [−0.038, 0.026] |
| Ours, ε = 1 | 0.1666 | |
| Ours, ε = 4 | 0.1679 | |
| Raw sharing with 3 banks (no privacy) | 0.1640 | 0.040 [−0.107, 0.161] |
| Random planner | 0.1698 | |
| Fixed-order planner | 0.1694 | |
| EIG planner | 0.1823 | |
| Learned policy | 0.1814 | |
| Enumerator | 0.1708 | |
| BED-LLM (best external agent, frozen on val) | 0.1956 | −0.410 |

Cost of our system per alert: 2.4 LLM calls, 12.9k tokens, 29 GPU-seconds.

### Hypotheses

| Hypothesis | Tier | Result |
|---|---|---|
| H1 gap closed ≥ 30% | primary | −0.007, not supported |
| H2 ≥ 70% of the raw-sharing benefit | primary | not supported |
| H3 LLM composing ≥ simple planners and enumerator, > fixed list | primary | not supported (p_holm 0.33) |
| H4 back-testing helps, FDR audit < 10% useless | primary | −0.005 [−0.010, −0.000], p_holm 0.072; not supported (the FDR audit fails) |
| H9 ours ≥ best ported external agent | primary | ours minus BED-LLM, gap closed 0.266, p_holm 0.004: **supported** |
| H7 formula beats LLM deciding | secondary | not supported |
| H6 retrieval, H10 vs DP-FedAvg | secondary | not run on test |
| H5 fast model, H8 batched questions | exploratory | not run on test (validation only) |

Definitions, thresholds and the Holm order are in [`../prereg/hypotheses.yaml`](../prereg/hypotheses.yaml).

### Ablations (subset, 208 askable alerts; Brier of the ablation minus the full system)

| Ablation | Δ Brier [95% CI] |
|---|---|
| Fixed question list | +0.0080 [0.0031, 0.0131] |
| No back-testing | +0.0049 [0.0003, 0.0099] |
| Act only | +0.0016 (not significant) |
| LLM decides | +0.0001 (not significant) |

### Reading the test results

- The private questions did not move the Brier score away from the home bank's own model (gap closed about 0).
  Raw sharing with three banks closes 0.040 of the gap, with a CI that includes 0.
- Within the system, dropping composed questions (fixed list) or back-testing makes the Brier score clearly worse on the subset.
- Our system has a much lower Brier score than the best external LLM agent under the same gate, budget and backbone (H9).
- Every deviation that affects these tables is listed in [`t4_reduced_final.md`](t4_reduced_final.md) and the
  "Known deviations" section of the [README](../README.md#known-deviations).

## 2. Validation results

All of these use val data only (HI-Medium unless stated). The test split was never read for them.

| Topic | Where | Summary |
|---|---|---|
| Dry run of the final analysis | [`results/val_analysis/`](results/val_analysis/) | every table of the test analysis on the 160 askable val eval alerts; no hypothesis supported; `fixed_on_val.json` is the input the test analysis takes as fixed |
| All LLM systems on val eval | [`t3_val_runs.md`](t3_val_runs.md) (generated) | Brier, CIs, calls, tokens and seconds of our 7 arms and every external agent |
| Why no planner mode beats p0 | [`t3_val_analysis.md`](t3_val_analysis.md) | replay analysis of the val runs (p0 0.1896 against ours 0.1907 on the 160 alerts) |
| Detection baselines | [`t1_baselines_val.md`](t1_baselines_val.md) (generated) | GFP, Multi-GNN, MEGA-GNN, FraudGT, federated, layer-wise on the val eval part, with paired CIs |
| Baseline fidelity | [`baseline_fidelity.md`](baseline_fidelity.md) | per baseline: original method, what we had to invent, evidence of fidelity |
| Fusion and threshold | [`t3_headroom.md`](t3_headroom.md), [`t4_fusion_findings.md`](t4_fusion_findings.md) | why naive fusion hurts; the fitted p0-conditional fusion that was frozen |
| Federated ε reporting | [`t1_federated_eps.md`](t1_federated_eps.md), [`figures/`](figures/) | DP-FedAvg Brier against ε |
| Experiments E8–E18 | [`EXPERIMENTS.md`](EXPERIMENTS.md) | one row per experiment with its note |

## 3. Result archives

Generated reports copied from the machines that made them, so no result lives only in a gitignored folder.

| Folder or file | What |
|---|---|
| [`results/test_analysis/`](results/test_analysis/) | the test tables of g5 |
| [`results/val_analysis/`](results/val_analysis/) | the validation dry run of the final analysis |
| [`results/README.md`](results/README.md) | the list of the other archived reports: RAG bench, System 1, E15 with the fast-model screen, E16, S1BENCH Part B, graph-versus-legacy parity |
| [`bench/`](bench/) | the raw LLM throughput benchmark on the 5070 Ti |
