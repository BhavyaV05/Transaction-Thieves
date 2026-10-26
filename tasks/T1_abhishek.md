# T1: Data, detection models and retrieval (Abhishek)

Checklist generated from tracks.md section 3.9. Read the whole track section first: it has the specs and acceptance tests.
Tick items off in your PRs and post one line per week in STATUS.md. Weeks count from TA approval (G0).

## Week 0
- [x] Download both datasets (HI-Small and HI-Medium built; see ftm/data/README.md)
- [x] install snapml and PyG
- [x] trace the ~80 F1 source (ExSTraQt, arXiv:2604.02899; docs/t1_week0.md)
- [x] check code/licence for Group A/B and pick the AML-specific federated method (Tian et al.; docs/t1_week0.md)
- [x] report RAM/CPU
- [x] **test whether Qwen3-8B 4-bit runs on your GPU at the harness context cap**: passed as an ollama Q4_K_M proxy (6.4 GB VRAM, 62 tok/s); vLLM check still open.
- [ ] run the conversion benchmark: script available; BLOCKED for the official factor (T3's reference JSON is not on this machine); indicative ollama proxy run, factor 0.664, `docs/t1_bench.md`

## Week 1
- [x] Parquet, splits, pattern parsing, bank views, aliasing, time guard (HI-Small and HI-Medium; docs in ftm/data/README.md). **Hand off to everyone by end of week**

## Week 2
- [x] GFP+GBT per bank and full (HI-Small: both modes; HI-Medium: one-bank done, see-everything see `ftm/baselines/README.md`; reproduction F1 62.95/62.51 vs 62.86/63.23)
- [x] Multi-GNN reproduction on HI-Small. **G1** (Multi-PNA+EU, own split, 5 epochs: test F1 67.64 vs paper 68.16 +- 2.65)

## Week 3
- [x] Conformal intake (Mondrian + time weighting + selective-risk bound reported, not applied; achieved vs target in `ftm/intake/README.md`; exact SCRC of plan ref [38] is a TODO there)
- [x] Multi-GNN on our split (one-bank + see-everything), HI-Small; HI-Medium command in ftm/baselines/README.md (5070 Ti)
- [x] p₀ and upper-limit predictions to T2/T4. **G2** support (GFP p₀ and `p0_multi_gnn.parquet` on disk for HI-Small, schema in `ftm/baselines/README.md`; HI-Medium see-everything still blocked-compute)

## Week 4
- [x] Final alert sets (train/val sets final under decisions 1 to 5 of `docs/t4_decisions_week2.md`, alert definition in `configs/t1_final/intake.yaml`; the test set comes from `make_alerts.py --final` at G5, `docs/t1_final_run.md`)
- [x] freeze 1k subset (the rule, not the IDs, decision 1: `configs/t1_final/subset_rule.yaml`; the IDs are drawn by `run_final.py` at G5; T4 quotes the rule commit in `prereg/`)
- [x] start edge scorer training. **G3** (trained: val eval AUC 0.993, AP 0.228; `ftm/retrieval/README.md`)

## Week 5
- [x] DP-FedAvg setup
- [x] PCST (time-respecting; needs the patched pcst_fast, see ftm/retrieval/README.md)

## Week 6
- [x] DP-FedAvg training (client-level and edge-level curves, HI-Small)
- [x] FraudGT (secondary): partial reproduction on our val days (3 h, 111 of 500 epochs, val F1 0.648; full run is a 5070 Ti job), see ftm/baselines/README.md

## Week 7
- [x] AML-specific federated method (secondary): Ceydeli et al. layer-wise exchange from the paper, HI-Small (`ftm/baselines/README.md`)
- [x] four evidence views final (`view()` API, both datasets, H6)
- [ ] planner-view decision with Akshat (recommendation written, `docs/t1_planner_view.md`; awaiting his confirmation)

## Week 8
- [x] Held-out-cycles rerun (data, GFP, conformal, T2's held-out back-test run with her; T4 refits fusion) (`docs/t1_heldout_cycles.md`)

## Week 9
- [x] All baselines on val (`docs/t1_baselines_val.md`; HI-Medium see-everything and FraudGT full run still blocked-compute)
- [x] retrieval demo (`scripts/retrieval_demo.py`). **G4**

## Week 10
- [x] MEGA-GNN (secondary): reproduction partial (59.7 vs 74.88), our split HI-Small both modes and HI-Medium one-bank, HI-Medium see-everything BLOCKED-COMPUTE (`ftm/baselines/README.md`, `docs/t1_baselines_val.md`)
- [x] GNNExplainer / PGExplainer / PPR comparisons (`ftm/retrieval/README.md`, both datasets)

## Week 11
- [x] Federated ε reporting and caveat text (`docs/t1_federated_eps.md`, `docs/figures/t1_federated_eps_<ds>.png`)

## Week 12–13
- [x] Final configs (`configs/t1_final/`, `scripts/t1_final_check.py`, `docs/t1_final_run.md`; FraudGT and HI-Medium see-everything need the 5070 Ti)
- [ ] buffer

## Week 14
- [ ] Final test run for: GFP, SOTA GNNs, federated, retriever evaluation
- [ ] **half of the LLM investigations run on your server if the week-0 test passed**

## Week 15–16
- [ ] Results tables
- [ ] review
