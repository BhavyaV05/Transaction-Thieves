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
- [ ] run the conversion benchmark (waiting for the T3 script)

## Week 1
- [x] Parquet, splits, pattern parsing, bank views, aliasing, time guard (HI-Small and HI-Medium; docs in ftm/data/README.md). **Hand off to everyone by end of week**

## Week 2
- [x] GFP+GBT per bank and full (HI-Small: both modes; HI-Medium: one-bank done, see-everything see `ftm/baselines/README.md`; reproduction F1 62.95/62.51 vs 62.86/63.23)
- [x] Multi-GNN reproduction on HI-Small. **G1** (Multi-PNA+EU, own split, 5 epochs: test F1 67.64 vs paper 68.16 +- 2.65)

## Week 3
- [x] Conformal intake (Mondrian + time weighting + selective-risk bound reported, not applied; achieved vs target in `ftm/intake/README.md`; exact SCRC of plan ref [38] is a TODO there)
- [ ] Multi-GNN on our split (one-bank + see-everything)
- [ ] p₀ and upper-limit predictions to T2/T4. **G2** support (p₀ one-bank + see-everything GFP are on disk, schema in `ftm/baselines/README.md`; SOTA GNN upper limits not done)

## Week 4
- [ ] Final alert sets (draft train/val sets built; drop-rule ruling and test projection need the team, see `ftm/intake/README.md`)
- [ ] freeze 1k subset
- [ ] start edge scorer training. **G3** (data prep + tiny MLP baseline in `ftm/retrieval/edge_scorer.py`, val AUC 0.986)

## Week 5
- [ ] DP-FedAvg setup
- [ ] PCST

## Week 6
- [ ] DP-FedAvg training
- [ ] FraudGT (secondary)

## Week 7
- [ ] AML-specific federated method (secondary)
- [ ] four evidence views final
- [ ] planner-view decision with Akshat

## Week 8
- [ ] Held-out-cycles rerun (data, GFP, conformal; T2 reruns back-testing, T4 refits fusion)

## Week 9
- [ ] All baselines on val
- [ ] retrieval demo. **G4**

## Week 10
- [ ] MEGA-GNN (secondary)
- [ ] GNNExplainer / PGExplainer / PPR comparisons

## Week 11
- [ ] Federated ε reporting and caveat text

## Week 12–13
- [ ] Final configs
- [ ] buffer

## Week 14
- [ ] Final test run for: GFP, SOTA GNNs, federated, retriever evaluation
- [ ] **half of the LLM investigations run on your server if the week-0 test passed**

## Week 15–16
- [ ] Results tables
- [ ] review
