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
- [ ] GFP+GBT per bank and full
- [ ] Multi-GNN reproduction on HI-Small. **G1**

## Week 3
- [ ] Conformal intake
- [ ] Multi-GNN on our split (one-bank + see-everything)
- [ ] p₀ and upper-limit predictions to T2/T4. **G2** support

## Week 4
- [ ] Final alert sets
- [ ] freeze 1k subset
- [ ] start edge scorer training. **G3**

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
