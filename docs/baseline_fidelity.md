# Baseline fidelity record (T4, 1 Nov 2026)

One row per baseline: what is the authors' code or the authors' method, what we had to invent, what evidence shows our version behaves like the original, and what is still missing. Every cell is taken from the owner's doc named in the last column; nothing here is a new measurement. Status words: **reproduced** (our run reaches the published number within its stated spread on the published split), **partial** (run, but fewer epochs or a different setting, number not reached), **not reproduced / not verified**, **reimplementation** (no code or no licence: written from the paper, so no number to compare; we only report ours).

Open work is tracked in #106 (external agents), #107 (detection baselines and upper limits), #108 (recalibrated p0), #109 (raw-sharing reference).

## Groups A and B: detection models and federated training (T1)

| System | Implementation | Reproduction evidence | Status | Gap to close | Source |
|---|---|---|---|---|---|
| GFP + LightGBM / XGBoost (p0) | IBM `snapml` GraphFeaturePreprocessor, settings of the GFP paper; bins (2, 3) are the library example's (the paper omits them); small fixed grid | Paper split HI-Small, minority F1 62.95 (LightGBM), 62.51 (XGBoost) vs 62.86 +- 0.25 / 63.23 +- 0.17 published | **reproduced** (within 0.8 points) | none for the claim; state that account ids are not features and that an edge sees at most 127 later edges of its own batch | `ftm/baselines/README.md` |
| Multi-GNN (Multi-PNA+EU) | official IBM code, Apache-2.0, outside the repo at `252b025` | Own split of the paper, test F1 67.64 vs 68.16 +- 2.65, 5 of 100 epochs, 1 seed | **reproduced** (inside one published sd) | our-split see-everything has ~9 of 100 epochs: an under-trained upper limit (#107) | `ftm/baselines/README.md` |
| MEGA-GNN (MEGA-GenAgg) | official repo, Apache-2.0, archived, last code commit `ef60a71`, patched | test F1 59.7 vs 74.88 published after 10 of 80 epochs, noisy curve | **partial, not reproduced** | run to the configured epochs or drop from the upper-limit set; never call it SOTA until reproduced (#107) | `ftm/baselines/README.md` |
| FraudGT | official checkout, no licence, `f88db9d`, patch kept outside the repo; test split redirected to val | the 76.41 reference could not be verified (HTTP 429); our run is short, on our val days | **not verified** | reproduce on its own published split with the authors' config, or label "not reproduced" (#107) | `ftm/baselines/README.md` |
| Layer-wise exchange (Ceydeli et al.) | written from the paper (no code used) | HI-Small: best arm Sync-SGD fwd+bwd Brier 0.001116 vs one-bank GFP 0.001167; HI-Medium 0.001579, does not reach one-bank | **reimplementation** | say so in Table 5; no published number to match on our split | `docs/t1_baselines_val.md` |
| DP-FedAvg GNN | Flower + Opacus, ours | client-level and edge-level curves of Brier vs eps; edge level is flat at the constant predictor | own method | units of eps differ from per-question eps (H10 caveat) | `docs/t1_federated_eps.md` |
| Tian 2025, ExSTraQt | planned as secondary / stretch | no result in STATUS | **not run** | decide: run or remove from Table 5 | `prereg/external_list.yaml` |

## Groups C and D: LLM agents (T2, T3, T4)

All run on Gemma 4 12B (QAT W4A16, thinking off), the same harness toolbox, gate and accountant, 20 LLM calls, 6,000-token prompt cap, eps 2 per alert, single seed, temperature 0 (Sharma 0.1). Val `eval`, 160 askable alerts, p0 alone 0.1985 (`docs/t3_val_runs.md`).

| System | Implementation | What is the paper's, what is ours | Val Brier (verdict rate) | Sees p0? | Known gap | Source |
|---|---|---|---|---|---|---|
| Sharma | official MIT code ported (Groq llama-3.3-70b swapped for Gemma 12B) | agent loop and tools are theirs; verdict to probability map (0.9 / 0.1 / escalate = p0) is ours | native 0.2150 (1.00), ported 0.2308 (0.988) | yes (tool `get_shap_explanation`) | map not fitted (`VerdictMap.fit` exists, never run); backbone 12B vs 70B | `ftm/external/sharma.py` |
| Torres | **reimplementation** (no code, no prompts, no backbone named) | architecture from section 3; prompts, quotas, disposition map ours | native 0.4302 (0.919), ported 0.4226 (0.969) | p0 as an evidence line, ignored (AUC 0.50) | paper numbers are on SynthAML10, cannot be checked on AMLworld | `docs/external_code_check.md` |
| Co-Investigator | **reimplementation** of a simplified pipeline (paper gives no prompts or thresholds) | SAR narrative to probability map is ours | native 0.3062 (0.456), ported 0.3100 (0.412) | p0 as evidence line | narrative rejected by the rule check, so most alerts are scored p0: not a usable baseline until the validator rule is decided | `prereg/external_list.yaml` |
| FinFRE-RAG | **reimplementation**, pipeline as published (RF top-10 features, filter + cosine, 20 exemplars, 1-5 score) | 1-5 score to probability table fixed (0.05 for 1, 0.2 for 2, 0.5 for 3, 0.8 for 4) | native 0.2323 (1.00), ported 0.2334 (1.00) | via retrieved exemplars | table not fitted (`fit_score_map` exists) | `ftm/external/finfre_rag.py` |
| BED-LLM | **reimplementation** (no code) | K = 4 hypotheses, 3 candidates per step | 0.2096 (1.00) | p0 is one particle | protected by its own smoothing, so it looks best mostly because it stays near p0 | `ftm/external/bed_llm.py` |
| UoT | **reimplementation** (no licence), depth 1 | paper builds a multi-level tree; ~3^d calls exceed the call cap | 0.2879 (1.00) | yes | depth 1 only, 53% of verdicts extreme | `ftm/external/uot.py` |
| Debate (Du et al.) | **reimplementation** (no licence); AG2 not used because its calls bypass the shared caps | 3 agents, 2 rounds, mean aggregation | 0.4225 (1.00) | no | no p0 | `ftm/external/debate_autogen.py` |
| ReAct, Reflexion, ReWOO | reimplemented in the harness (official MIT code read, nothing copied) | prompts and loop follow the repos; output rule ours | 0.4557 (0.925), 0.4365 (0.887), 0.5776 (1.00) | **no** | AUC 0.39-0.51; mean p 0.75 vs base rate 0.33; verdict rate under 0.95 for two | `docs/t3_val_analysis.md` 4.3 |
| LATS, LLMCompiler | official MIT code adapted | rollouts removed (LATS), waves through the harness clock | 0.3890 (0.800), 0.4598 (0.963) | no | LATS loses 20% of alerts to no verdict | `docs/t3_val_analysis.md` |
| DSPy-ReAct | DSPy library, MIPROv2 on 40 val `fit` examples | objective Brier | 0.4429 (0.975) | no | program optimised without p0 in the case text; 39 prompts too long | `docs/t3_val_runs.md` |
| MediQ | stretch port, CC-BY-4.0 code adapted | abstain until confident at 0.8 | not in the locked external list | | exploratory row only | `STATUS.md` T2 week 11 |

## Reading rules for the report

1. A reimplementation row is labelled "reimplementation" in Table 5a and its paper numbers are not quoted as ours (the papers use other data).
2. Rows that do not see p0 are shown next to the p0-aware arm once #106 lands; until then they are labelled "as published, no p0".
3. Every Brier is also shown against p0 recalibrated on labelled validation alerts (S3, #108); on val `eval` that baseline alone gains up to 0.018 over raw p0, so gains below ~0.02 are not credited to the questions or the agents.
4. A system with a verdict rate below 0.95 is not G4-ranked, and its Brier is mostly p0 by construction.
5. "Upper limit" means the best **reproduced** see-everything model; MEGA-GNN and FraudGT enter only if reproduced.
