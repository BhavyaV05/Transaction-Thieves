# Baseline fidelity record (T4, updated 7 Nov 2026)

One row per baseline: what is the authors' code or method, what we had to invent, what evidence shows our version behaves like the original, and what is still missing. Every cell comes from the owner's doc named in the last column or from a check named here; nothing is a new measurement unless it says so. Status words: **reproduced** (our run reaches the published number within its stated spread on the published split), **partial**, **not reproduced / not verified**, **reimplementation** (no code or no licence: written from the paper, so there is no number to compare and we only report ours).

Open work: #106 (external agents), #107 (detection baselines and upper limits), #108 (recalibrated p0), #109 (raw-sharing reference), #127 (look-ahead in the see-everything GNN scores).

## What changed since the 1 Nov version of this record

1. **The see-everything Multi-GNN and MEGA-GNN scores use future edges** of the validation window (the loaders sample neighbours without `time_attr`), and so do the official loaders. Re-scored time-respecting on HI-Medium (same checkpoints, nothing trained): Multi-GNN Brier 0.000658 -> 0.001113 and PR-AUC 0.611 -> 0.199; MEGA-GNN 0.000988 -> 0.001266 and PR-AUC 0.342 -> 0.019. GFP (0.000859) then beats both. Caveat from T1: the checkpoints were trained with the look-ahead loader, so the time-respecting rows mix "the signal is gone" with "a distribution the model was not trained on"; a causally trained model is the clean test and was not run (`docs/t1_baselines_val.md`). **Consequence: the stored GNN see-everything scores are baseline rows, not upper limits.** The analysis now refuses them in an upper-limit slot (`ftm.eval.final.check_upper_limit_specs`; each score table declares `look_ahead`) and the baselines table prints the column.
2. **p0 is now v2** (an intercept shift fitted on all `cal` rows, frozen with a hash; v1's calibrator was fit on the weekend-heavy `fit` part and over-predicts later). The selection tolerance was set after eval numbers existed; disclosed in `docs/t1_baselines_val.md`. The recalibrated one-bank baseline is also reported as sensitivity S3.
3. **Agent orchestration moved to LangGraph** (T3, PR #113): our planner and every Group C/D agent run as `StateGraph`s (`ftm/graph`, `langgraph==1.2.19` pinned). The harness (gate, accountant, clock, banks, DSL, back-test) is unchanged and is called from graph nodes; LLM calls still go through `BudgetedClient`. The old loops are kept (`FTM_LEGACY_AGENT_LOOPS=1`) and the graphs are tested to give the **same verdict, probability, question log, trace and LLM requests as the loops on the mock LLM** over fuzzed seeds (`tests/test_graph_agents_ported.py`). What that does not show: the same on the real model. A real-model parity runner exists (`scripts/ada/tasks/parity_real.sh`); I have not seen its result, so "identical on the real model" is not claimed here.
4. **Agents got fixes and a calibration step** (T3/T4, #113): Reflexion makes a forced final answer at loop end, the DSPy metric scores a missing probability as p0, Co-Investigator logs whether its narrative validated, Group C case memory and the FinFRE exemplar pool are built from stratified train alerts, and calibration maps for the LLM systems (Platt, score levels, Sharma's verdict map) are fitted on val `fit`, checked on `cal` and frozen with a hash (`scripts/calibrate_external.py`).
5. **New exploratory arms**: ten `*_adapted` arms run each agent's control strategy inside our system graph (they see p0, the posterior, evidence, rules and the routed menu; the verdict is the fused posterior). They answer "is the gap the control strategy or the knowledge of LRs and p0?" and are reported apart from the published-method rows (prereg: exploratory, subset only).

## Groups A and B: detection models and federated training (T1)

| System | Implementation | Reproduction evidence | Status | Gap to close | Source |
|---|---|---|---|---|---|
| GFP + LightGBM / XGBoost (p0) | IBM `snapml` GraphFeaturePreprocessor, settings of the GFP paper; bins (2, 3) are the library example's | paper split, HI-Small: F1 62.95 (LightGBM), 62.51 (XGBoost) vs 62.86 +- 0.25 / 63.23 +- 0.17 | **reproduced** (within 0.8 points) | an edge sees at most 127 later edges of its own batch (declared `look_ahead: minor`); p0 v2 calibration | `ftm/baselines/README.md`, `docs/t1_baselines_val.md` |
| Multi-GNN (Multi-PNA+EU) | official IBM code, Apache-2.0, outside the repo at `252b025` | paper's own split: test F1 67.64 vs 68.16 +- 2.65 (5 of 100 epochs, 1 seed) | **reproduced as the published method**, which samples future edges | our-split see-everything has about 9 of 100 epochs; the stored scores use look-ahead (above), so it is not an upper limit; the one-bank scores come from the same loader (to check, #127) | `ftm/baselines/README.md`, `docs/t1_baselines_val.md` |
| MEGA-GNN (MEGA-GenAgg) | official repo, Apache-2.0, archived, last code commit `ef60a71`, patched | test F1 59.7 vs 74.88 published after 10 of 80 epochs | **partial, not reproduced** | about 8 of 80 epochs on HI-Medium see-everything; look-ahead as above | `ftm/baselines/README.md` |
| FraudGT | official checkout, no licence, `f88db9d`, patch kept outside the repo; test redirected to val | the 76.41 reference could not be verified (HTTP 429); full 500-epoch run on our split (HI-Small, 7 Nov): best val F1 0.686 at epoch 291, calibrated val eval Brier 0.000673, PR-AUC 0.589 | **our-split run done, published number not verified** | HI-Small only while the headline data is HI-Medium; test was scored once before the freeze (deviation, `docs/t1_final_run.md`, PR #130); its sampled neighbourhoods may share the look-ahead problem (not checked) | `ftm/baselines/README.md` |
| Layer-wise exchange (Ceydeli et al.) | written from the paper | HI-Small best arm Brier 0.001116 vs one-bank GFP 0.001167; HI-Medium 0.001579, does not reach one-bank | **reimplementation** | no published number on our split | `docs/t1_baselines_val.md` |
| DP-FedAvg GNN | Flower + Opacus | client- and edge-level Brier vs eps | own method | training-time eps and per-question eps are different units (H10) | `docs/t1_federated_eps.md` |
| Tian 2025, ExSTraQt | planned as secondary / stretch | no result | **not run** | run or remove from Table 5 | `prereg/external_list.yaml` |

## Groups C and D: LLM agents (T2, T3, T4)

All use Gemma 4 12B (QAT W4A16, thinking off), the same harness toolbox, gate and accountant, 20 LLM calls, a 6,000-token prompt cap, eps 2 per alert. **The Brier values in this table are the 1 Nov validation runs (160 val `eval` alerts, p0 v1 0.1985, old code); T3's reruns on the rewritten pipeline are not in the repo yet** (`docs/t3_val_runs.md` still describes the old code). T3 reports for the new code, on 7 Nov: p0 v2 0.1896, `ours_compose` 0.1907, `bed_llm` 0.2165, `co_investigator_ported` 0.211 after calibration with verdicts on 28% of alerts; that message is the only source, so treat those four numbers as unverified until the docs are regenerated.

| System | Implementation | What is the paper's, what is ours | 1 Nov val Brier (verdict rate) | Sees p0? | Known gap | Source |
|---|---|---|---|---|---|---|
| Sharma | official MIT code ported (Groq llama-3.3-70b swapped for Gemma 12B) | agent loop and tools theirs; verdict map (0.9 / 0.1 / escalate = p0) ours, now fitted on val `fit` | native 0.2150 (1.00), ported 0.2308 (0.988) | yes | backbone 12B vs 70B | `ftm/external/sharma.py` |
| Torres | **reimplementation** (no code, prompts or backbone named) | architecture from section 3; prompts, quotas, disposition map ours | native 0.4302 (0.919), ported 0.4226 (0.969) | p0 as an evidence line, ignored (AUC 0.50) | paper numbers are on SynthAML10, not checkable here | `docs/external_code_check.md` |
| Co-Investigator | **reimplementation** of a simplified pipeline | SAR narrative to probability map ours | native 0.3062 (0.456), ported 0.3100 (0.412) | p0 as evidence line | most alerts rejected by the rule check and scored p0 | `prereg/external_list.yaml` |
| FinFRE-RAG | **reimplementation**, pipeline as published | 1-5 score to probability table (0.05, 0.2, 0.5, 0.8 for 1 to 4) fixed, now with a fitted map | native 0.2323 (1.00), ported 0.2334 (1.00) | via exemplars | exemplar pool now stratified 1:1 | `ftm/external/finfre_rag.py` |
| BED-LLM | **reimplementation** (no code) | K = 4 hypotheses, 3 candidates per step | 0.2096 (1.00) | p0 is one particle | looks best mostly because its smoothing keeps it near p0 | `ftm/external/bed_llm.py` |
| UoT | **reimplementation** (no licence), depth 1 | paper builds a multi-level tree; 3^d calls exceed the cap | 0.2879 (1.00) | yes | depth 1 only | `ftm/external/uot.py` |
| Debate (Du et al.) | **reimplementation** (no licence); AG2 not used because its calls bypass the shared caps | 3 agents, 2 rounds | 0.4225 (1.00) | no | no p0 | `ftm/external/debate_autogen.py` |
| ReAct, Reflexion, ReWOO | reimplemented in the harness; official MIT code read, nothing copied | prompts and loop follow the repos; output rule ours | 0.4557 (0.925), 0.4365 (0.887), 0.5776 (1.00) | **no** | AUC 0.39-0.51; Reflexion now forces a final answer | `docs/t3_val_analysis.md` 4.3 |
| LATS, LLMCompiler | official MIT code adapted | rollouts removed (LATS) | 0.3890 (0.800), 0.4598 (0.963) | no | LATS loses 20% of alerts to no verdict | `docs/t3_val_analysis.md` |
| DSPy-ReAct | DSPy library, MIPROv2 on 40 val `fit` examples | objective Brier | 0.4429 (0.975) | no | program optimised without p0 in the case text; frozen with sha `0697b613...` | `docs/t3_val_runs.md` |
| `*_adapted` arms | ours (agents' control strategies as policies in our graph) | see above | not in the 1 Nov table | yes (it is our pipeline) | exploratory | `docs/rewrite_prereg_proposal.md` |
| MediQ | stretch port, CC-BY-4.0 code adapted | abstain until confident at 0.8 | not in the locked external list | | exploratory row only | `STATUS.md` |

## Component checks (this session)

- Full CPU suite on `main` after #113 and #125: 1729 passed, 1 failed (`test_demo`, needs `data/alerts/hi_medium_v2/` that is absent on this laptop; CI has no data and skips it), 34 skipped.
- The reduced-run selection is accepted by `run_final.py` and writes a partial manifest (`docs/t4_reduced_final.md`); the graph agents are covered by the parity tests above on the mock LLM only.
- No baseline was re-run by T4: GPU and data are not on this laptop.

## Reading rules for the report

1. A reimplementation row is labelled "reimplementation" in Table 5a, and its paper numbers are not quoted as ours.
2. Rows that do not see p0 are shown next to a p0-aware arm; until the reruns land they are labelled "as published, no p0".
3. Every Brier is also shown against p0 recalibrated on labelled validation alerts (S3, #108). On val `eval` that baseline alone gains up to 0.018 over p0 v1, so gains below about 0.02 are not credited to the questions or the agents.
4. A system with a verdict rate below 0.95 is not G4-ranked, and its Brier is mostly p0 by construction.
5. "Upper limit" means the best **time-respecting, reproduced** see-everything model: GFP. Multi-GNN and MEGA-GNN see-everything rows are labelled "uses future edges".
6. The orchestration is LangGraph for every LLM system; the parity of graph and loop is shown on the mock LLM only until T3's real-model parity result is in the repo.
