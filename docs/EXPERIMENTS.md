# Experiments

The experiments E1–E18 of [`plan.md`](../plan.md) section 8, what each one answers, whether it ran in the final test run
(g5), and the note that holds the detail. "Val" means it ran on validation only. Test numbers are in
[`RESULTS.md`](RESULTS.md); the reasons an arm did not run on test are in [`t4_reduced_final.md`](t4_reduced_final.md).

| ID | Experiment | Answers | On test (g5) | Notes and results |
|---|---|---|---|---|
| E1 | Reproduce the detection baselines (GFP + GBT, Multi-GNN, MEGA-GNN, FraudGT) | G1 | T1 `--final` steps (see the deviations in the T1 note) | [`t1_week0.md`](t1_week0.md), [`t1_baselines_val.md`](t1_baselines_val.md), [`baseline_fidelity.md`](baseline_fidelity.md), [`../ftm/baselines/README.md`](../ftm/baselines/README.md), [`t1_final_run.md`](t1_final_run.md) |
| E2 | Privacy-detection curve: ours at several ε, raw sharing, one bank, see-everything | RQ1, H1, H2 | ε = 1, 2, 4 and raw sharing ran; ε = 0.5 and ∞ not run | [`RESULTS.md`](RESULTS.md), [`t2_final_run.md`](t2_final_run.md), [`../ftm/orchestrator/README.md`](../ftm/orchestrator/README.md) |
| E3 | Planner comparison: non-LLM planners, LLM fixed list, LLM compose | RQ2, H3 | yes (H3 not supported) | [`t2_final_run.md`](t2_final_run.md), [`../ftm/orchestrator/README.md`](../ftm/orchestrator/README.md), [`t3_val_runs.md`](t3_val_runs.md) |
| E4 | Back-testing: with and without, FDR audit on later data | H4 | subset (`no_backtest`) and the FDR audit (H4 not supported) | [`t3_headroom.md`](t3_headroom.md), [`t4_fusion_findings.md`](t4_fusion_findings.md), [`../ftm/backtest/README.md`](../ftm/backtest/README.md), [`t1_handoffs.md`](t1_handoffs.md) |
| E5 | External LLM agents (Group C and D) under the same gate, budget and backbone | RQ6, H9 | BED-LLM, the one frozen G4 pick (H9 supported) | [`t3_val_runs.md`](t3_val_runs.md), [`t3_planner_agents.md`](t3_planner_agents.md), [`baseline_fidelity.md`](baseline_fidelity.md), [`external_code_check.md`](external_code_check.md), [`../ftm/external/GROUPC_T2.md`](../ftm/external/GROUPC_T2.md) |
| E6 | DP federated training against private questions | H10 | not run (H10) | [`t1_federated_eps.md`](t1_federated_eps.md) |
| E7 | Gap closed against the see-everything upper limits | headline | GFP see-everything only (time-respecting) | [`t1_baselines_val.md`](t1_baselines_val.md), [`t4_reduced_final.md`](t4_reduced_final.md) |
| E8 | Evidence views (what the planner is shown) | RQ4, H6 | val (H6 not run) | [`t3_e8.md`](t3_e8.md), [`t1_planner_view.md`](t1_planner_view.md), [`../ftm/retrieval/README.md`](../ftm/retrieval/README.md) |
| E9 | Formula against the LLM deciding | H7 | subset (`llm_decides`; H7 not supported) | [`t3_val_runs.md`](t3_val_runs.md), [`RESULTS.md`](RESULTS.md) |
| E10 | Fast model (System 1) and conformal cascade | RQ3, H5 | val | [`t3_system1.md`](t3_system1.md) (current), [`t3_s1bench.md`](t3_s1bench.md), [`t3_e10.md`](t3_e10.md) (first version), [`t3_finetune_fastmodel.md`](t3_finetune_fastmodel.md), archives in [`results/`](results/README.md) |
| E11 | Strategy library: by shape against by account | leakage | val | [`t3_library_draft.md`](t3_library_draft.md), [`t3_rerun.md`](t3_rerun.md) |
| E12 | Held-out laundering cycles | generalisation | val | [`t1_heldout_cycles.md`](t1_heldout_cycles.md) |
| E13 | One question at a time against batched questions | RQ5, H8 | val | [`t3_e13.md`](t3_e13.md) |
| E14 | Robustness: refusals, timeouts, per-account cap | robustness | not run on test | [`../ftm/orchestrator/README.md`](../ftm/orchestrator/README.md) |
| E15 | Prompt-injection tests, with and without the checker | safety | val | [`t4_injection.md`](t4_injection.md), [`t3_system1.md`](t3_system1.md) (fast-model screen), archives in [`results/`](results/README.md) |
| E16 | Fine-tuned (QLoRA) planner against the prompted model and the teacher | fine-tuning | val | [`t3_e16.md`](t3_e16.md), [`t3_finetune_fastmodel.md`](t3_finetune_fastmodel.md), archives in [`results/`](results/README.md) |
| E17 | Three-way decision (FCAC) against one threshold | decision | not run (stretch); one calibrated threshold is used | [`../ftm/decision/threshold.py`](../ftm/decision/threshold.py) |
| E18 | Backbone size sweep that chose Gemma 4 12B | backbone | val (decides the backbone) | [`t3_e18.md`](t3_e18.md), [`../prereg/e18_decision.yaml`](../prereg/e18_decision.yaml), [`t3_week0.md`](t3_week0.md) |

## Other studies (validation only)

| Study | Notes |
|---|---|
| RAG bench: five retrieval components, with and without | [`t3_rag_bench.md`](t3_rag_bench.md), archives in [`results/`](results/README.md) |
| Planner prompt budget under the tuned profile | [`t3_prompt_budget.md`](t3_prompt_budget.md) |
| LangGraph rewrite: graph versions against the old loops | [`results/parity_real.md`](results/parity_real.md), [`rewrite_prereg_proposal.md`](rewrite_prereg_proposal.md), [`t3_ada.md`](t3_ada.md) |
| First real runs (engineering, 20 alerts) | [`t3_first_runs.md`](t3_first_runs.md) |
| Stub audit of the production paths | [`t3_stub_audit.md`](t3_stub_audit.md) |
| p0 recalibration (p0 v2) | [`t1_baselines_val.md`](t1_baselines_val.md), `configs/t1_final/p0_recal_*.json` |
