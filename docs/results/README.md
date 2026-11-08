# Results archive (validation only, 8 Nov 2026)

Generated reports copied verbatim from the machines that produced them, so no result lives only in a gitignored `outputs/` folder or on the ada cluster. Validation data only; nothing here reads or reports the test split.

| File | What | Produced by | Source on the machine |
|---|---|---|---|
| `rag_bench_fitcal.md`, `rag_bench_selection.json` | RAG bench: every arm on val fit + cal, the pre-written selection, the recorded `menu_topk_all` override | ada, `scripts/t3_rag_bench.sh report` | `~/tt/tt-main-git/outputs/rag_bench/` |
| `rag_bench_selection_before_override.json` | the selection before the override (kept for the record) | ada | same |
| `rag_bench_world460_fitcal.md` | the FinFRE-RAG pool arms, run in the copy with all 31 bank views | ada | `~/tt/tt-main-460/outputs/rag_bench_460/` |
| `rag_bench_eval.md`, `rag_bench_eval.json` | RAG bench: defaults, winners and the combined arm on the val eval part, scored once (155 askable alerts in the ada world) | ada, `t3_rag_bench.sh eval` | `~/tt/tt-main-git/outputs/rag_bench_eval/` |
| `system1_fitcal.md`, `system1_selection.json` | System 1: every function arm on val fit + cal and the frozen selection | 5070 Ti, `scripts/t3_rerun.sh system1` | `outputs/s1/fitcal_report/` |
| `system1_eval.md`, `system1_eval.json` | System 1: off vs the selected setting on val eval, scored once; cascade and injection-screen tables | 5070 Ti | `outputs/s1/eval_report/` |
| `e15_system1_screen_{planner,react}.json` | E15 on the held-out injection families: no gate, gate, gate + fast-model screen | 5070 Ti | `outputs/injection_v3/e15_s1_*/` |
| `e16_analysis_{fit,cal,eval}.txt` | E16: prompted Qwen3-8B vs the three QLoRA students vs Gemma compose | 5070 Ti, `t3_rerun_report.py e16` | `outputs/t3_val_v3/analysis_e16_*.log` |
| `s1bench_partb_eval.md`, `.json` | S1BENCH Part B, scored once on the project's own decision rows | 5070 Ti | `outputs/s1bench/partb_eval_once/` |
| `parity_real.md` | graph versions against the legacy loops on the real model (21 runs) | computed from ada's `results/parity_real/*/{g1,g2,l1}/per_alert.jsonl` | this PR |

Raw run folders (per-alert logs, traces, run logs) are backed up on the 5070 Ti under `outputs/ada_results/2026-11-08/` with SHA-256 lists (`SHA256SUMS`, `full_tt.SHA256SUMS`).
