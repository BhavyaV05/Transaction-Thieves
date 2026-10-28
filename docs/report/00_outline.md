# Report outline

Owner of the report: T4 (Shreyas). Status written on 25 Oct 2026 from `STATUS.md`, `docs/` and `docs/schedule_nov10.md`. The full draft is due on **6 Nov** and the submission (G6) on **10 Nov**; 7-10 Nov is slack for changes. Test numbers exist only after the final run (G5, 31 Oct-3 Nov); until then every number that comes from a script is a `{{TBD:<id>}}` placeholder (convention in `README.md`).

The course brief asks for: problem, related work, method, experiments, results, limitations and future work; every capability used justified and measured; baselines, ablations, a small carefully designed evaluation set, capability-specific and safety metrics, qualitative examples and a **failure analysis**; no reliance on "the demo looks good" or an unvalidated LLM judge. Sections 6 to 9 are organised to answer that list directly, and section 9 is a section of its own.

Word counts are for the running text of the section, not for tables, figures or references. Target for the whole report: about 15,000 words plus tables and figures.

## Title and abstract

Title: *Follow the Money Without Moving the Data: catching money laundering across banks by asking small, private, tested questions* (the proposal's title).

Abstract: `{{TBD:abstract}}` placeholder, about 200 words, written last (after the results exist, 6 Nov). A skeleton with every number as a placeholder is in `00_abstract.md`. It states the problem, the idea in one sentence, the data, the three headline results (gap closed against both upper limits with CIs, the LLM against the simple planners, the best external agent) and the main limitation. It must contain no number that is not in a table of section 7.

- Owner: T4. Source: sections 1 and 7 of the report.
- Target length: 200 words.

## Sections

### 1 Introduction

- Owner: **T4**. Status: draft 1 in `01_introduction.md` (no test results).
- Sources: `docs/proposal_v5.pdf` sections 1, 2, 3, 6; `plan.md` sections 0, 1, 2; `docs/ta_addendum.md`.
- Contents: problem and motivation; the idea; RQ1-RQ6; H1-H10 with exact thresholds; contributions; scope and honesty; preview of findings (placeholders); roadmap.
- Target length: 2,500 words.

### 2 Related work

- Owner: **T4**; T1 reviews 2.1-2.2 (detection models, federated), T3 reviews 2.4 (agent frameworks). Status: draft 1 in `02_related_work.md`; `references.md` has open CHECK items.
- Sources: `docs/proposal_v5.pdf` Table 1 and references [1]-[41]; `plan.md` sections 7.1 and 18; `docs/external_code_check.md`; `docs/t1_week0.md`; `docs/t3_week0.md`; `configs/compliance/*.yaml` for the regulation part.
- Contents: AML data and detection; federated and privacy-preserving cross-bank work; LLM agents for fraud and AML; general agent frameworks and information-seeking agents; error-controlled use of LLM ideas; RAG and memory; prompt-injection defences; regulation. Each group ends with what it leaves open.
- Target length: 3,000 words.

### 3 Problem and approach

- Owner: **T4**, with a read by every track. Status: draft 1 in `03_problem_approach.md` (27 Oct; Figure 1 redrawn as text, illustrative worked example, Table 3; no results, no placeholders).
- Sources: `docs/proposal_v5.pdf` sections 3-6 and Figure 1; `plan.md` sections 0 and 3 (architecture, invariants); the worked example of one alert; `docs/proposal_v5.pdf` Table 3 (why each capability is needed and how it is measured).
- Contents: formal setting (home bank, alert, partner banks, noisy typed answers, budget); the invariants (no records leave a bank, LLM text never enters the decision, time guard, aliases, one gate for every system); one alert followed through the system; why each of the five capabilities is needed and how it is measured (the brief requires this).
- Target length: 1,500 words and Figure 1 (system overview, redrawn from the proposal).

### 4 Data and setup

- Owner: **T1** (data, splits, intake, bank views, alert sets), T4 (evaluation resources, subset rule). Status: not started; T1 writes it in phase A (25-27 Oct).
- Sources: `ftm/data/README.md`; `ftm/intake/README.md`; `docs/t4_decisions_week2.md` (items 1-6, 9); `docs/t1_handoffs.md`; `plan.md` section 5; `docs/proposal_v5.pdf` section 7 and Table 4; `docs/t1_week0.md` section 4 (snapml licence).
- Contents: AMLworld HI-Small and HI-Medium; temporal split (days 1-9, 10-12, 13-16; the sparse tail reported separately and never pooled); bank views (30 largest banks); aliasing and the time guard; conformal alert intake and alert counts with achieved miss rates; the two reporting levels (on alerts, end to end); the 1,000-alert external subset and its frozen rule; the expert question list, rules corpus, small-decision dataset and injection set; held-out-cycles variant.
- Target length: 1,500 words.

### 5 Methods per component

- Owners by sub-section; T4 assembles and keeps the notation consistent. Status: not started; each owner writes their part in phase B (28-30 Oct) while the runs go.
- Target length: 3,500 words in total; every sub-section ends with how the component is tested ("done when").

| Sub-section | Owner | Sources |
|---|---|---|
| 5.1 One-bank model and upper limits (GFP+GBT, Multi-GNN, FraudGT, MEGA-GNN); conformal intake; evidence retriever (edge scorer, PCST, four views); federated baselines | T1 | `ftm/baselines/README.md`, `ftm/intake/README.md`, `ftm/retrieval/README.md`, `docs/t1_planner_view.md`, `docs/t1_baselines_val.md`; `plan.md` sections 6.1-6.3, 6.12 |
| 5.2 Partner banks, question language, DP mechanisms and accountant, back-testing (e-values, e-BH, families), non-LLM planners, orchestrator and harness | T2 | `ftm/dsl/README.md`, `ftm/banks/README.md`, `ftm/backtest/README.md`, `ftm/harness/README.md`, `ftm/planners/NONLLM.md`; `plan.md` sections 6.4-6.10, 6.16 |
| 5.3 LLM planner and backbone choice, distillation and QLoRA, strategy library, fast model and conformal cascade, Group D frameworks | T3 | `docs/t3_planner_agents.md`, `docs/t3_e18.md`, `docs/t3_finetune_fastmodel.md`, `docs/t3_library_draft.md`, `docs/t3_e13.md`, `docs/t3_week0.md`; `plan.md` sections 6.11, 6.13, 6.14 |
| 5.4 Compliance gate, privacy accounting at the gate and regulatory RAG, evidence fusion, decision and stopping (threshold, FCAC), Group C ports (Sharma, Torres), Group D lighter ports; draft 1 in `05_4_gate_fusion_decision.md` | T4 | `ftm/compliance/`, `ftm/fusion/`, `ftm/decision/`, `ftm/external/`, `docs/external_code_check.md`, `prereg/external_list.yaml`; `plan.md` sections 6.7, 6.9, 6.15, 6.17 |

### 6 Experiments and baselines

- Owner: **T4** (assembles), with each experiment lead supplying the setup of their rows. Status: draft 1 in `06_experiments.md` (systems compared, fairness protocol, metrics, statistics, hypotheses table, test lock); the setup of each experiment's rows is still to be supplied by the leads.
- Sources: `plan.md` sections 7 (Tables 5a and 5b, fairness protocol, component-level comparisons), 8 (E1-E18), 9 (metrics and statistics); `prereg/hypotheses.yaml`, `prereg/external_list.yaml`, `prereg/e18_decision.yaml`, `prereg/subset_manifest.txt`; `docs/t4_decisions_week2.md` items 2 and 7; `tracks.md` section 7.1.
- Contents: systems compared (Table 5a external, Table 5b ablations, nothing removed); the fairness protocol (same alerts, banks, language, gate, ε, backbone, caps; end-to-end and questions-only scores; native and ported; tuning parity); metrics by capability (the brief requires at least one per capability); statistics (cluster bootstrap, Holm families, equivalence margin, power); what was pre-registered and when.
- Target length: 2,000 words and two tables.

### 7 Results

- Owners by sub-section, T4 for 7.4 and for Holm and bootstrap over everything. Status: nothing on test yet; validation-set results are in the docs listed in the numbers table. Written 4-6 Nov.
- Target length: 3,000 words, and the tables and figures of the numbers table below.

| Sub-section | Owner | Contents |
|---|---|---|
| 7.1 Detection baselines and upper limits | T1 | one-bank and see-everything for GFP and the GNNs; federated (H10); retrieval views (H6); held-out cycles |
| 7.2 Questions and planners | T2 | back-testing (acceptance, H4), non-LLM planners, ablations, privacy curve data, robustness |
| 7.3 LLM planner and components | T3 | backbone sweep (E18), planner against planners (H3), fine-tuned planner, fast model and cascade (H5), library, async (H8) |
| 7.4 Headline and external comparison | T4 | gap closed against both upper limits (H1, H2), external agents (H9), hypothesis table with Holm, per-typology breakdown, Pareto plots |

### 8 Safety

- Owner: **T4**; T2 supplies the privacy-ceiling numbers; T3 the fast-model screen. Status: draft 1 in `08_safety.md` (28 Oct; threat model, layers, E15 design, expected table with placeholders, limits; no result). Injection set, gate and E15 pipeline built and dry-run on the mock LLM; real injection runs 28-30 Oct.
- Sources: `plan.md` section 10; `docs/proposal_v5.pdf` section 10; `ftm/safety/`; `scripts/make_injection_set.py`; `scripts/ceiling_check.py`; `configs/compliance/`; `tests/test_compliance_gate.py`, `tests/test_injection_set.py`.
- Contents: what privacy protects and what it costs (one transaction; per-account cap; calibration budget; the ceiling); no records leave a bank (static check and tests); the compliance gate (allowlist, legal basis, 100% of disallowed actions blocked in the hand-written suite); prompt injection with and without the checker, for our system and every external agent, against CaMeL, Spotlighting and StruQ; rule-breaking actions proposed; launderers wrongly cleared; dual use and the weights decision.
- Target length: 1,200 words.

### 9 Failure analysis and case studies

- Owner: **T3** writes the 10 case studies (candidates chosen on validation by 31 Oct, test ones swapped in after G5); **T4** defines the error tags and writes the analysis. Status: not started.
- Sources: run logs from the final run (`ftm/eval/runlog.py`, `per_alert.jsonl`); `plan.md` section 9.3 (error tagging, 10 case studies); `docs/t3_first_runs.md` for the failure modes already seen (planner asks no accepted question; ReAct not finishing; the library collapsing onto one menu question).
- Contents: every error tagged with the step where it went wrong (intake miss, evidence view, planner, question refused or timed out, back-test, fusion, decision); failures by typology and by bank; where the LLM planner and the external agents fail; ten case studies, good and bad, with the question asked, the answer, the posterior path and the label. This section is required by the course brief.
- Target length: 2,000 words.

### 10 Limitations and future work

- Owner: **T4**. Status: draft 1 in `10_limitations.md` (threats to validity, limits of the design, future work; placeholders only where a test number is needed); the known limits are listed in `plan.md` section 15 and in the notes of each doc.
- Sources: `plan.md` sections 14, 15; `docs/t4_decisions_week2.md`; `docs/t1_baselines_val.md` (val look-ahead of the GNNs; Multi-GNN see-everything on few epochs); `docs/schedule_nov10.md` rule 6 (G2); each owner's "not done" lines in `STATUS.md`.
- Contents: synthetic data and a simulator; simulated banks and simulated private set intersection; protection unit is one transaction; approximate conformal and FCAC guarantees under drift; intake auto-clears most laundering at the alert budget; small number of positives on validation; reimplemented external systems whose published numbers cannot be checked on AMLworld; items cut or not run (cut order); future work.
- Target length: 800 words.

### 11 Compute and reproducibility

- Owner: **T4**; T3 supplies GPU-hours and serving details. Status: not started.
- Sources: `plan.md` sections 4.3, 11, 17; `docs/hardware.md`; `GPU_CALENDAR.md`; `docs/t3_week0.md`; `configs/llm/serving.yaml`; `prereg/`.
- Contents: hardware and serving stack; GPU-hours per machine against the budget; one final run with the test guard (`FTM_FINAL_RUN`); seeds; pre-registration hash; how each table is regenerated from a script and a config; what is released (code, simulator, frozen alert sets, harness) and what is not (trained weights).
- Target length: 700 words.

### References

`references.md`: [1]-[41] as in the proposal, [42]-[77] from `plan.md` section 18 and the repo's documents. Open CHECK items are listed in `README.md`.

## Numbers and figures the report needs

One row per placeholder. "Produced by" names the script or experiment (`plan.md` section 8 ids). "Status today" is as of 25 Oct 2026 and says what exists on validation, not on test. Nothing in the report may quote a test number that is not in this table. Row ids are used exactly as written in the text, as `{{TBD:<id>}}`.

| Placeholder | What | Produced by | Who runs it | Status today | Used in |
|---|---|---|---|---|---|
| `{{TBD:gap_closed_gfp}}` | Gap closed at ε = 2 against the GFP see-everything upper limit, on alerts and end to end, with 95% cluster-bootstrap CI (H1) | E2 and E7; final run `scripts/run_final.py` (a stub today) with `ftm/eval` (metrics, bootstrap, stats); denominator from `scripts/eval_report.py` | T3 hosts the LLM jobs, T4 does the statistics | Denominator exists on HI-Small val (one-bank 0.001167 against see-everything 0.000906 Brier); HI-Medium see-everything is blocked on compute (CPU job on S3, phase A). No system number: G2 is not met on validation (expert questions: HI-Small -0.0053 Brier; HI-Medium naive -0.0008, per-family +0.0129 with CI [-0.034, 0.064]); team decision due 27 Oct | 1.7, 7.4 |
| `{{TBD:gap_closed_sota}}` | Gap closed against the best state-of-the-art see-everything upper limit (Multi-GNN, plus others if they finish) | E7 with E1; `scripts/eval_report_models.py`, `scripts/baselines_val_table.py` | T1 (models), T4 (statistics) | Multi-GNN on our split for HI-Small only: see-everything 0.000720 against GFP 0.000830 Brier (9 of 100 epochs), one-bank 0.001110 against 0.001167 (28 of 30 bank views scored); HI-Medium see-everything on S3 overnight 28-30 Oct; FraudGT full run 28-30 Oct; MEGA-GNN in phase A | 1.7, 7.1, 7.4 |
| `{{TBD:tab_baselines}}` | Table of the one-bank and see-everything models on test (GFP, Multi-GNN, FraudGT, MEGA-GNN), with the reproduction checks | E1; `scripts/baselines_val_table.py`; `docs/t1_baselines_val.md` is the validation version | T1 | Validation tables done for both datasets (`docs/t1_baselines_val.md`); test jobs run first in the final run (about 6 h) | 7.1 |
| `{{TBD:n_alerts}}` | Alert counts per split, positives, intake targets against achieved miss rates, and the end-to-end intake miss rate | `ftm/intake` conformal intake; `docs/t1_handoffs.md` | T1 | HI-Medium val 2,238, train 10,962, projected test 3,475 alerts with 1,011 positives (a projection, from validation); intake auto-clears about 60% of laundering at alpha about 0.6 (`docs/t4_decisions_week2.md` item 2); final test count only after test intake | 4, 7.1 |
| `{{TBD:h2_ratio}}` | Share of the raw-record-sharing benefit that the questions reach (H2), with CI | E2 raw-share arm | T2 (arm), T4 (ratio and CI) | Arm built (three banks, exact, no noise, gated by `request_records`) and tested on the synthetic world; no real number | 1.7, 7.4 |
| `{{TBD:fig_privacy_curve}}` | Privacy-detection curve: gap closed against ε in {0.5, 1, 2, 4, no noise} for our system, raw sharing, one-bank and see-everything, with the theoretical ceiling drawn on the same plot | E2; `scripts/eps_sweep.py`; ceiling `ftm/eval/ceiling.py`, `scripts/ceiling_check.py` | T3 (lead), T2 (banks, sweep), T4 (statistics) | Ceiling check on T2's mechanisms passes (exact 35 of 35, empirical 33 of 35 with 2 inconclusive); sweep tooling done; no real curve (needs real back-test tables at ε 0.25-2 and the LLM runs) | 7.4, 8 |
| `{{TBD:planner_comparison}}` | LLM planner (compose, fixed list) against random, fixed order, EIG, learned policy and the enumerator; equivalence margin (H3) | E3; `scripts/run_ablations.py` (non-LLM), `scripts/run_llm_agent.py` (LLM) | T2 (non-LLM), T3 (LLM), T4 (statistics) | Non-LLM planners run on the synthetic world; G4 on real validation not run (needs `data/` and the three back-test tables); LLM planner on 20 HI-Medium val alerts only (Brier 0.115 against p0 0.082, descriptive, `docs/t3_first_runs.md`) | 1.7, 7.3 |
| `{{TBD:h4_useless_share}}` | Back-testing: Brier with and without it, share of accepted questions that are useless on later data, FDR audit, realistic against exact mode (H4) | E4; `scripts/backtest.py`, `ftm/backtest` | T2 (runs), T4 (analysis) | Realistic tables on validation: HI-Small 1 of 440 questions accepted, HI-Medium 23 of 440; exact-labels runs and comparison done; audit on later data needs the test period | 5.2, 7.2 |
| `{{TBD:tab_5a}}` | Table 5a: external systems (Groups A-D) end to end and questions-only, native and ported, on the 1,000-alert subset; top 2-3 on the full test set | E5; `scripts/run_llm_agent.py` over `ftm/external/*`; subset rule `prereg/subset_manifest.txt` | T3 (Group D), T4 (Group C core, assembly), T2 (Co-Investigator, FinFRE-RAG), T1 (Groups A and B numbers) | All ports written; Sharma and Torres only on the mock LLM (real runs need the endpoint); first real ReAct run on 20 val alerts finished 10 of 20, fixed afterwards; list locked in `prereg/external_list.yaml`; no table | 6, 7.4 |
| `{{TBD:h9_difference}}` | Our system against the best ported external agent on the same backbone, with CI (H9) | E5 with `ftm/eval/stats.py` | T4 | After G5 | 1.7, 7.4 |
| `{{TBD:tab_5b}}` | Table 5b: ablations (raw sharing, simple planners, enumerator, fixed list, no back-testing, LLM decides, fast-model options, evidence views, and the other switches) | E2-E4, E8-E11, E14; `scripts/run_ablations.py`, `configs/experiments/ablations_nonllm.yaml` | T2 (non-LLM arms), T3 (LLM arms) | Arms defined; real validation run missing | 6, 7.2 |
| `{{TBD:hypothesis_verdicts}}` | Table of H1-H10 with statistic, threshold, CI and verdict; Holm over the primary family, secondary corrected separately, exploratory uncorrected | `ftm/eval/stats.py`; `prereg/hypotheses.yaml` | T4 | Code and draft prereg done; thresholds to be frozen 30 Oct; nothing on real data | 1.7, 7.4 |
| `{{TBD:h6_pattern_edges}}` | Share of real pattern edges among the 40 shown, per evidence view, and the Brier effect (H6) | E8; `scripts/retrieval_demo.py`; `docs/t1_planner_view.md` | T1 (views), T3 (planner Brier arms), T4 (statistics) | Pattern-edge share, scorer plus PCST minus top-40 by amount: +0.064 [0.030, 0.104] on HI-Small and +0.061 [0.044, 0.079] on HI-Medium validation, not met under the pre-registered denominator; Brier arms not run | 7.1 |
| `{{TBD:h7_formula_vs_llm}}` | Formula against the LLM deciding (H7) | E9 | T3 (runs), T4 (fusion, statistics) | Not run | 7.3 |
| `{{TBD:h10_federated}}` | Private questions against DP-FedAvg and the AML-specific federated method at a comparable privacy level (H10), with the unit caveat | E6; `scripts/train_federated.py`, `scripts/train_layerwise.py`; `docs/t1_baselines_val.md` | T1 | HI-Small validation done (client-level and edge-level DP-FedAvg curves; layer-wise exchange: Sync-SGD with forward and backward exchange best, Brier 0.001116 against 0.001200 without exchange); HI-Medium not run (layer-wise exchange scheduled 28-30 Oct); Tian et al. licence answer pending | 7.1 |
| `{{TBD:e18_sweep}}` | Backbone size sweep: Brier, valid-DSL rate, latency and GPU-s per alert against model size, per family; the choice and the thinking arm | E18; `scripts/run_e18.py`; rule `prereg/e18_decision.yaml`; `docs/t3_e18.md` | T3 | Driver and tests done, dry-run plan done; sweep starting; backbone decision by 26 Oct | 5.3, 7.3 |
| `{{TBD:ft_planner}}` | Fine-tuned planner against the same model prompted, the teacher and the E18 backbone (E16) | `scripts/train_planner.py`; `docs/t3_finetune_fastmodel.md` | T3 | Distillation filter and QLoRA scripts written, not trained (Unsloth and the bf16 base missing at last report); scheduled 28-30 Oct | 7.3 |
| `{{TBD:fast_model}}` | Fast model: accuracy, calibration and latency per decision type, share kept by the cascade, accuracy against its bound (H5) | E10; `scripts/score_fast_baseline.py`, `ftm/fastmodel` | T3 (models), T2 (dataset, classifier baselines) | Untuned 4B/8B cascade baseline on all 13,792 validation decision rows (bound holds; the 4B keeps 0 to 49 percent); the 50,000-row set not built on real data; fine-tuned model not trained | 7.3 |
| `{{TBD:library_shape}}` | Strategy library filed by shape against by account, frozen against continual (E11) | E11; `ftm/library`; `docs/t3_library_draft.md` | T3 | Draft built on 1,983 train alerts, not frozen; 0 success and 4 failure notes | 7.3, 9 |
| `{{TBD:heldout_cycles}}` | Held-out cycles: detection, intake and accepted-question transfer for cycle alerts (E12) | `scripts/heldout_report.py`; `docs/t1_heldout_cycles.md` | T1 (T2 back-test, T4 fusion refit) | Done on validation for both datasets: cycle detection and question transfer do not degrade; fusion refit not done | 7.1 |
| `{{TBD:async_h8}}` | One question at a time against batched waves (2 then 1), benefit and simulated time (H8) | E13; `docs/t3_e13.md`; T2's SimPy runner | T2, T3 | LLM planner on 20 val alerts, descriptive; no real waves formed (budget too small for a second question) | 7.3 |
| `{{TBD:shuffled_answers}}` | Shuffled-answers test: the gain vanishes when answers are permuted across alerts | E14; fusion tests in `ftm/fusion` | T4 | Passes on the synthetic world; not run on real data | 7.2 |
| `{{TBD:fusion_calibration}}` | Calibration (ECE) of the fused posterior, summed against learned per-family weights | C8; `ftm/fusion`; `scripts/g2_check.py` | T4 | Fusion v0 done; on the live likelihood ratios the naive fusion moves p0 the wrong way (G2 not met); v1 per-family on the real back-test table is phase A | 5.4, 7.2 |
| `{{TBD:three_way}}` | Three-way decision (FCAC) against one calibrated threshold, with the achieved error rates (E17) | E17; `ftm/decision` | T4 | Threshold and stop rule done; FCAC port scheduled 28-30 Oct (stretch) | 7.4 |
| `{{TBD:inj_results}}` | Prompt-injection results: how often injected text gets an agent to propose a bad action, with and without the checker, for our system and every external agent; CaMeL, Spotlighting and StruQ comparison | E15; `scripts/make_injection_set.py`, `ftm/safety` | T4, with T3 for the fast-model screen | Set generator done (400 attacks, 400 benign, 4 types); E15 pipeline and CaMeL-style arms built, dry-run on the mock LLM only (no real number); real runs scheduled 28-30 Oct | 8 |
| `{{TBD:gate_rag}}` | Compliance gate: share of disallowed actions blocked in the hand-written suite; regulatory RAG: retrieval recall and citation accuracy on 100 cases, retrieved rules against whole corpus in the prompt | `tests/test_compliance_gate.py`, `scripts/rag_eval.py`, `ftm/compliance/rag_eval.py` | T4 (CPU) | Gate and 44-flag corpus done; the numbers are not recorded in the docs and can be produced on CPU now | 5.4, 8 |
| `{{TBD:typology_breakdown}}` | Every main metric per laundering typology and per bank | `ftm/eval` with the patterns file; `scripts/heldout_report.py` has the cycle slice | T4 | After G5; the pattern file is parsed, no by-typology result for the full system | 7.4, 9 |
| `{{TBD:fig_pareto}}` | Pareto plots: gap closed against ε, against LLM calls and against GPU-seconds; end-to-end against questions-only per external agent; native against ported | run logs (`ftm/eval/runlog.py`, `ftm/eval/runs.py`, `per_alert.jsonl`); timing replay on the reference machine | T4 (plots), T3 (timing) | Run-log v1 proposed; first timings exist (median 7.5 s per alert for the planner, 4.8 s for ReAct, Qwen3-8B); no cost data across systems | 7.4 |
| `{{TBD:error_tags}}` | Counts of errors by the pipeline step where they went wrong | tagging code over the final-run logs (to be written) | T4 | Tag list to be defined in phase B; not started | 9 |
| `{{TBD:case_studies}}` | Ten case studies, good and bad | candidates chosen on validation by T3, test ones swapped in | T3 writes, T4 reviews | Not started; selection rule to be agreed before the final run | 9 |
| `{{TBD:compute_table}}` | GPU-hours per machine and per item against the 250 h budget; tokens and LLM calls per system | `GPU_CALENDAR.md`, `docs/hardware.md`, run logs | T4 with T3 | Calendar kept weekly; final totals after G5 | 11 |
| `{{TBD:g2_decision}}` | Wording of the team's decision on gate G2 (reported as not met on validation, or changed in phase B), with the validation numbers it rests on | Team decision, due 27 Oct (`docs/schedule_nov10.md` rule 6); numbers from `scripts/g2_check.py` | Team; T4 writes the sentence | Open: validation gain of naive fusion -0.0053 (HI-Small) and -0.0008 (HI-Medium); fitted fusion +0.0007 and +0.0053 with CIs covering 0; not met | Abstract, 5.4, 7.4, 10 |
| `{{TBD:fusion_setting}}` | The fusion hyperparameters chosen by cross-validation on validation (mode, ridge, prior, folds, seed) and written to `prereg/` | `ftm/fusion` `fit_cv` through `scripts/g2_check.py --fusion per_family --fusion-select cv` (`fusion_chosen_all`) | T4 | Tooling done; the setting to pre-register is not yet recorded in `prereg/` | 5.4 |
| `{{TBD:fusion_diagnostics}}` | Share of the answers' evidence that p0 already holds (explained_by_p0, with CI), share of answers whose claimed and validation likelihood-ratio direction disagree, and the cells contradicted | `scripts/fusion_diagnostics.py` (`data/runs/fusion_diagnostics_<ds>.json`); `docs/t4_fusion_findings.md` | T4 | Run on real validation data by T4 (reading: shift or winner's curse); the values are not yet recorded in the docs | 5.4, 7.2 |
| `{{TBD:equiv_margin}}` | Equivalence margin for H3 and H10: half the Brier gap between random and the best simple planner, on the same alerts | `prereg/hypotheses.yaml` (`fix_on_val`); `ftm/eval/stats.py` | T4 | To be fixed on validation before G3 | 6.4, 7.3 |
| `{{TBD:hub_cap}}` | Hub cap for the cluster bootstrap on all rows (accounts in more rows than the cap stop linking clusters) | `ftm/eval/bootstrap.py` `cluster_ids`; `scripts/eval_report.py --hub-cap`; rule in `prereg/hypotheses.yaml` | T4 with T1 | Not needed on alerts (largest cluster 1.5% on HI-Medium validation); rows 58%, cap to be fixed on validation before test | 6.4, 10 |
| `{{TBD:n_askable_test}}` | Number and share of askable test alerts (partner bank among the 30 modelled views), with positives | `ftm/intake` after the test intake; `scripts/cap_sweep.py`-style counts | T1 | Validation: 87 (HI-Small eval), 160 and 460 (HI-Medium eval and all parts); test count only at G5 | 10 |
| `{{TBD:cut_items}}` | List of items cut or not run in the final run, and what each cut changes in the claims | `plan.md` section 14 cut order; `STATUS.md` | T4 with all | After G5 | 10 |
| `{{TBD:power_statement}}` | Power statement: detectable Brier difference at this run's alert count, alerts needed for a difference of 0.01 at the observed paired standard error (80% power, two-sided 5%), detectable difference at the projected test size of 3,475 alerts | `power.txt` and `status.json` (`power`) of `scripts/final_analysis.py`; `ftm/eval/final.py` `power_details` | T4 | Implemented and run on the synthetic world; on HI-Medium val the dry run gives about 0.0245 over 806 alerts, the needed-n line is new | 6.4, 7.4, 10 |
| `{{TBD:eps_infinity}}` | Brier gain against ε per alert including the no-noise (ε = ∞) row, with a one-line reading of whether noise is the limiter | `scripts/eps_sweep.py` (`data/runs/eps_sweep_hi_medium.json`), laid out by `scripts/final_analysis.py` (table `eps_infinity`, input `eps_sweep.path`) | T4 (table), T2 (sweep) | Sweep exists on HI-Medium val (EIG, naive fusion, 460 askable alerts): gain −0.0100 at ε 1, −0.0058 at ε 2, −0.0155 at ε 4, −0.0287 with no noise; table code tested on the synthetic world | 7.2, 7.4 |
| `{{TBD:review_capacity}}` | S1 (secondary, sensitivity, outside the Holm families): precision and recall at a fixed review capacity per system, with paired CIs against one bank | `scripts/final_analysis.py` (table `review_capacity`); capacity in `fixed_on_val.json`; `prereg/hypotheses.yaml` S1 | T4 | Implemented and run on the synthetic world; capacity (top 10% of alerts) is a placeholder to be pre-registered before the freeze; no real number | 7.4 |
| `{{TBD:adjusted_estimates}}` | S2 (sensitivity, unadjusted is primary): the paired Brier differences of H1 and H3 adjusted for p0 and the number of answered questions, with cluster-bootstrap CIs | `scripts/final_analysis.py` (table `adjusted_estimates`); `prereg/hypotheses.yaml` S2 | T4 | Implemented and run on the synthetic world; no real number | 7.4, 9 |
| `{{TBD:abstract}}` | Abstract | written last | T4 | Placeholder | Abstract |

## Order of work to 6 Nov

1. Now (25-27 Oct): skeleton, section 1 and 2 drafts (this PR); owners write 4 and 5 for their parts in phases A and B.
2. 28-30 Oct: T4 writes sections 3, 6 and 8 (they do not need test numbers), pre-builds every table and figure from validation runs (`docs/schedule_nov10.md` rule 4), and resolves the CHECK items in `references.md`.
3. 31 Oct-3 Nov: numbers arrive from the final run and are filled in place; no new text structure.
4. 4-6 Nov: results (7), failure analysis (9), limitations (10), compute (11), abstract; full draft on 6 Nov.
5. 7-10 Nov: slack, review, submit on 10 Nov.
