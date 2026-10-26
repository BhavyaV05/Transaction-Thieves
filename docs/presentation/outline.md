# Presentation outline

Owner: T4 (Shreyas). Draft 1, 27 Oct 2026, written before the final run. The presentation is part of the G6 hand-in on 10 Nov (`plan.md` section 17, `docs/schedule_nov10.md`); the draft is due with the report draft on 6 Nov, and 7-10 Nov is slack. The date and length of the talk are not recorded in the repo; this outline assumes about 15 minutes. The main talk below has 15 slides budgeted at 17.5 minutes, and the merges listed under the table bring it to 15.

Rules for the slides, taken from the report:

- No number on a slide that is not in a table of report section 7. Until the final run (G5, 31 Oct-3 Nov) every such number is the same `{{TBD:<id>}}` placeholder as in `docs/report/00_outline.md`, and it is replaced in place afterwards. Validation numbers may be shown only when the slide says "validation".
- The talk follows the course brief: problem, method, experiments, results, failure analysis, limitations. Slide 14 is the failure analysis the brief requires; the negative results (G2 on validation, the pattern-edge half of H6) are shown, not hidden.
- Figures are built on validation first (`docs/schedule_nov10.md`, rule 4) so that after the final run only numbers change.
- One message per slide, written as a sentence a listener could repeat.

## Main talk

| # | Title | Message (one line) | Figure or table | Produced by | Speaker | Min |
|---|---|---|---|---|---|---|
| 1 | Follow the Money Without Moving the Data | Catch cross-bank laundering by asking small, private, tested questions instead of sharing records | Title and team | none | T4 | 0.5 |
| 2 | Each bank sees one slice | A model that sees one bank catches less than one that sees everything, and the missing part is where the laundering is | Bar chart: Brier of one-bank against see-everything (GFP, Multi-GNN), HI-Small and HI-Medium, with CI | `scripts/eval_report.py`, `scripts/baselines_val_table.py` (the chart script is to be written); validation values in `docs/t1_baselines_val.md`; test version `{{TBD:tab_baselines}}` | T1 | 1 |
| 3 | The idea in one picture | The home bank asks other banks typed questions, each answered with one noisy value; a formula, not the LLM, decides | Figure 1 of the proposal, redrawn (the figure of report section 3) | `docs/proposal_v5.pdf` Figure 1; redraw by T4 | T4 | 1 |
| 4 | One alert, followed through | A question must pass the gate, the accountant and a back-test before it moves the probability; here is one end to end | The worked example of `docs/report/01_introduction.md` section 1.2, labelled "design numbers, not results" | by hand from the report text | T4 | 1 |
| 5 | Data and setup | AMLworld HI-Small and HI-Medium, a temporal split, 30 simulated banks, conformal alert intake | Table: alert counts per split with achieved intake miss rates `{{TBD:n_alerts}}` | `ftm/intake`; `docs/t1_handoffs.md`; `scripts/make_alerts.py` | T1 | 1 |
| 6 | Upper limits and baselines | The upper limits are real (Multi-GNN is a stronger one than GFP on our split) and intake auto-clears most laundering, so every result is reported at two levels | Table: one-bank and see-everything for GFP, Multi-GNN, FraudGT, MEGA-GNN `{{TBD:tab_baselines}}`; one line on the intake miss rate | `scripts/baselines_val_table.py`; `{{TBD:gap_closed_sota}}` | T1 | 1 |
| 7 | What a private question costs | A yes/no answer at budget eps can shift the odds by at most e^eps, and the ceiling is drawn on every privacy plot | Ceiling plot: empirical likelihood ratio against the e^eps ceiling | `scripts/ceiling_check.py`, `ftm/eval/ceiling.py` | T2 | 1 |
| 8 | Back-testing: questions must prove themselves | An LLM that invents thousands of questions cannot slip useless ones in, because e-value FDR control gates every one | Chart: questions proposed and accepted (validation: HI-Small 1 of 440, HI-Medium 23 of 440), useless on later data `{{TBD:h4_useless_share}}` | `scripts/backtest.py`, `ftm/backtest`; `docs/t1_handoffs.md` A4 | T2 | 1 |
| 9 | The LLM planner against the simple planners | Does inventing questions beat random, fixed, EIG and learned planners and the enumerator (H3)? | Bar chart of Brier by planner with the equivalence margin `{{TBD:planner_comparison}}`; backbone sweep `{{TBD:e18_sweep}}` as backup | `scripts/run_ablations.py`, `scripts/run_llm_agent.py`, `scripts/run_e18.py`; statistics from `ftm/eval/stats.py` | T3 | 1.5 |
| 10 | Fusion and decision: a pre-registered negative result | On validation, naive Bayes fusion made things worse (-0.0053 HI-Small, -0.0008 HI-Medium); the diagnostics read as shift or winner's curse, not double counting | Table of fusion gains with CIs (report section 5.4.4); one example, `dormant(ALERT.receiver, days=3)`: claimed LR 0.49 against validation LR 1.15; G2 wording `{{TBD:g2_decision}}` | `scripts/g2_check.py`, `scripts/fusion_diagnostics.py`; `docs/t4_fusion_findings.md` | T4 | 1.5 |
| 11 | Same rules for every agent | External agents get the same alerts, banks, gate, budget, backbone and caps; two scores separate asking from deciding | Diagram of the shared layers (gate and accountant, investigation, budgeted client) next to the Table 5a systems | `prereg/external_list.yaml`; `docs/external_code_check.md` | T4 | 1 |
| 12 | Headline results | How much of the gap the questions close against both upper limits, and against raw sharing and the best external agent | Forest plot of gap closed with 95% CI `{{TBD:gap_closed_gfp}}`, `{{TBD:gap_closed_sota}}`, `{{TBD:h2_ratio}}`, `{{TBD:h9_difference}}`; privacy-detection curve `{{TBD:fig_privacy_curve}}` | final run; `scripts/eps_sweep.py`, `ftm/eval/stats.py`; the plot script is to be written | T4 | 2 |
| 13 | Safety: gate and injection | The gate blocks every disallowed action in the hand-written suite; injected text in replies is data, not instructions | Table: share blocked `{{TBD:gate_rag}}`; actions triggered by injection with and without the checker, for our system and each external agent `{{TBD:inj_results}}` | `tests/test_compliance_gate.py`, `scripts/make_injection_set.py`, `scripts/rag_eval.py` | T4 | 1 |
| 14 | Where it fails | Every error is tagged with the pipeline step where it went wrong; here are the common ones and one case study | Bar chart of errors by step `{{TBD:error_tags}}`; one case study with question, answer, posterior path and label `{{TBD:case_studies}}` | tagging code over the final-run logs (`ftm/eval/runlog.py`; the tagging script is to be written); T3 picks the case | T3 | 1.5 |
| 15 | What we claim, and what we do not | Synthetic data, simulated banks, a guarantee per transaction, small samples; the hypothesis verdicts are reported as they fall | Table of H1-H10 verdicts with Holm `{{TBD:hypothesis_verdicts}}`; the main limits as one line each | `ftm/eval/stats.py`; `prereg/hypotheses.yaml`; report section 10 | T4 | 1 |

To fit 15 minutes, merge slides 3 and 4 into one and slides 5 and 6 into one; slide 9 or slide 13 can move to the backups if the final run changes what is worth saying. The merge decision is taken at the rehearsal.

## Backup slides (for questions)

| # | Title | Content | Source | Owner |
|---|---|---|---|---|
| B1 | Per-typology breakdown | Gap closed and detection per laundering pattern and per bank `{{TBD:typology_breakdown}}` | `ftm/eval` with the patterns file | T4 |
| B2 | Pareto plots | Gap closed against epsilon, LLM calls and GPU-seconds; native against ported `{{TBD:fig_pareto}}` | run logs, `ftm/eval/runlog.py`, `ftm/eval/runs.py` | T4, T3 |
| B3 | Federated training against questions | DP-FedAvg and the layer-wise method on the same axes with both privacy units labelled `{{TBD:h10_federated}}` | `scripts/train_federated.py`, `scripts/federated_eps_figure.py`; `docs/figures/t1_federated_eps_hi_small.png`, `docs/figures/t1_federated_eps_hi_medium.png`; `docs/t1_federated_eps.md` | T1 |
| B4 | Evidence views and retriever | Pattern edges among the 40 for the four views; the pre-registered 15-point target is not reachable with about 3 pattern edges per window `{{TBD:h6_pattern_edges}}` | `scripts/retrieval_demo.py`; `docs/t1_planner_view.md` | T1 |
| B5 | Held-out cycles | Detection and question transfer when cycles are removed `{{TBD:heldout_cycles}}` | `scripts/heldout_report.py`; `docs/t1_heldout_cycles.md` | T1 |
| B6 | Fast model and cascade | Accuracy against its bound, share kept, latency `{{TBD:fast_model}}` | `scripts/score_fast_baseline.py` | T3 |
| B7 | Fine-tuned planner and library | `{{TBD:ft_planner}}`, `{{TBD:library_shape}}` | `scripts/train_planner.py`, `scripts/build_library.py` | T3 |
| B8 | Formula against LLM deciding; shuffled answers | `{{TBD:h7_formula_vs_llm}}`, `{{TBD:shuffled_answers}}` | `ftm/fusion` tests | T4, T3 |
| B9 | Compute | GPU-hours per machine against the budget `{{TBD:compute_table}}`; one final run, seeds, pre-registration hash | `GPU_CALENDAR.md`, `docs/hardware.md` | T4 |
| B10 | Compliance gate and retrieval | The six `ask_bank` checks in order; retrieval recall and citation accuracy `{{TBD:gate_rag}}` | `ftm/compliance`, `scripts/rag_eval.py` | T4 |

## Preparation

- **Now to 30 Oct:** build every figure and table of the main talk on validation with its script, so that after the final run only numbers change (rule 4 of the schedule). Scripts marked "to be written" are needed for slides 2, 12 and 14.
- **4-6 Nov:** fill numbers from the report tables, cut to 15 minutes, one rehearsal with a timer. The wording of slides 10 and 15 on G2 follows the team's decision `{{TBD:g2_decision}}`.
- **7-10 Nov:** slack for changes after review. Each speaker checks that every number on their slides matches the report table named in the "Produced by" column.
