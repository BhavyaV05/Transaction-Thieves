# 6 Experiments and baselines

*Owner: T4 (Shreyas), with each experiment lead supplying the setup of their rows. Draft 1 (27 Oct 2026): the evaluation protocol, the statistics, the fairness protocol for external agents and the pre-registered hypotheses. It contains no result; results are in section 7. Systems that are listed here from `prereg/external_list.yaml` are marked draft there and are locked at gate G3. Remove this line before the final build.*

The course brief asks for baselines and ablations, a small carefully designed evaluation set, capability-specific metrics and no reliance on an unvalidated LLM judge. This section says what is compared (6.1), how every external agent is made comparable (6.2), what is measured (6.3), how uncertainty is handled (6.4), what was predicted in advance (6.5) and how the test set is protected (6.6). Every score is computed by code against the dataset's own labels; nothing is graded by an LLM.

## 6.1 What is compared

The comparison is in two tables, as in the plan (`plan.md` section 7). **Table 5a** holds external systems, in four groups:

| Group | Systems (tier) | Built by |
|---|---|---|
| A, detection models, each as the one-bank start and as a see-everything upper limit | Multi-GNN [42] (core); FraudGT [43], MEGA-GNN [44] (secondary) | T1, official code where the licence allows |
| B, privacy-preserving cross-bank training | DP-FedAvg GNN with 30 clients (core); the AML-specific federated method of Tian et al. [8] (secondary) | T1 |
| C, AML and fraud LLM agents, each native and ported | Torres et al. [6] and Sharma [3] (core); Co-Investigator AI [45] and FinFRE-RAG [28] (secondary) | T4 (Torres, Sharma), T2 (the rest) |
| D, general agent frameworks and information-seeking agents, ported | ReAct [49], Reflexion [50], ReWOO [51] (core); LATS [52], LLMCompiler [53], DSPy-tuned ReAct [61], multi-agent debate [55], Uncertainty of Thoughts [57], BED-LLM [10] (secondary) | T3 (ReAct, Reflexion, ReWOO, LATS, LLMCompiler, DSPy), T4 (debate, UoT, BED-LLM) |

Stretch systems (`plan.md` section 14) are run only if time allows and are listed as such. **Table 5b** holds our own ablations and reference points, and none of the proposal's ablations was removed when the external comparisons were added: one bank only; see everything; sharing raw records among three banks with no noise; the simple planners (random, fixed order, expected information gain, learned policy); the enumerator of all expressible questions; the LLM limited to a fixed list; no back-testing; the LLM deciding instead of the formula; the fast-model options; the four evidence views; and the other switches (realistic against exact back-testing, rules retrieved against the whole corpus, library filed by shape against by account, the privacy sweep, the per-account cap, held-out cycles, one question at a time against waves, refusals and timeouts, shuffled answers, fixed range against conformal, summed against learned fusion). The experiments are E1 to E18 (`plan.md` section 8); the hypotheses each answers are in 6.5. The numbers are in {{TBD:tab_5a}} and {{TBD:tab_5b}}.

## 6.2 Fairness protocol for external agents

An external agent can look better or worse than our system for reasons that have nothing to do with its method: a larger model, more calls, more privacy budget, a lighter rule check, a different set of alerts, or a prompt tuned for it. The protocol removes each of these in code, so that the only thing that differs is the control loop and the prompts the original paper specifies.

**Same inputs.** Every system gets the same alerts, the same simulated partner banks, the same question language, the same compliance gate and accountant, the same ε per alert (2) and per-account cap, and the same backbone model, the one chosen by the backbone sweep E18 (`prereg/external_list.yaml`: `backbone: e18`, `thinking: off`). Sampling is at temperature 0 by default. Departures are written next to the system: Sharma keeps its official 0.1, and the debate agents run at 0.7 and differ by seed, because at 0 they would be identical. FinFRE-RAG's paper uses 0.6 and we use 0 for fairness (`prereg/external_list.yaml`, `fixed_choices`).

**How the sameness is enforced, not promised.** Four layers are shared by every LLM system:

- *The harness toolbox and the gate.* A system reaches a partner bank only through `ask_bank` in the harness toolbox, which runs the compliance gate (section 5.4.1), then the privacy accountant, then the clock. The toolbox keeps the authoritative log of the questions asked and replaces the system's own account of them, so a system cannot under-report its questions or its budget.
- *The investigation layer* (`ftm/llm/investigation.py`). One object per alert tracks ε left and simulated time, refuses a repeat of the same question to the same bank and any question above the ε left (no call is made), turns each answer into the posterior through the same fusion, and assembles the verdict with its cost fields. Every Group D port and our planner use it, so a loop cannot get extra information by asking in another way.
- *The budgeted client* (`ftm/llm/budget.py`, `BudgetedClient`). Every LLM call is made through one wrapper per alert, with a cap of 20 calls, a prompt cap of 6,000 tokens and 5 simulated days (`configs/llm/default.yaml`, `Limits`). The same wrapper fills the cost fields (calls, tokens in and out, GPU-seconds), so cost is counted in one place and a system that would exceed the cap is stopped, not allowed to run over. For this reason AG2/AutoGen is not used for the debate port: its own LLM calls would bypass the wrapper.
- *The untrusted-text wrapper* (`ftm/llm/untrusted.py`). A note in a partner's reply is truncated to 300 characters, escaped and wrapped, for every system alike, so the safety comparison of section 8 does not start from different quoting.

**Two scores for every LLM system.** *End to end* is the system's own final probability; if it returns none (a loop that did not finish, an unparseable output), the alert is scored with p0 and the rate of such alerts is reported per system (the no-verdict rate), so a system is never rewarded or hidden by failing silently. *Questions only* ignores the system's final answer and computes p0 plus its answers through our own fusion (`ftm/eval/runs.py`): only answers with an accepted back-tested likelihood ratio count and the agent's own words never do. The difference between the two separates asking better questions from making better final calls, which is the central question for the Group C agents.

**Native and ported.** A domain agent runs as published (*native*: the one-bank view, no partner banks, its own tools) and given our question tool with the same ε (*ported*). The native score says what the published method does on this benchmark; the ported score says what it does when it can ask. H9 is tested on the ported scores.

**Tuning parity.** We fine-tune our planner on validation data, so the strongest tuned external agent receives the same: DSPy-optimised ReAct uses MIPROv2 with Brier as the objective on the same validation alerts as our fine-tuning (`prereg/external_list.yaml`). Every reimplementation fixes its open settings before its first run (`fixed_choices` in the same file), and none is tuned after results are seen. The best external agent for H9 is chosen on validation at gate G4, not on test.

**Evaluation set.** All external systems run on a frozen subset of 1,000 test alerts. The *rule* is frozen at G3: stratify by home bank and p0 decile (both known at alert time, so no labels are needed), proportional allocation, seed 0 (`prereg/subset_manifest.txt`; `docs/t4_decisions_week2.md`, item 1). The alert ids are drawn at G5, before any system runs, and their hash is written to the run log. The two or three best external systems (chosen on validation) and our own system and ablations also run on the full test set, so subset numbers are directly comparable across rows. A 1,000-alert subset has less power than the full set (6.4), and we say so wherever a subset number is reported.

**Honesty about what is and is not ported.** Torres [6], Co-Investigator AI [45], FinFRE-RAG [28] and BED-LLM [10] are reimplementations because no code was released; debate [55] and UoT [57] because their repositories have no licence; ReAct, Reflexion and ReWOO are reimplementations of the published loops inside our harness; LATS and LLMCompiler adapt official MIT code; Sharma [3] is a port of MIT-licensed code (`docs/external_code_check.md`, `prereg/external_list.yaml`). The published numbers of a reimplementation cannot be checked on AMLworld, since Torres' paper uses another dataset. A reimplementation can therefore be weaker than the original, and Table 5a marks each row as official, adapted or reimplemented. A strong external agent that beats our planner is still a result we report.

## 6.3 Metrics

The headline is the **Brier score** (lower is better), and **gap closed** turns it into a percentage (6.4). The full set follows `plan.md` section 9.2, with at least one metric for each capability, as the brief requires.

| Area | What is reported |
|---|---|
| Detection | Brier, gap closed, balanced accuracy, precision-recall curve; launderers caught at a fixed daily review capacity; correct calls flipped to wrong and the reverse; auto-decision error against its target |
| Privacy and cost | Gap closed per unit ε with the theoretical ceiling drawn; ε per alert and per account; ε_cal; records shared; questions per alert; LLM calls, tokens and GPU-seconds per alert |
| Comparison | Pareto plots: gap closed against ε, against LLM calls and against GPU-seconds; end to end against questions only per external agent; native against ported |
| Speed and tools | Simulated time to decision; questions that parse and run; recovery after timeouts, refusals and broken replies |
| Reasoning | New questions that pass back-testing; passed questions that are useless on later data (likelihood ratio at most 1 among those asked at least 30 times); stated against real confidence; agreement of the planner's stated hypothesis with the question, checked by two teammates on 100 steps (the only human judgement in the study) |
| Retrieval | Real pattern edges among the 40 shown; one connected trail; library reuse; cited rules that exist and apply |
| Fine-tuning and fast model | Planner against its teacher; fast-model accuracy, calibration and speed per decision type; share kept by the cascade; accuracy against its bound |
| Safety | Rule-breaking actions proposed; actions triggered by injected text; launderers wrongly cleared |
| Breakdown | Every main metric per laundering typology and per bank |

Every error is tagged with the pipeline step at which it went wrong (section 9). The scores are reported at **two levels side by side**: *on alerts* (the alerts the agent actually sees) and *end to end over all rows* (alerts plus the rows that conformal intake auto-cleared), with the intake miss rate printed next to every number. This is a decision of week 2 (`docs/t4_decisions_week2.md`, item 2): at the alert budget the intake auto-clears most laundering, so gap closed on alerts alone would overstate what a bank recovers.

## 6.4 Statistics

**Gap closed** is [Brier(one bank) - Brier(system)] / [Brier(one bank) - Brier(upper limit)]: 0 is the bank's own model and 1 is the upper limit. It is reported against both upper limits, the dataset authors' graph-feature model with gradient boosting and the best state-of-the-art see-everything model, because a weaker upper limit flatters the result. If the denominator is below 1e-5, gap closed is not reported and raw Brier differences are given instead (`prereg/hypotheses.yaml`, `ftm/eval/metrics.py`). Numerator and denominator are computed on the same rows, so one paired resample covers both.

**Confidence intervals** are 95% **cluster-bootstrap** intervals with 10,000 resamples and seed 0 (`ftm/eval/bootstrap.py`). Alerts that belong to the same laundering attempt, or are linked through a shared account, are not independent, so a row-level bootstrap gives intervals that are too narrow. Rows are therefore joined into clusters (attempt plus shared accounts, by union-find), and whole clusters are resampled. Every headline quantity is a function of per-row means over the same rows, so a paired resample needs only per-cluster sums. On HI-Small validation alerts the largest cluster holds 3.0% of the rows, and the Brier-gap interval moves little between clustering and row-level resampling ([0.0104, 0.0748] against [0.0153, 0.0665]; `docs/t4_decisions_week2.md`, item 6). Over all rows the giant component grows (16.3% on HI-Small validation; for HI-Medium rows the largest share is 58%, against 1.5% on alerts; `STATUS.md`, week 7), so the pre-registered rule is: if the largest cluster passes 20% on the HI-Medium validation alerts, accounts that appear in more than a fixed number of rows stop linking (a hub cap), and the cap is fixed before test ({{TBD:hub_cap}}). Headline numbers are on alerts, where clusters are small and the rule is not triggered (1.5%); for the row-level, end-to-end numbers the hub-capped version is reported as a sensitivity row.

**Multiple testing.** The primary family is H1, H2, H3, H4 and H9, corrected by Holm's step-down procedure at 0.05. The order H1, H2, H3, H4, H9 is only the reporting order: Holm orders by p-value, and H9 is last so that it cannot affect the others. If the TA prefers a pre-specified order, fixed-sequence testing is implemented as well (`ftm/eval/stats.py`); the pre-registration names Holm. The secondary hypotheses (H6, H7, H10) form their own Holm family. The exploratory ones (H5, H8) are not corrected and are labelled "uncorrected, exploratory", and we make no significance claim for them.

**Equivalence margin.** "As good as" is never read from a non-significant difference. Two planners are equivalent only if the whole confidence interval of their Brier difference lies inside the margin, and non-inferiority needs only the upper bound below the margin. The margin is half the Brier gap between the best simple planner and random question choice, measured on the same alerts ({{TBD:equiv_margin}}; fixed on validation, never on test).

**A missing verdict is p0.** An alert for which a system returns no verdict is scored with the bank's own probability and counted in the system's no-verdict rate (agreed between T3 and T4, `prereg/hypotheses.yaml`); it is neither dropped nor counted as an error.

**Power.** With about 4,000 test alerts the study can detect Brier improvements of about 0.011; on the 1,000-alert subset, about 0.022 (`plan.md` section 9.3, design values). The number of positives is the real limit, and several validation intervals are wide for this reason (section 10).

**Where data is used.** All tuning happens on validation: the backbone sweep, the fusion weights (cross-fitted inside validation, because the HI-Small validation `eval` part has only 49 positive alerts) and the G2 check. The hypotheses' thresholds that are marked `fix_on_val` are set from validation and committed before the test set is touched. The evaluation set is the test alerts from conformal intake; the headline is HI-Medium test days 13 to 16, and the sparse tail after day 16 is reported separately and never pooled (`prereg/hypotheses.yaml`).

## 6.5 Hypotheses

Each hypothesis is a prediction that can fail, with a statistic and a rule written in `prereg/hypotheses.yaml` before the test set is touched. The thresholds below are the draft values of `plan.md` section 2; the entries marked "on validation" are fixed from validation before gate G3, and any change to a primary hypothesis after G3 goes back to the TA.

| H | Tier | Prediction | Statistic and rejection rule | Experiment |
|---|---|---|---|---|
| H1 | primary | At ε = 2 per alert, the questions close at least 30% of the gap | gap closed against the see-everything model; reject the null if the CI lower bound is above 0 and the point estimate is at least 0.30 | E2 |
| H2 | primary | The questions reach at least 70% of the benefit of sharing raw records (three banks, no noise) | (Brier one bank - Brier ours) / (Brier one bank - Brier raw sharing); reject if the CI lower bound is at least 0.70 | E2 |
| H3 | primary | The LLM composing questions is as good as the best simple planner and the enumerator, and better than itself limited to a fixed list | two non-inferiority bounds against the margin and one superiority bound against the fixed list; all three must hold (intersection-union test) | E3 |
| H4 | primary | Back-testing matters: results are worse without it, and under 10% of accepted questions are useless on later data | superiority over no-back-testing (CI upper bound below 0) and the upper 95% bound of the audit share below 0.10 | E4 |
| H9 | primary | At ε = 2 our system closes at least as much of the gap as the best ported external agent on the same backbone | gap closed of ours minus the best ported agent (chosen on validation); reject if the CI lower bound is above minus the margin (set on validation); end-to-end is the test, questions-only is reported | E5 |
| H6 | secondary | The retriever shows at least 15 points more real pattern edges among the 40 than the baseline view and improves results | difference of pattern-edge shares with CI lower bound at least 0.15 and Brier CI upper bound below 0 | E8 |
| H7 | secondary | A calibrated formula deciding beats the LLM deciding | Brier(ours) - Brier(LLM decides) with the CI upper bound below 0 | E9 |
| H10 | secondary | Private questions match or beat DP federated training at a comparable privacy level | Brier(ours) - Brier(DP-FedAvg) with the CI upper bound below the margin; the privacy units differ (training-time ε against per-question ε) and both are labelled on the plot | E6 |
| H5 | exploratory | The fast model handles most routine decisions within its accuracy bound, at least 10 times faster, with no loss in detection | P(answer accepted and wrong) at most twice the conformal level per decision type; error given accepted is reported but not tested | E10 |
| H8 | exploratory | Two questions at once, then one more, keeps 80% of the benefit in 60% of the time | the two ratios, descriptive | E13 |

Two of these predictions are already known to be in trouble on validation, and the report says so where the results are given: the pattern-edge half of H6 (the pre-registered 15-point difference is not reachable with about 3 pattern edges per window; +0.064 [0.030, 0.104] on HI-Small and +0.061 [0.044, 0.079] on HI-Medium, `STATUS.md`, week 7) and the validation check G2 that H1 builds on (section 5.4.4). Pre-registration is meant to make such outcomes reportable and not to hide them. The verdicts after the final run are {{TBD:hypothesis_verdicts}}.

## 6.6 Pre-registration and the test lock

The test split is locked in code: it loads only when `FTM_FINAL_RUN=1` is set, which happens at gate G5 only (`ftm/guards.py`). Every tool query takes an `as_of` time and sees only data at or before it. The pre-registration files in `prereg/` (hypotheses, backbone-decision rule, external list, subset rule) are committed before the final run, and the commit hash is sent to the TA before G5. Results that are not on the pre-registered list are labelled exploratory. There is one final test run, and it runs every experiment; the ablation and external comparisons are then results, not choices.
