# 9 Failure analysis and case studies

*Owner: T4 (Shreyas) writes the error analysis (9.1 to 9.3); T3 (Akshat) writes the ten case studies (9.4). Draft 1 (28 Oct 2026), written before the final run: it defines the error tags and the way case-study candidates are chosen, and it reports the validation findings that explain why gate G2 is not met. Those findings are descriptive validation results, not test results. Test paragraphs are placeholders. Numbers from a script are `{{TBD:<id>}}` (convention in `docs/report/README.md`). Remove this line before the final build.*

The course brief asks for a failure analysis. This section says where in the pipeline a wrong call went wrong (9.1), how the worst and best alerts are picked for reading (9.2), why the fused answers did not beat the one-bank score on validation (9.3) and, written by T3, the ten case studies (9.4).

## 9.1 Error tags

A **wrong call** is an alert whose end-to-end probability is on the wrong side of 0.5 (p ≥ 0.5 against the label). Every wrong call receives exactly one tag, the **first step of the pipeline that failed**, so that an alert is counted once and the counts add up (`ftm/eval/final_tables.py`, `tag_errors`). The tags run in pipeline order, and the first condition that holds decides.

**Table 9a.** Error tags, in the order they are tested.

| Tag | Step | Holds when |
|---|---|---|
| `no_verdict` | verdict | the system returned no probability (scored as p0) |
| `no_question_asked` | planner | the verdict has no question step |
| `all_blocked` | gate / accountant | questions were asked and none was allowed |
| `no_reply` | partner bank reply | questions were allowed and every one ended in a decline or a timeout |
| `no_backtested_lr` | back-test | replies came, but none had an accepted back-tested likelihood ratio, so fusion could not use it |
| `verdict_override` | verdict | the system's own probability is more than 0.05 farther from the label than the questions-only fusion of its answers |
| `fusion_moved_wrong` | fusion / evidence | the fused answers are farther from the label than p0 |
| `answers_insufficient` | evidence too weak | none of the above: the answers were used correctly and did not move the score across the line |

The table of counts per system, variant, mode and alert set is {{TBD:error_tags}}. Two steps of the pipeline are not tags. A **conformal intake miss** (a launderer the one-bank model auto-clears) never reaches the agent; it is counted with the intake miss rate beside the end-to-end numbers (section 7.1, {{TBD:n_alerts}}). The **evidence view** is not tagged because no run log records which view led to a mistake; its effect is measured as an arm (H6, {{TBD:h6_pattern_edges}}). The tags say where the chain failed first, not why. The same counts are cut by laundering typology and by home bank ({{TBD:typology_breakdown}}), and by system, with the stop reason of each alert beside them, so that where the LLM planner and each external agent fail can be read off in one place.

Failure modes seen so far on 20 validation alerts, before any final run (`docs/t3_first_runs.md`, descriptive): the planner wrote invalid programs (14 syntax errors and 4 invented functions among 18), repeated questions, and in two alerts looped on refusals until it used all 20 calls; 10 of 20 ReAct runs ended without `finish` (a timeout past the simulated five days, a loop on gate refusals, or the prompt cap) and were scored as p0. Of the refusals, 36 of 79 for the planner were a harness mismatch (the prompt listed banks the harness does not model), fixed afterwards. The strategy library collapsed onto one menu question, so it holds no success note (`docs/t3_library_draft.md`).

## 9.2 How the case studies are selected

The final analysis writes a table of **case-study candidates** for our system (`case_studies`, {{TBD:case_studies}}), by a rule that does not look at the story of any alert. The **errors** are the six alerts with the largest Brier contribution, (p − y)². The **recoveries** are the four alerts where p0 was on the wrong side of 0.5 and the final probability was not, ordered by how much Brier the questions gained over p0; if fewer than four exist, the largest gains fill the list and are marked as such. Each row gives the alert, home bank, typology, label, p0, both probabilities, the questions asked and answered, the stop reason and the error tag.

T3 reads the question chain of each candidate (the question, the partner's answer, the posterior after each step and the label) and writes ten studies, good and bad. Candidates are chosen on validation by 31 Oct so that the format is settled; after the final run the test candidates are swapped in by the same rule. Worst-and-best selection shows what failures look like, not how often they happen: frequency is the job of Table 9a. [CHECK: T3 and T4 to agree before the final run which of the ten candidates are written up, so that the choice is fixed in advance and not made by reading the labels]

## 9.3 Why G2 is not met: validation findings

Gate G2 asked that the expert questions improve Brier by at least 0.01 over p0 on the validation `eval` part. They did not (section 5.4.4). What follows is the explanation as far as validation can give it. **Everything in this sub-section is a validation result on small samples (87 askable alerts for HI-Small, 160 for HI-Medium `eval`, 460 for all HI-Medium validation parts). None of it is a test result. The G2 numbers are the pre-registered check; the rest was computed afterwards to explain the failure, is descriptive and was not pre-registered.** The sources are `docs/t4_fusion_findings.md` section 6 and GitHub issue #84 (the prototype of a replication filter, T4); the write-up in section 7 of that file is to replace the issue as the citation. [CHECK: once section 7 of docs/t4_fusion_findings.md has these results, cite it instead of issue #84]

**Naive and fitted fusion.** Summing the back-tested log-likelihood ratios moves p0 the wrong way: the gain over p0 is -0.0053 on HI-Small and -0.0008 on HI-Medium. Fusion with per-family weights chosen by cross-validation reaches +0.0007 and +0.0053, and the interval of each gain spans 0 (HI-Small [-0.015, +0.019], HI-Medium [-0.041, +0.054]). Neither fusion shows a gain, and with these intervals the validation data cannot say whether the answers carry reliable information.

**The back-test picks partly by luck.** Back-testing with exact labels and no noise (`data/runs/a4_compare_hi_medium_labels.json`) gives an independent view of which questions are strong. Of 424 matched candidates (question and ε pairs) on HI-Medium, 23 are accepted with realistic noise and 64 with exact labels; only 9 are accepted by both, and the rank correlation of their strengths is 0.12. Acceptance under noise is a weak guide to acceptance with exact labels.

**Noise is not the limiter.** In the privacy sweep (naive fusion, expected-information-gain planner, all 460 HI-Medium validation alerts) the gain is -0.0100 at ε = 1, -0.0058 at ε = 2, -0.0155 at ε = 4 and -0.0287 with no noise (`docs/t1_handoffs.md`, A4). More budget and less noise make the result worse, which is the opposite of what a noise-limited system would do. The planner also asks more questions as the budget grows (0.94, 1.56, 2.22 and 6.0 per alert), which is consistent with more of the claimed likelihood ratios being applied and the loss growing with them.

**History-based questions are confounded by time.** The recomputed back-test of `dormant(ALERT.receiver, days=7)` at ε = 2 on train matches the cache exactly, and its pooled log-likelihood ratio for the answer `true` is -0.92. Split in time order into two halves, the same answers give +0.02 in the first and -0.15 in the second. The launder rate jumps from 5% to 37% between the halves and the share answering `true` falls from 88% to 15%, most likely because accounts have little history early in the ledger (a warm-up; not tested). The pooled ratio is a Simpson's-paradox artefact: both the label and the answer drift with time, and inside each half the question says almost nothing. The label-permutation e-values that accept a question [37] assume labels are exchangeable given the answers, which fails under such drift. The same check across all 23 accepted questions gives:

- 12 keep the sign of their log-likelihood ratio in both halves of train and 11 do not (the non-replicating are the dormant, new-account, repeated-counterparty, new-counterparty and volume-spike questions);
- on validation `eval`, the train direction agrees for 10 of the 12 replicating questions and for 4 of the 11 others (14 of 23 overall; cells of 5 to 81 answers);
- the mean |log LR| is 0.82 as claimed on pooled train, 0.46 with a time-stratified (Mantel-Haenszel, four blocks) estimate and 0.50 observed on validation, so the stratified estimate is the one that matches.

**A lead, not a result.** If the 11 non-replicating questions are removed (their e-value set to 1, so the number of hypotheses in the e-BH step is unchanged, 12 questions accepted), the naive gain turns positive on both independent validation parts.

**Table 9b.** Naive-fusion gain over p0 in Brier on HI-Medium validation, with and without the replication filter (95% cluster-bootstrap interval; 2,000 resamples). Descriptive; every interval includes 0.

| Part | All 23 accepted questions | Replicating questions only (12) |
|---|---|---|
| `eval` | -0.0008 [-0.013, +0.011] | +0.0073 [-0.003, +0.018] |
| `cal` | -0.0013 [-0.017, +0.014] | +0.0086 [-0.001, +0.018] |

This does not meet G2: every interval includes 0, the `eval` part was looked at for two variants, and the filter was not in the pre-registration. It is a hypothesis for the final run, not a claim: that removing questions whose evidence does not replicate in time helps. The acceptance rule is T2's code (`ftm/backtest/engine.py`) and the team's decision on whether to change it before the freeze, or to report the confound and keep G2 as not met, is {{TBD:g2_decision}}.

**What this says for the failure analysis.** The validation evidence points to a failure at the back-test and fusion steps, not at the planner or the banks: the questions that were accepted do not keep their claimed strength on later data. If that reading is right, the test tags `fusion_moved_wrong` and `answers_insufficient` should outnumber `no_question_asked` and `all_blocked` for our system, and the history-based question families should account for most `fusion_moved_wrong` alerts. That is a prediction to check against Table 9a, not a finding.

## 9.4 Case studies

*Written by T3 (Akshat).* Ten case studies, good and bad, each with the alert, the questions asked, the answers, the posterior path, the label and the error tag: {{TBD:case_studies}}.
