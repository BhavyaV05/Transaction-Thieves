# Abstract

*Owner: T4 (Shreyas). Skeleton written on 27 Oct 2026, before the final run: the structure and the sentences that do not depend on test numbers are fixed, every number is a placeholder. Fill it last (6 Nov), from the tables of section 7; it must contain no number that is not in one of them. Remove this line and the placeholder list below before the final build. Placeholder for the whole paragraph in the numbers table of `00_outline.md`: `{{TBD:abstract}}`.*

Each bank sees only the transactions that touch its own customers, so a model trained on one bank's slice of the data misses laundering that is routed through other banks on purpose [1]. Pooling the data is not allowed, and answering another bank's request usually means sending raw records. We study a narrower option: for the alerts its own model cannot decide, an agent at the home bank asks other banks small, typed questions that each bank answers exactly on its own ledger and returns as one differentially private value. Every question, including those an LLM invents, must first prove a likelihood ratio on past alerts with error-controlled back-testing; every action passes an allowlist compliance gate; and a formula, not the LLM's text, makes the final decision. We evaluate on IBM's synthetic AMLworld data (HI-Small and HI-Medium) with simulated partner banks, against two see-everything upper limits, raw record sharing, simple question planners, differentially private federated training and existing AML and general LLM agents run under one fairness protocol. At a privacy budget of 2 per alert, the questions close {{TBD:gap_closed_gfp}} of the gap to the graph-feature upper limit and {{TBD:gap_closed_sota}} of the gap to the best state-of-the-art upper limit, with 95% cluster-bootstrap intervals. The LLM planner compares with the best simple planner and the enumerator as {{TBD:planner_comparison}}, and with the best ported external agent as {{TBD:h9_difference}}. On validation data the first version of the evidence formula did not reach its pre-registered target, and we report that outcome as it stands: {{TBD:g2_decision}}. The main limitations are the synthetic data, the simulated banks and private set intersection, a privacy guarantee per transaction rather than per customer, and a small number of askable alerts and positives, which make several intervals wide.

## Placeholders in this skeleton

| Placeholder | Where it comes from |
|---|---|
| `{{TBD:gap_closed_gfp}}`, `{{TBD:gap_closed_sota}}` | H1 and H2 rows of section 7.4 |
| `{{TBD:planner_comparison}}` | H3 row of section 7.3 |
| `{{TBD:h9_difference}}` | H9 row of section 7.4 |
| `{{TBD:g2_decision}}` | the team's G2 wording, `docs/report/05_4_gate_fusion_decision.md` section 5.4.4 |

Sentence order for the final version: problem, idea, safeguards, data and comparisons, headline numbers (gap closed against both upper limits; LLM against simple planners; best external agent), the G2 outcome, limitations. Target 200 words; the draft above is about 280 and has to be cut when the numbers are in.
