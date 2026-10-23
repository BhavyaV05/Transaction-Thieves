# Group C/D code and licence check (week 0)

Owner: T4 (Shreyas). Checked 2026-10-16 against arXiv abstract pages, the papers, GitHub (API licence field and repo contents) and web search. Re-check any "none found" row before the system's target week; authors sometimes release code late.

Covers every Group C and Group D system in `plan.md` §7.1. Group A/B (detection models, federated) are T1's check.

## How to read the licence column

| Licence | What we may do |
|---|---|
| MIT, Apache-2.0 | Copy and adapt code. Keep the original copyright notice in every ported file (Apache-2.0: also keep any NOTICE file and mark changed files) |
| CC-BY-4.0 | Reuse with attribution. Unusual for code; credit the authors in the file header and the report |
| No licence file | All rights reserved. Read it to understand the algorithm, but **do not copy code into this repo**. Reimplement from the paper and label it "reimplementation" |
| No code found | Reimplement from the paper and label it "reimplementation" |

## Group C: domain agents (AML / fraud)

| System | Tier, owner | Paper | Official code | Licence | Port plan |
|---|---|---|---|---|---|
| **Torres et al. [6]** | Core, T4 | arXiv 2604.19755 (paper CC-BY-4.0) | None found | n/a | **Reimplementation.** See the Torres notes below: the paper leaves out the backbone, prompts and several settings |
| **Sharma [3]** | Core, T4 | arXiv 2607.19266 | [rahil1303/auditable-fraud-investigation](https://github.com/rahil1303/auditable-fraud-investigation) (linked from the arXiv comments) | **MIT** | **Faithful port of official code.** Agent is in `src/agent/` (`agent_loop.py`, `tools.py`, `decision_policy.py`, `groq_client.py`). Swap Groq `llama-3.3-70b-versatile` (temperature 0.1) for our backbone; map their tools onto `ToolBox`; keep the MIT notice |
| Co-Investigator AI | Secondary, T2 | arXiv 2509.08380 (PayPal) | None found | n/a | Reimplementation. The paper's output is a SAR narrative, so the port needs an explicit mapping to a probability verdict, fixed before any run |
| FinFRE-RAG [28] | Secondary, T2 | arXiv 2512.13040 (ACL 2026) | None found; one aggregator says code is "being assembled" | n/a | Reimplementation (importance-guided feature reduction + retrieval of labelled exemplars). Re-check for code before week 7 |
| Palma [4] | Stretch | arXiv 2609.32643 (EMNLP 2026) | None found | n/a | Reimplementation if attempted |
| FIA [5] | Stretch | arXiv 2506.11635 | None found | n/a | Reimplementation if attempted. **Citation fix:** the arXiv title is "**FAA** Framework" (Fraud Analyst Assistant), by Shuster, Zaloof, Shabtai and Puzis. Our references call it "FIA" |
| AEGIS [22] | Stretch | lablab.ai hackathon submission (June 2026) | [Kunsh162007/AEGIS](https://github.com/Kunsh162007/AEGIS) | **None** | Read only. Hackathon code with 15 agents on CrewAI + LangGraph and no licence: reimplement the core loop (challenger, verifier, adjudicator) or drop it |

### Torres notes (core, week 5)

The paper gives the architecture but not enough to reproduce it exactly:

- **No backbone named.** Section 4.6 says only "a high-capacity general-purpose LLM and a smaller, cost-efficient LLM". This doesn't conflict with us: under the fairness protocol it runs on our E18 backbone anyway.
- **No prompts and no appendix.** We write the prompts from §3 and publish them with the port.
- **What is specified:**
  - an evidence bundle of *k* items with per-type quotas (policy, KYC/context, transaction slice, similar cases), with *k* tuned on validation
  - a structured output (decision, confidence, supporting evidence IDs, contradicting/missing evidence IDs, rationale with citations, next actions)
  - a verifier that rejects non-existent citations or inconsistent amounts and dates, then triggers a repair step
  - decision and evidence counterfactuals with an edit budget
  - dispositions {dismiss, monitor, escalate}
- **Their numbers can't be checked on our data.** They evaluate on SynthAML10 (plus AMLSim and Elliptic), not AMLworld, so "check against the paper's numbers" (`tracks.md` §6.8) is not possible on our benchmark. Options:
  1. label it a reimplementation and report only our numbers (default)
  2. stretch: a small sanity run on SynthAML10 to see whether we land near their PR-AUC 0.75 / Escalate F1 0.62
- **Choices to fix before the first run** (record them in `prereg/external_list.yaml` at G3, so they aren't tuned after seeing results):
  - *k* and the quotas
  - the disposition → probability mapping (proposal: P(escalate) + ½·P(monitor) from the stated confidence; to agree with Akshat)
  - the repair budget
  - the counterfactual edit budget

## Group D: general frameworks and information-seeking agents

| System | Tier, owner | Official code | Licence | Port plan |
|---|---|---|---|---|
| ReAct | Core, T3 | [ysymyth/ReAct](https://github.com/ysymyth/ReAct) | MIT | Reimplement the loop in the harness using official prompts (plan §6.17) |
| Reflexion | Core, T3 | [noahshinn/reflexion](https://github.com/noahshinn/reflexion) | MIT | Same |
| ReWOO | Core, T3 | [billxbf/ReWOO](https://github.com/billxbf/ReWOO) | MIT | Same |
| LATS | Secondary, T3 | [lapisrocks/LanguageAgentTreeSearch](https://github.com/lapisrocks/LanguageAgentTreeSearch) | MIT | Official code may be adapted |
| LLMCompiler | Secondary, T3 | [SqueezeAILab/LLMCompiler](https://github.com/SqueezeAILab/LLMCompiler) | MIT | Official code may be adapted |
| DSPy-tuned ReAct | Secondary, T3 | [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) (actively maintained) | MIT | Use the library |
| Multi-agent debate | Secondary, T4 | [composable-models/llm_multiagent_debate](https://github.com/composable-models/llm_multiagent_debate) | **None** | Reimplement from the paper (Du et al., ICML 2024); the algorithm is a short loop |
| AutoGen | Secondary, T4 | [ag2ai/ag2](https://github.com/ag2ai/ag2) (AG2, already in our `llm` extra) | Apache-2.0 | Use AG2. (microsoft/autogen is MIT for code, CC-BY-4.0 for docs; not needed) |
| Uncertainty of Thoughts | Secondary, T4 | [zhiyuanhubj/UoT](https://github.com/zhiyuanhubj/UoT) | **None** | Reimplement from the paper (NeurIPS 2024); read the repo only for details |
| BED-LLM [10] | Secondary, T4 | None found (Apple / Oxford, ICLR 2026) | n/a | Reimplementation from the paper's EIG procedure |
| CodeAct | Stretch | [xingyaoww/code-act](https://github.com/xingyaoww/code-act) | MIT | Adaptable |
| MediQ | Stretch | [stellalisy/mediQ](https://github.com/stellalisy/mediQ) | CC-BY-4.0 | Adaptable with attribution |
| GraphDx [11] | Stretch | [ossiver-GEL/GraphDx](https://github.com/ossiver-GEL/GraphDx) ("official implementation") | **None** | Read only; reimplement |
| LLM feature acquisition [12] | Stretch | None found | n/a | Reimplementation |
| NOL\* [13] | Stretch | [tlarock/nol](https://github.com/tlarock/nol) (2020, Python 3.5) | MIT | Adaptable; expect dependency fixes |
| ExpeL | Stretch | [LeapLabTHU/ExpeL](https://github.com/LeapLabTHU/ExpeL) | Apache-2.0 | Adaptable |
| Agent Workflow Memory | Stretch | [zorazrw/agent-workflow-memory](https://github.com/zorazrw/agent-workflow-memory) | Apache-2.0 | Adaptable |

## What this changes

1. **Core systems are all buildable.**
   - Sharma has official MIT code.
   - Torres must be a reimplementation, and its paper numbers can't be checked on AMLworld. Say so in the TA addendum and in the report.
   - ReAct, Reflexion and ReWOO all have MIT reference code.
2. **Three secondary systems have no licence** (debate, UoT, plus GraphDx and AEGIS at stretch). We reimplement them and do not vendor their code.
3. **Four systems have no code at all** (Torres, Co-Investigator, FinFRE-RAG, BED-LLM), plus most stretch Group C systems. Their ports are reimplementations, labelled as such in Table 5a.
4. **Citation fix:** [5] is "FAA Framework", not "FIA".
5. **Open question for the team:** this repo has no licence file yet. Since we plan to release code (plan §17), we should pick one (MIT or Apache-2.0 are compatible with everything above) before vendoring any MIT/Apache code.
