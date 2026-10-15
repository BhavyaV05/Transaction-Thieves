# Revised Project Plan: Adding SOTA and External-Agent Baselines

**Project:** Follow the Money Without Moving the Data (Team *Cache me if you can*, LMA Group Project 2026)
**Base document:** `proposal_v5.pdf`. It stays unchanged; this plan lists the changes on top of it.

---

## 0. Instructor feedback

> "You only have ablation baselines. Add some SOTA systems as baselines. Also compare against other agents built for similar tasks, on your benchmark/eval set. Try several newer agentic frameworks on this domain, coming from the more general problem. Overall direction is good. Just don't drop your existing ablations when you add the comparisons to other systems."

**Diagnosis.**
- Every row in Table 5 is either one of our own variants or a reference point (one bank only, see everything, share raw records).
- Even "LLM decides" only imitates Sharma [3]; it isn't Sharma's actual system.
- Several papers we already cite ([3–6], [8–13]) only appear in related work and are never run.

**Hard rule from the feedback:** every existing ablation in Table 5 stays. New comparisons are added next to them; nothing is replaced.

---

## 1. Changes to the project outline

### 1.1 Split Table 5 into two tables

| Table | Contents |
|---|---|
| **5a: External comparisons (new)** | Groups A–D below |
| **5b: Ablations (unchanged)** | Every current row: one bank only, see everything, share raw records, simple planners, enumerator, LLM fixed list, no back-testing, LLM decides, fast-model options, evidence views, other switches |

### 1.2 Group A: SOTA detectors (no agents)

**Problem.**
- Both the starting point and the "see everything" upper limit use GFP + gradient-boosted trees (63.23 F1 on HI-Small).
- Benchmark aggregators list about 80 minority-class F1 on HI-Small, but they don't name the paper behind it.
- If a stronger model really reaches ~80, a GFP-based upper limit is too low. "Gap closed" would then look bigger than it really is.

**Change.**
- Add Multi-GNN (Egressy et al., AAAI 2024), FraudGT (ICAIF 2024), MEGA-GNN, and whatever model is behind the ~80 figure (trace its source first).
- Retrain on **our** split (days 1–9 / 10–12 / 13–16), in both one-bank and see-everything modes. Published numbers use a different split.
- Report gap closed against **both** upper limits: GFP and the best SOTA model.

### 1.3 Group B: SOTA privacy-preserving cross-bank methods

This is the most natural SOTA comparison for RQ1. Today, federated AML [7–9] appears only as "all-or-nothing" related work.

**Change.** Run at least:
- a federated GNN with differential privacy (DP-FedAvg), and
- one method built for AML, chosen from:
  - Tian et al. [8] (WWW 2025)
  - Ceydeli et al. [9]
  - FedGraph-VASP (boundary-embedding exchange, arXiv 2601.17935)
  - FC2L (NeurIPS 2026)

Put them on the same privacy-vs-detection plot as our system. Add a short paragraph on why their ε and ours measure different things:
- training-time privacy vs per-question privacy
- per-transaction protection vs per-customer protection

### 1.4 Group C: existing LLM agents for fraud and AML

Each agent runs in two modes:
- **Native:** as published, with one bank's view and no cross-bank tools. This tests whether an LLM agent on its own beats the one-bank model (the failure Sharma [3] reports).
- **Ported:** given our question language and the same ε budget.

| System | Priority | Notes |
|---|---|---|
| Torres et al. [6] | Must | Closest to our setup (AML triage, subgraph evidence, counterfactual checks) |
| Sharma [3] | Must | Faithful reimplementation, replacing today's imitation |
| Co-Investigator AI (arXiv 2509.08380) | Secondary | Multi-agent AML system from PayPal |
| FinFRE-RAG [28] | Secondary | Retrieves similar fraud cases |
| Palma [4], FIA [5] | Stretch | |
| AEGIS [22] | Stretch | Hackathon design sketch; only if code exists |

### 1.5 Group D: general agentic frameworks and information-seeking agents

The general problem behind our project is sequential information gathering under a budget, in a partially observed environment. Port these to our tool interface.

**Control loops:**
- ReAct
- Reflexion
- ReWOO / Plan-and-Execute
- LATS (tree search)
- LLMCompiler (parallel tool calls, which ties directly to RQ5)
- CodeAct
- multi-agent debate / AutoGen-style group chat

**Information-seeking specialists:**
- Uncertainty of Thoughts (NeurIPS 2024)
- BED-LLM [10]
- MediQ
- GraphDx [11]
- zero-shot LLM feature acquisition [12]
- NOL\* [13]: RL for exploring partially observed networks, an almost exact match to our problem

**Memory frameworks, compared against our strategy library:** ExpeL and Agent Workflow Memory.

**Fairness fix.** Our planner is fine-tuned while these run zero-shot. Add one **DSPy-tuned ReAct** (tuned on validation only) so the strongest external agent isn't a straw man.

### 1.6 Fairness protocol (new paragraph in Sec 8)

Every system, ours and external, gets the same:
- test alerts (from Step 1's conformal intake)
- simulated partner banks (latency, declines, timeouts)
- question language and compliance gate
- ε per alert and per-account ε cap
- backbone model (Qwen3-8B by default, so only the framework differs)
- cap on tokens, LLM calls and wall-clock time

Each external agent is scored two ways:
1. **End-to-end:** its own final probability (native scoring if it has one, otherwise a verbalised probability).
2. **Questions-only:** the answers it collected, run through **our** evidence fusion.

This shows whether we win on *asking better questions* or on *deciding better*. That split is a finding in itself.

### 1.7 Research questions and hypotheses (Sec 3, Table 2)

**New RQ.** How does the full system compare with SOTA systems and other agents on the same benchmark?

| ID | Tier | We expect | We're wrong if |
|---|---|---|---|
| **H9** | Primary (last in the Holm order, after H1–H4) | At ε=2, our system closes at least as much of the gap as the best ported external agent on the same backbone | The best external agent is clearly better (95% interval) |
| **H10** | Secondary | Private questions match or beat DP federated training on Brier at a comparable privacy level | Federated training is clearly better at a comparable privacy level |

If ReAct + our tools + our fusion matches our planner, the contribution becomes the tool and fusion layer. That's still a measured, publishable result; state this in Sec 6.

### 1.8 Component-level SOTA (secondary RQs)

| Component | Add as comparison |
|---|---|
| RQ3: fast model | RouteLLM / FrugalGPT-style cascades; fine-tuned ModernBERT/DeBERTa classifier; small general LLMs |
| RQ4: evidence selection | GNNExplainer, PGExplainer, personalised PageRank for picking the 40 transactions |
| Safety: injection defences | CaMeL [14], Spotlighting, StruQ; also run the injection test set on the external agents |

### 1.9 Smaller edits in other sections

- **Header "Evaluation" line, Sec 1–2:** list the four comparison groups.
- **Table 1:** add Multi-GNN/FraudGT, FedGraph-VASP/FC2L, Co-Investigator AI, and UoT/MediQ.
- **Sec 6 "What's new":** add (6) a reusable cross-bank investigation benchmark (alerts, simulated banks and tool API) that other agents can plug into.
- **Sec 9 metrics:**
  - cost-matched Pareto plots (gap closed vs ε, LLM calls and GPU-seconds per alert)
  - results per laundering pattern type
  - gap closed against both upper limits
- **Sec 10 safety:** run the injection test on external agents too.
- **Sec 11 compute, Sec 12 tiers/gates/risks:** updated below.

---

## 2. Plan

### 2.1 Scope tiers for the new work

| Tier | New items |
|---|---|
| **Core** (must be done by G5) | SOTA GNN as both starting point and upper limit (Multi-GNN at minimum); ReAct, Reflexion and ReWOO ported with Qwen3-8B; Torres [6] and Sharma [3] in native and ported modes; DP-FedAvg GNN; H9 |
| **Secondary** | LATS, LLMCompiler, multi-agent debate, UoT / BED-LLM, Co-Investigator AI, FinFRE-RAG, one AML-specific federated method, DSPy-tuned ReAct, component-level SOTA (1.8), injection tests on external agents, H10 |
| **Stretch** | CodeAct, MediQ, GraphDx, NOL\*, FC2L, FIA, Palma, AEGIS, a frontier-API backbone run |

All existing ablations stay in their current tiers (proposal Table 8).

### 2.2 Timeline, aligned to the existing gates

Streams (from proposal Sec 12):
- **(1)** data & baselines
- **(2)** simulated banks & simple planners
- **(3)** LLM, fine-tuning & retrieval
- **(4)** rules, tests, statistics & write-up

| When | Work | Stream |
|---|---|---|
| **Before G0** | Send the TA a one-page addendum with the new comparison table, H9/H10 and the fairness protocol. Check code availability and licences for every external system; mark the rest "reimplemented from paper". Trace the source of the ~80 F1 HI-Small figure. | 4 + all |
| **Wk 1–2 (G1)** | Train Multi-GNN (+ FraudGT if time) on HI-Small with our split, one-bank and see-everything, next to GFP. **Gate:** reproduce within a few F1 points of published numbers. | 1 |
| **Wk 2–3 (G2)** | Build the shared agent harness: tool API wrapper, budget enforcement, token/time caps, standard output (probability + question log), logging. Start the DP-FedAvg setup. | 2 (harness), 1 (federated) |
| **Wk 3–4 (G3)** | ReAct runs end-to-end on 50 validation alerts through the harness. Lock the external system list, fairness protocol and H9/H10 thresholds; any later change goes back to the TA. | 3 + 4 |
| **Wk 5–8** | Implement the core external agents, then the secondary ones. Native and ported modes for the domain agents. Train the federated baseline(s). | 3 (frameworks), 2 (domain-agent ports), 1 (federated) |
| **Wk 9 (G4)** | All external systems run on validation. DSPy-tune the top frameworks. Choose which go to the full test set. Pick stretch items. | All |
| **Wk 10–13** | Component-level SOTA (fast model, evidence selection, injection defences). Injection test set on external agents. Dry-run all analysis scripts on validation. | 3 + 4 |
| **Wk 14 (G5)** | One final test run: ablations + external systems, same code path. | All |
| **Wk 16 (G6)** | Report and presentation: Tables 5a/5b, Pareto plots, per-pattern breakdown, questions-only vs end-to-end analysis. | 4 + all |

**New gate checks:**
- **G1:** a SOTA GNN reproduces the published numbers.
- **G3:** the harness runs ReAct end-to-end; the external list is locked.
- **G4:** every core external system has validation results.

### 2.3 Compute (changes to Table 7)

**Issue.** The current expected total is ≈115 GPU-h. Running ~10 external agents on all 4,000 test alerts would add another ~90–220 GPU-h, which breaks the 250 h budget.

**Approach.**
- Run external agents on a fixed, stratified **1,000-alert subset** (≈100–150 laundering cases). The smallest detectable Brier difference rises from about 0.011 to about 0.022; state this in the report.
- Run only the top 2–3 external agents (chosen at G4) on the full test set.
- Cap LATS and multi-agent debate by tokens per alert.

| New item | Estimate |
|---|---|
| External agents (subset + top 2–3 on full test) | 40–70 GPU-h |
| SOTA GNNs (one-bank + see-everything) and federated training | 20–40 GPU-h |
| **New expected total** | **≈175–225 GPU-h** (budget: 250) |

The worst case goes over budget. Add to the **start** of the cut order (Table 10): RL planner → frontier-API run → stretch external agents → (existing order).

### 2.4 New risks (additions to Table 10)

| Risk | What we do about it |
|---|---|
| No public code for a cited system | Reimplement from the paper, label it as a reimplementation, and check it against any numbers the paper reports |
| "Unfair comparison" criticism | Fairness protocol (1.6), native and ported modes, DSPy-tuned external agent, same backbone |
| A strong external agent beats our planner | Still a measured result; questions-only scoring shows whether our tool and fusion layer still adds value |
| SOTA GNNs don't fit on 16 GB for HI-Medium | Neighbour sampling; if still infeasible, report SOTA upper limits on HI-Small and say so |
| Compute overrun | Cut order in 2.3 |

### 2.5 Open decisions for the team

1. **Frontier API models** (GPT/Claude/Gemini) as a backbone on a small subset would make the comparison more convincing. It breaks the "$0, all local" story, though; the data is synthetic, so privacy isn't a concern. *Recommendation: stretch.*
2. **H9 in the primary tier.** *Recommendation: yes.* It's clearly what the instructor cares about, and putting it last in the Holm order leaves H1–H4 unaffected.
3. **TA addendum timing.** The proposal says nothing is built before the TA signs off, so the addendum goes out before any implementation.

---

## Sources (new references)

- WizWand IBM AML benchmark listing: https://www.wizwand.com/dataset/ibm-aml
- Co-Investigator AI: https://arxiv.org/pdf/2509.08380
- FedGraph-VASP: https://arxiv.org/abs/2601.17935
- FC2L (NeurIPS 2026): https://neurips.cc/virtual/2026/163758
- Networked Markets, Fragmented Data: https://arxiv.org/pdf/2512.24487
- AEGIS hackathon submission: https://lablab.ai/submissions/b08nogx0pv8rkh07pb9sbeet
