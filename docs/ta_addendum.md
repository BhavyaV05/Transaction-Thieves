# TA addendum (DRAFT, for team review before sending)

**To:** our TA  **From:** Team Cache me if you can (Shreyas, for the team)
**Re:** "Follow the Money Without Moving the Data": changes in response to the review of `proposal_v5.pdf`
**Status:** draft. Every owner checks their part; Shreyas sends it at G0. The full detail is in `plan.md` (sections cited below).

---

Thank you for the review. You asked us to (1) add SOTA systems as baselines, (2) compare against other agents built for similar tasks on our benchmark, (3) try newer general agentic frameworks on this domain, and (4) keep our existing ablations. This addendum lists what changed. **Nothing from the proposal's Table 5 has been removed.**

## 1. External comparisons (new Table 5a; plan §7.1)

Table 5 is now split in two: **5a, external systems** (new) and **5b, our ablations** (unchanged).

| Group | Systems | Role | Tier |
|---|---|---|---|
| A: SOTA detection models | Multi-GNN (AAAI 2024); FraudGT (ICAIF 2024), MEGA-GNN | Stronger one-bank starting model **and** stronger "see everything" upper limit | Multi-GNN core; others secondary |
| B: privacy-preserving cross-bank | DP-FedAvg GNN (30 banks as clients); one AML-specific federated method (chosen by code availability) | Private *training* vs our private *questions* | DP-FedAvg core; AML-specific secondary |
| C: AML / fraud LLM agents | **Torres et al. [6]**, **Sharma [3]** (the real system, not our imitation of it); Co-Investigator AI, FinFRE-RAG; Palma, FAA [5], AEGIS | Existing agents for the same task | Torres and Sharma core; others secondary or stretch |
| D: general agent frameworks | **ReAct, Reflexion, ReWOO**; LATS, LLMCompiler, debate / AutoGen, DSPy-tuned ReAct, Uncertainty of Thoughts, BED-LLM; more at stretch | Newer agentic frameworks applied to our problem | ReAct, Reflexion, ReWOO core; others secondary or stretch |

Group C agents run in two modes: **native** (as published: one bank's view, no cross-bank questions) and **ported** (given our cross-bank question tool with the same privacy budget). Every LLM system gets **two scores**: end-to-end (its own verdict) and questions-only (its collected answers passed through our fusion formula). The second score separates "asks better questions" from "makes better final calls".

**Code status** (full checks: `docs/external_code_check.md` for Groups C/D, `docs/t1_week0.md` for Groups A/B):
- **Multi-GNN, MEGA-GNN:** official code, Apache-2.0. Multi-GNN is already reproduced on HI-Small (test F1 67.64 vs the paper's 68.16).
- **FraudGT, Tian et al. [8]** (our pick for the AML-specific federated method): official code, but **no licence**. We are asking the authors and run them from separate checkouts until they answer.
- **The ~80 F1 on HI-Small** that benchmark sites list is ExSTraQt (arXiv 2604.02899): a feature-based method on a different split, with code but no licence. We add it to Group A as a reference if the authors allow it.
- **Sharma:** official MIT-licensed code, ported faithfully.
- **Torres:** no released code, prompts or named model. It is a labelled **reimplementation**, and its published numbers (on SynthAML10) cannot be checked on AMLworld.
- **ReAct, Reflexion, ReWOO:** MIT reference code.

## 2. Same conditions for every system (plan §6.16, §7.2)

A shared agent harness gives every system, ours or external, the same:
- alerts
- simulated banks
- question language
- compliance gate
- privacy budget (ε per alert, cap per account)
- LLM backbone
- caps on LLM calls, tokens and simulated time

All external systems run on a frozen 1,000-alert subset. The top 2–3 on validation (chosen at G4) also run on the full test set. Our system and all ablations run on both.

## 3. New research question and hypotheses (plan §2)

- **RQ6 (core):** How does the full system compare with SOTA detection models, privacy-preserving cross-bank methods and other LLM agents on the same benchmark?
- **H9 (primary, last in the Holm order H1 → H2 → H3 → H4 → H9):** at ε=2, our system closes at least as much of the gap as the best ported external agent on the same backbone. We're wrong if the best external agent is clearly better (95% CI). Placing it last leaves H1–H4 exactly as approved.
- **H10 (secondary):** private questions match or beat DP federated training on Brier at a comparable privacy level. It is secondary because training-time ε and our per-question ε are different units. We plot them on the same axes with an explicit caveat.

Gap closed is now reported against **both** upper limits: the dataset authors' GFP model and the best SOTA model.

## 4. Planner model size is now measured, not assumed (plan §6.11, E18)

The proposal fixed the planner at Qwen3-8B. Since it must reason over several kinds of evidence at once, we now pick its size with a small validation-only experiment (E18). It compares well-known open-weight models that fit our 16 GB GPU: Qwen3-8B / 14B / Qwen3.8-27B, Gemma 4 12B / 26B-A4B (/ 31B), gpt-oss-20b and Phi-4-reasoning.

The decision rule is committed before the run: the smallest model statistically equivalent to the best, subject to our per-alert time target. The fine-tuning experiment now asks the question the course brief suggests: does a QLoRA-tuned small planner match a larger prompted one?

## 5. Compute and hardware (plan §11–§12)

The proposal assumed one RTX 5070 Ti. We now use the team's four machines:

| Machine | Role |
|---|---|
| RTX 5070 Ti, 16 GB | Reference machine: LLM serving, fine-tuning, final run |
| RTX 3070 Ti laptop, 8 GB | GNNs and federated training |
| RTX 3050 Ti laptop, 4 GB | CPU-heavy simulation and small models |
| CPU-only | Evaluation, statistics, gate |

Timing numbers come from the reference machine only.

| | Proposal | Now |
|---|---|---|
| GPU hours | 100–215 (expected ≈115) | 172–347 (expected ≈190–240) |
| Budget | 250 | 250 (unchanged) |
| Cost | $0 | $0 |

The increase is the external systems (≈40–70 h), the SOTA GNNs and federated training (≈20–40 h), and the model-size sweep (≈12–22 h). The expected case fits the budget. The worst case does not, so a weekly GPU-hour log and a fixed cut order (plan §14) apply. The core is never cut.

## 6. Scope (plan §14)

The core tier now also contains:
- Multi-GNN
- the harness and fairness protocol
- ReAct, Reflexion, ReWOO
- Torres and Sharma (native and ported)
- DP-FedAvg

The main claims rest on RQ1, RQ2 and RQ6 (H1–H4, H9). Everything else keeps its proposal tier. The added secondary and stretch systems are cut first if we run late.

## 7. What we ask you to confirm

1. **H9 as a primary hypothesis**, last in the Holm order (or tell us to make it secondary).
2. **The planner-size change (§4)**: the planner backbone is chosen by E18 at G3 instead of being fixed at 8B.
3. **The revised compute estimate (§5)** and the cut order.
4. **Optional frontier-API run** (GPT/Claude/Gemini as backbone on 200–300 alerts, to test whether model size is the bottleneck). It breaks "$0, all local", so it is stretch only, decided at G4. Is this acceptable in principle?
5. **Using IBM snapml** for the dataset authors' graph features. It installs free from PyPI, but its licence is IBM's proprietary ILAN: no redistribution, and an "authorised use" that a free download leaves undefined. We would use it as a library only, never bundled into our code, and keep a path that doesn't need it. If you say no, we reimplement the features ourselves (30–50 h).

**Where we are:** data preparation, the baselines and the alert intake are already built, so that the G1 checks are ready. The test split is locked in code and has not been read. Everything else waits for your sign-off (G0).
