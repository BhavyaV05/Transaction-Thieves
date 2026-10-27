# Group C secondary ports (T2): Co-Investigator AI and FinFRE-RAG

Owner: T2 (Bhavya). Spec: `plan.md` 6.17, `tracks.md` 4.9 (week 7). Both are **reimplementations**: no code was released (`docs/external_code_check.md`), so they follow the papers as far as the papers go and say where they do not. Each runs **native** (one-bank view, no `ask_bank`) and **ported** (the same gate, accountant, eps and caps as every other system, `ask_bank` through the shared `Investigation` layer). Fixed choices are in `prereg/external_list.yaml` (`fixed_choices`), committed before any run. Both use T4's evidence bundle, case memory, T3's `BudgetedClient` and mock LLM, so the fairness protocol (same backbone, same prompt cap, same call cap) holds.

## Co-Investigator AI (arXiv:2509.08380, PayPal) - `co_investigator.py`

What the paper specifies (read in full, v2): a Planning Agent that spawns typology agents and an external-intelligence agent; seven typology agents; a Narrative Generation Agent with chain-of-thought and structured confidence scores; a Compliance Validation Agent (LLM coherence plus rule checks on amounts, dates, entities); a Feedback Agent that revises from flagged issues; three memory layers. **What it does not give:** any prompt, the model, step/retry limits, thresholds, how memory is retrieved, a typology list, or any decision or probability (the output is a SAR narrative for a human).

Our pipeline, per alert (<= 10 LLM calls): plan (1) -> up to 4 of 6 typology agents (the dispute-pattern agent is dropped: AMLworld has no disputes) -> [ported] external intelligence = up to 3 partner questions -> narrative (1) -> validator (rule check + 1 coherence call) -> at most one revision -> verdict. **Probability** = the narrative agent's stated `confidence` if the validator accepts the narrative, else `None` (scored as p0). The AI-Privacy Guard is the aliasing the alert text already has; crime-type detection by tree ensembles is not reproduced.

## FinFRE-RAG (Tan, Ma, Zhang; arXiv:2512.13040, ACL 2026) - `finfre_rag.py`

The paper is fully specified and followed: Random Forest feature importance on the labelled pool, top k = 10 features, z-scores; per query categorical equality filters (most important first, while the candidate set stays non-empty) then cosine similarity on the numeric z-vectors, n = 20 labelled exemplars; rows written as natural-language sentences ("It is a fraud." / "It is not a fraud."), missing values as "missing"; the instruction and system prompt as published; the model answers `Score: 1-5` and the paper counts >= 4 as fraud. A query may get fewer than 20 exemplars when the filters leave fewer candidates (Eq. 2 as written).

Our adaptation: the pool is **TRAIN alerts only** with their labels (the pool refuses any other split); a row is the alert (the alerted payment, p0, the shape buckets) because AMLworld has none of the paper's fields; RF needs scikit-learn and falls back, logged, to univariate |AUC - 0.5| without it; the 1-5 score becomes a probability with a fixed table (and `fit_score_map` can replace it from TRAIN data); temperature 0 for every system, not the paper's 0.6; ported mode adds one question round and appends the answers to the current case (the paper has no partner banks). One repair call if the `Score:` line is missing.

## Using them

```python
from ftm.external.co_investigator import CoInvestigator
from ftm.external.finfre_rag import ExemplarPool, FinfreRag

system = CoInvestigator(llm, mode="ported", memory=case_memory)          # llm: any ChatClient (endpoint or MockLLM)
pool = ExemplarPool.from_alerts(train_alert_contexts, train_labels)       # TRAIN alerts only
system = FinfreRag(llm, pool, mode="native")
```

Both read `alert.summary` like the other Group C ports (`amount`, `currency`, `format`, banks, `time`, `evidence`, `partner_banks`, `shape`). Wiring them into `scripts/run_llm_agent.py` is a two-line change in T3's script (not done here).

## Stretch (week 11): MediQ - `mediq.py`

Group D, ported only. **Not in `prereg/external_list.yaml`**, so it is not part of the pre-registered Table 5a and the final-run job plan does not include it; adding it needs the TA (the list is locked at G3). It is registered in `scripts/run_llm_agent.py` (`--system mediq`) so it can run on validation as an exploratory row if the team wants one.

MediQ (Li et al., NeurIPS 2024, arXiv:2406.00922) is an expert that abstains from answering until it is confident, and asks the patient follow-up questions meanwhile. Reimplemented from the paper (the official repo is CC-BY-4.0; nothing copied). Per turn: an abstention call (the model states a probability and a 0-1 confidence; the paper's "numerical" variant), and, if the confidence is under the fixed threshold 0.8, a question call. At most 4 questions, no rationale, no self-consistency (the shared call cap of 20 does not leave room for them). The probability is the last one the model stated; a model that never stated one is scored as p0. It uses neither the back-tested LRs nor an EIG calculation, which is what separates it from our planner, UoT and BED-LLM. Tests: `tests/test_mediq_port.py` (mock LLM).

Not done from the stretch list (`plan.md` 7.1): CodeAct, GraphDx, NOL*, FC2L, ExpeL, Agent Workflow Memory. ExpeL and AWM are memory frameworks meant to replace T3's strategy library, which is still being built; GraphDx has no licence; CodeAct and NOL* do not fit a fixed DSL tool. They stay unported.

