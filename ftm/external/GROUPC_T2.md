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
