# T4: Rules, safety, evaluation and domain agents (Shreyas)

Checklist generated from tracks.md section 6.11. Read the whole track section first: it has the specs and acceptance tests.
Tick items off in your PRs and post one line per week in STATUS.md. Weeks count from TA approval (G0).

## Week 0
- [ ] **Send TA addendum**
- [ ] code/licence check for every Group C/D system
- [ ] collect machine specs from everyone
- [ ] confirm AML-specific federated method choice with Abhishek

## Week 1
- [x] Repo skeleton, CI, test-split guard
- [x] rules corpus collection
- [x] expert questions with Bhavya
- [x] run-log schema

## Week 2
- [x] Fusion v0
- [x] metrics + bootstrap code. **G1**

## Week 3
- [x] Ceiling checks with Bhavya. **G2**

## Week 4
- [ ] **Pre-register thresholds**
- [ ] lock external list. **G3**

## Week 5
- [x] Compliance gate + regulatory RAG
- [x] **Torres port**

## Week 6
- [x] Injection set generation
- [x] **Sharma port**

## Week 7
- [x] Fusion v1 (per-family weights)

## Week 8
- [x] Eval dashboard on val (`scripts/final_analysis.py`, dry-run on real HI-Medium val for the non-LLM arms on 28 Oct; LLM and external arms need Akshat's val runs)
- [x] debate / AutoGen port (secondary)

## Week 9
- [ ] Mid-project review
- [ ] **all core external agents on val**. **G4**

## Week 10
- [x] UoT / BED-LLM ports (secondary)
- [ ] FCAC (stretch)

## Week 11
- [ ] **Injection tests on our system and the external agents** (runner merged, mock only; real run needs the endpoint, `docs/t4_injection.md`)
- [ ] CaMeL comparison (CaMeL-style arm merged, ours and text-based; real run pending)

## Week 12
- [ ] **Dry-run all analysis scripts on val** (non-LLM arms done on real val; the rest waits for LLM and external val logs)
- [x] figure templates (`ftm/eval/final_tables.py`)

## Week 13
- [ ] Freeze pre-registration (30 Oct; `docs/t4_final_run.md` checklist, `docs/t4_prereg_values.md` for the `fix_on_val` values)
- [ ] hash to TA

## Week 14
- [ ] Stats, Holm, bootstrap. **G5**

## Week 15
- [ ] Report draft (sections 1-3, 5.4, 6, 8-11 and abstract skeleton drafted; 4, 5.1-5.3, 7 wait for their owners and results)

## Week 16
- [ ] Final report and presentation. **G6**
