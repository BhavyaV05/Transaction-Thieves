# Prereg changes implied by the LangGraph rewrite (proposal for T4; prereg/* untouched)

## R2: Group D agents as graphs
1. PORTED/NATIVE arms keep their names (react, reflexion, rewoo, lats, llmcompiler, dspy_react, debate, uot, bed_llm, mediq). They are now
   LangGraph StateGraphs (`ftm/graph/agents/*`) proven identical to the old loops on the mock LLM (tests/test_graph_agents_ported.py:
   same verdict, probability, question log, trace and LLM requests over fuzzed seeds). No change to external_list.yaml fixed_choices.
   The old loops stay in `ftm/external` (A/B via FTM_LEGACY_AGENT_LOOPS=1).
2. NEW ARMS `<agent>_adapted` (ten, mediq is a stretch): each agent's control strategy as a `PlannerPolicy` inside the system graph. It sees
   p0, the posterior, evidence, rules and the routed back-tested menu with bin -> LR maps, and picks menu options (no composing); the verdict is
   the fused posterior (never an LLM number). Adds to prereg: the system names, `alerts: subset` only, the fixed settings of
   `ftm/graph/agents/adapted.py` (reflexion: reflection on a refused/uninformative answer + CONFIDENT/RETRY check on stop, max 3 retries; rewoo:
   one plan call, max 6 steps; lats: 3 samples at T=0.7 + one value call; llmcompiler: DAG waves, 1 replan; debate: 3 analysts x 2 rounds,
   majority; uot/bed_llm: EIG stop 0.02 bits, clean side of UoT derived from our LRs; mediq: confidence 0.8, 4 questions).
3. dspy_react_adapted reuses the MIPROv2 instruction paragraph of outputs/t3_dspy/react_miprov2.json (sha 0697b613...) without
   re-optimisation; its few-shot/field structure is not reused. If T4 prefers a re-optimised policy prompt, omit this arm.
4. Hypothesis wording: H9 compares externals on the same harness; the adapted arms answer "is the gap the control strategy or the
   knowledge of LRs/p0?". They are exploratory, reported separately from Table 5a.
