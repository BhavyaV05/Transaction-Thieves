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

## R3: E18 backbone refresh on the new pipeline (labelled deviation from prereg/e18_decision.yaml)
1. What: after the rewrite (LangGraph planner, routed menu, ConditionalFusion headroom_v1, p0 v2, edge scorer v2, tuned threshold) the backbone is re-selected on
   validation `fit` + `cal` (300 askable alerts) among Gemma 4 12B (incumbent), Qwen3.5 9B and 4B, Gemma 4 E4B, Qwen3-14B and gpt-oss-20b (1,024-token output
   cap, `reasoning_effort low`), plus Phi-4 14B optionally. Trigger: the question "is Gemma 4 12B really the best, newer small Qwen/Kimi models should be smarter";
   Kimi models were checked and are 30 GB to 1.5 TB (docs/t3_e18.md, "E18 refresh (2 Nov)").
2. How it differs from prereg/e18_decision.yaml: (a) different alerts (fit + cal, 300, not eval 160 / all 460); (b) the new pipeline, not the naive-fusion planner that
   the prereg names; (c) the candidate list is new; (d) the rule is "lowest Brier with a paired cluster-bootstrap CI, prefer the smaller model when the CI of the
   difference spans 0" (equivalence by CI containing 0 instead of the 0.5 x |Brier(random) - Brier(best simple)| margin) and adds a usefulness gate: the model must
   beat or tie the non-LLM EIG reference of the same fusion; (e) no stage 2 and no thinking arm (thinking stays off for all). Kept from the prereg: paired
   attempt+accounts cluster bootstrap (10,000, 95%, seed 0), the valid-DSL filter >= 0.95, thinking off, 8,192 context / 6,000 prompt cap, one model at a time.
3. The rule was written into docs/t3_e18.md before any refresh run (2 Nov). It uses only fit and cal; the eval part and test are untouched, so it cannot leak.
4. Status: preliminary and exploratory until T4 decides. Options for T4: (i) report it as an addendum to E18 and keep the 26 Oct pick on the prereg row;
   (ii) adopt its pick as the production backbone and re-run the eval-part rows once on it; (iii) drop it. The prereg file is not edited by T3.
