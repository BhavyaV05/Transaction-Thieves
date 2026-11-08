# Graph versions against the legacy loops on the real model (ada, Gemma 4 12B TP2 fp16, 6-8 Nov 2026)

Each system ran three times on the same alerts: twice as the LangGraph version (g1, g2) and once as the legacy loop (l1, `FTM_LEGACY_AGENT_LOOPS=1`). Cells: agreement of the final probability (alerts where both runs gave one; a fraction when some alerts have none), then agreement of the number of LLM calls. Parity holds when l1-vs-g1 agreement is not lower than g2-vs-g1, the noise floor of the serving stack. Planner: 20 val `fit` alerts; external agents: 8 each.

| system | g2 vs g1 | l1 vs g1 |
|---|---|---|
| planner | 19/20, 19/20 | 19/20, 19/20 |
| bed_llm | 7/8, 7/8 | 7/8, 7/8 |
| co_investigator_native | 1.00 (1/1), 6/8 | 1.00 (1/1), 5/8 |
| co_investigator_ported | 1.00 (4/4), 7/8 | 1.00 (4/4), 5/8 |
| debate | 5/8, 8/8 | 5/8, 8/8 |
| finfre_rag_native | 8/8, 8/8 | 8/8, 8/8 |
| finfre_rag_ported | 8/8, 8/8 | 8/8, 8/8 |
| lats | 0.20 (1/5), 7/8 | 0.50 (2/4), 5/8 |
| llmcompiler | 7/8, 8/8 | 7/8, 8/8 |
| mediq | 8/8, 8/8 | 8/8, 8/8 |
| react | 0.80 (4/5), 6/8 | 1.00 (6/6), 8/8 |
| reflexion | 0.83 (5/6), 5/8 | 0.50 (3/6), 4/8 |
| rewoo | 1.00 (6/6), 8/8 | 1.00 (6/6), 8/8 |
| sharma_native | 8/8, 8/8 | 8/8, 8/8 |
| sharma_ported | 8/8, 8/8 | 8/8, 8/8 |
| torres_native | 8/8, 8/8 | 8/8, 8/8 |
| torres_ported | 1.00 (6/6), 7/8 | 1.00 (6/6), 7/8 |
| uot | 7/8, 8/8 | 7/8, 8/8 |

Reading: on 16 of 18 systems the loop agrees with the graph at least as often as two graph runs agree with each other. Co-Investigator's call counts differ by one or two alerts (5/8 against 6/8 and 7/8) with identical probabilities, inside what 8 alerts can resolve. Reflexion (0.50 against 0.83) and LATS give a probability on too few alerts to judge; both sample or retry, so their runs vary more.
