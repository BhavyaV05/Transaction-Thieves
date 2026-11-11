## Errors by the pipeline step where they went wrong (first failing step; tags: no_verdict = verdict, no_question_asked = planner, all_blocked = gate / accountant, no_reply = partner bank reply, no_backtested_lr = back-test (no accepted LR), verdict_override = verdict (system overrides the fused answers), fusion_moved_wrong = fusion / evidence, answers_insufficient = evidence too weak)

| system | variant | mode | alerts | n | wrong calls (p>=0.5 vs label) | no_verdict | no_question_asked | all_blocked | no_reply | no_backtested_lr | verdict_override | fusion_moved_wrong | answers_insufficient |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours_compose | compose | ported | full_test | 771 | 178 | 0 | 25 | 0 | 7 | 30 | 84 | 27 | 5 |
| ours_compose | compose | ported | subset | 208 | 42 | 0 | 6 | 0 | 0 | 9 | 21 | 5 | 1 |
| ours_fixed_list | fixed_list | ported | subset | 208 | 46 | 0 | 7 | 0 | 1 | 0 | 22 | 15 | 1 |
| ours_act_only | act_only | ported | subset | 208 | 47 | 0 | 6 | 0 | 2 | 3 | 22 | 12 | 2 |
| ours_no_backtest | no_backtest | ported | subset | 208 | 43 | 0 | 1 | 0 | 0 | 16 | 21 | 4 | 1 |
| ours_llm_decides | llm_decides | ported | subset | 208 | 40 | 0 | 7 | 0 | 0 | 13 | 13 | 4 | 3 |
| ours_compose | eps_1 | ported | full_test | 771 | 173 | 0 | 48 | 0 | 2 | 43 | 57 | 20 | 3 |
| ours_compose | eps_4 | ported | full_test | 771 | 181 | 0 | 14 | 2 | 6 | 24 | 97 | 36 | 2 |
| co_investigator_ported | ported | ported | subset | 208 | 66 | 34 | 0 | 16 | 2 | 5 | 8 | 1 | 0 |
| bed_llm | ported | ported | subset | 208 | 53 | 0 | 5 | 24 | 0 | 1 | 15 | 6 | 2 |
| bed_llm | ported | ported | full_test | 771 | 212 | 0 | 23 | 84 | 4 | 3 | 84 | 9 | 5 |
| random | random |  | full_test | 3873 | 915 | 734 | 0 | 0 | 3 | 0 | 145 | 26 | 7 |
| random | random |  | subset | 1000 | 232 | 185 | 0 | 0 | 0 | 0 | 39 | 8 | 0 |
| fixed | fixed_order |  | full_test | 3873 | 920 | 734 | 0 | 0 | 1 | 0 | 154 | 19 | 12 |
| fixed | fixed_order |  | subset | 1000 | 228 | 185 | 0 | 0 | 0 | 0 | 40 | 2 | 1 |
| eig | eig |  | full_test | 3873 | 926 | 734 | 13 | 0 | 5 | 0 | 114 | 59 | 1 |
| eig | eig |  | subset | 1000 | 234 | 185 | 3 | 0 | 2 | 0 | 31 | 13 | 0 |
| learned | learned_policy |  | full_test | 3873 | 927 | 734 | 13 | 0 | 8 | 0 | 112 | 58 | 2 |
| learned | learned_policy |  | subset | 1000 | 233 | 185 | 3 | 0 | 3 | 0 | 29 | 13 | 0 |
| enumerator | enumerator |  | full_test | 3873 | 916 | 734 | 13 | 0 | 7 | 0 | 108 | 50 | 4 |
| enumerator | enumerator |  | subset | 1000 | 230 | 185 | 3 | 0 | 3 | 0 | 25 | 14 | 0 |
| raw_share | raw_share |  | subset | 1000 | 226 | 185 | 0 | 0 | 41 | 0 | 0 | 0 | 0 |
| raw_share | raw_share |  | full_test | 3873 | 902 | 734 | 0 | 0 | 168 | 0 | 0 | 0 | 0 |
| rewoo_adapted | rewoo_adapted | ported | subset | 208 | 47 | 0 | 6 | 0 | 3 | 0 | 23 | 15 | 0 |
| dspy_react_adapted | dspy_react_adapted | ported | subset | 208 | 49 | 0 | 6 | 0 | 0 | 0 | 26 | 16 | 1 |
