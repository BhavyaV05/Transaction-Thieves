## Case-study candidates for ours_compose (largest errors and largest recoveries; T3 and T4 pick the ten)

| kind | alert_id | home_bank | typology | label | p0 | p_e2e | p_questions_only | brier | gain vs p0 | questions | answered | with back-tested LR | stop_reason | tag | step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| error | F000347 | 0 |  | False | 1 | 1 | 0.5 | 1 | 0 | 0 | 0 | 0 |  | no_question_asked | planner |
| error | F000577 | 12 |  | False | 1 | 1 | 0.5 | 1 | 0 | 0 | 0 | 0 |  | no_question_asked | planner |
| error | F000802 | 20 |  | False | 1 | 1 | 0.5 | 1 | 0 | 0 | 0 | 0 |  | no_question_asked | planner |
| error | F001139 | 20 |  | False | 1 | 1 | 0.5 | 1 | 0 | 0 | 0 | 0 |  | no_question_asked | planner |
| error | F001221 | 20 |  | False | 1 | 1 | 0.5 | 1 | 0 | 0 | 0 | 0 |  | no_question_asked | planner |
| error | F001294 | 20 |  | False | 1 | 1 | 0.5 | 1 | 0 | 0 | 0 | 0 |  | no_question_asked | planner |
| recovery | F000485 | 544 |  | False | 0.6107 | 0.4261 | 0.1771 | 0.1816 | 0.1914 | 3 | 3 | 3 |  | verdict_override | verdict (system overrides the fused answers) |
| recovery | F001333 | 0 |  | False | 0.5873 | 0.4915 | 0.3249 | 0.2416 | 0.1034 | 2 | 2 | 2 |  | verdict_override | verdict (system overrides the fused answers) |
| recovery | F002772 | 0 |  | False | 0.5873 | 0.492 | 0.3265 | 0.242 | 0.1029 | 2 | 2 | 1 |  | verdict_override | verdict (system overrides the fused answers) |
| recovery | F000874 | 12 | random | True | 0.4089 | 0.5005 | 0.6555 | 0.2495 | 0.0999 | 2 | 2 | 2 |  | verdict_override | verdict (system overrides the fused answers) |
