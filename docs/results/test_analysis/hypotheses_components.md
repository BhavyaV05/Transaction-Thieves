## Every tested component

| hypothesis | component | estimate | ci_lo | ci_hi | p | rule_met |
|---|---|---|---|---|---|---|
| H1 | gap closed vs gfp_see_everything | -0.0066 | -0.0382 | 0.0263 | 0.6745 | False |
| H2 | ratio of benefits | -0.1649 | -2.882 | 0.7502 | 0.9742 | False |
| H3 | B(llm) - B(best simple: fixed_order) | -0.0013 | -0.0063 | 0.0034 | 9.999e-05 | True |
| H3 | B(llm) - B(enumerator) | 0.0007171 | -0.0143 | 0.0156 | 0.1103 | False |
| H3 | B(llm) - B(llm fixed list) | -0.008 | -0.0131 | -0.0031 | 0.0007999 | True |
| H4 | B(ours_compose) - B(no_backtest) | -0.0049 | -0.0099 | -0.0002791 | 0.018 | True |
| H4 | single factor: B(llm_decides) - B(no_backtest) (reported, not part of the test) | -0.0049 | -0.0088 | -0.0014 |  |  |
| H4 | FDR audit: share of accepted questions with LR <= 1 on this period | 0.7143 |  | 0.9633 |  | False |
| H9 | end_to_end vs gfp_see_everything | 0.2663 | -0.0215 | 0.7988 | 0.0007999 | True |
| H9 | questions_only vs gfp_see_everything | -0.1191 | -0.3216 | 0.0377 | 0.2954 | False |
| H7 | B(ours_compose) - B(llm_decides) | -6.877e-05 | -0.0038 | 0.0035 | 0.4936 | False |
