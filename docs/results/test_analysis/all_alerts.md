## All frozen alerts (secondary): non-askable alerts scored with p0; difference = Brier(arm) - Brier(one bank), negative = better

| system | variant | mode | alerts | n | askable | brier | brier one bank | difference | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|---|---|---|
| ours_compose | compose | ported | full_test | 3873 | 771 | 0.1586 | 0.1585 | 9.256e-05 | -0.0003242 | 0.000506 |
| ours_compose | compose | ported | subset | 1000 | 208 | 0.1561 | 0.1562 | -8.63e-05 | -0.0009271 | 0.0007348 |
| ours_fixed_list | fixed_list | ported | subset | 1000 | 208 | 0.1577 | 0.1562 | 0.0016 | 0.0005075 | 0.0027 |
| ours_act_only | act_only | ported | subset | 1000 | 208 | 0.1564 | 0.1562 | 0.0002455 | -0.0007801 | 0.0012 |
| ours_no_backtest | no_backtest | ported | subset | 1000 | 208 | 0.1571 | 0.1562 | 0.0009489 | 0.000349 | 0.0016 |
| ours_llm_decides | llm_decides | ported | subset | 1000 | 208 | 0.1561 | 0.1562 | -6.579e-05 | -0.0005967 | 0.0004463 |
| ours_compose | eps_1 | ported | full_test | 3873 | 771 | 0.1584 | 0.1585 | -5.14e-05 | -0.0004105 | 0.0003079 |
| ours_compose | eps_4 | ported | full_test | 3873 | 771 | 0.1587 | 0.1585 | 0.0002093 | -0.0003553 | 0.0007798 |
| co_investigator_ported | ported | ported | subset | 1000 | 208 | 0.1719 | 0.1562 | 0.0157 | 0.0095 | 0.0224 |
| bed_llm | ported | ported | subset | 1000 | 208 | 0.1608 | 0.1562 | 0.0047 | -0.0005868 | 0.01 |
| bed_llm | ported | ported | full_test | 3873 | 771 | 0.1642 | 0.1585 | 0.0057 | 0.003 | 0.0084 |
| random | random |  | full_test | 3873 | 771 | 0.1591 | 0.1585 | 0.0005864 | -6.469e-05 | 0.0012 |
| random | random |  | subset | 1000 | 208 | 0.1564 | 0.1562 | 0.0002222 | -0.0011 | 0.0016 |
| fixed | fixed_order |  | full_test | 3873 | 771 | 0.159 | 0.1585 | 0.0005043 | 0.0001849 | 0.0008232 |
| fixed | fixed_order |  | subset | 1000 | 208 | 0.1564 | 0.1562 | 0.0001987 | -0.000501 | 0.000875 |
| eig | eig |  | full_test | 3873 | 771 | 0.1615 | 0.1585 | 0.0031 | 0.0017 | 0.0045 |
| eig | eig |  | subset | 1000 | 208 | 0.1602 | 0.1562 | 0.004 | 0.0012 | 0.0068 |
| learned | learned_policy |  | full_test | 3873 | 771 | 0.1614 | 0.1585 | 0.0029 | 0.0015 | 0.0043 |
| learned | learned_policy |  | subset | 1000 | 208 | 0.159 | 0.1562 | 0.0029 | 0.0001539 | 0.0054 |
| enumerator | enumerator |  | full_test | 3873 | 771 | 0.1593 | 0.1585 | 0.0007903 | -0.0007274 | 0.0023 |
| enumerator | enumerator |  | subset | 1000 | 208 | 0.1559 | 0.1562 | -0.0002292 | -0.0031 | 0.0027 |
| raw_share | raw_share |  | subset | 1000 | 208 | 0.1553 | 0.1562 | -0.0009057 | -0.0034 | 0.0017 |
| raw_share | raw_share |  | full_test | 3873 | 771 | 0.1579 | 0.1585 | -0.0005612 | -0.0022 | 0.0014 |
| rewoo_adapted | rewoo_adapted | ported | subset | 1000 | 208 | 0.1574 | 0.1562 | 0.0012 | 0.0001236 | 0.0023 |
| dspy_react_adapted | dspy_react_adapted | ported | subset | 1000 | 208 | 0.1571 | 0.1562 | 0.000902 | -0.0005814 | 0.0023 |
