## Cost per system: LLM calls, tokens, GPU-seconds, eps

| system | variant | mode | alerts | n | LLM calls/alert | tokens in/alert | tokens out/alert | GPU-s/alert | GPU-s total (all seeds) | eps spent/alert | questions/alert | sim days/alert |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours_compose | compose | ported | full_test | 771 | 2.3969 | 12553.4034 | 313.9364 | 29.0178 | 22372.7559 | 1.4569 | 1.5538 | 1.3989 |
| ours_compose | compose | ported | subset | 208 | 2.4279 | 12617.8029 | 318.4423 | 28.4252 | 5912.4506 | 1.4675 | 1.5288 | 1.2704 |
| ours_fixed_list | fixed_list | ported | subset | 208 | 1.2404 | 3947.3942 | 136.7933 | 8.9662 | 1864.9637 | 1.5228 | 1.1827 | 1.0023 |
| ours_act_only | act_only | ported | subset | 208 | 2.0337 | 10450.1683 | 158.6779 | 22.3577 | 4650.3913 | 1.4832 | 1.4567 | 1.2764 |
| ours_no_backtest | no_backtest | ported | subset | 208 | 5.3894 | 26464.3413 | 744.125 | 60.1361 | 12508.3032 | 1.3245 | 2.7067 | 2.3007 |
| ours_llm_decides | llm_decides | ported | subset | 208 | 2.4471 | 12738.2067 | 351.8558 | 29.1956 | 6072.6766 | 1.4555 | 1.524 | 1.3436 |
| ours_compose | eps_1 | ported | full_test | 771 | 1.1505 | 6031.2542 | 154.4501 | 14.0904 | 10863.7079 | 0.7438 | 0.8171 | 0.7018 |
| ours_compose | eps_4 | ported | full_test | 771 | 4.1492 | 21630.5875 | 545.917 | 49.8462 | 38431.3849 | 2.6706 | 2.3709 | 2.0262 |
| co_investigator_ported | ported | ported | subset | 208 | 7.5721 | 14992.2212 | 1870.9567 | 38.6535 | 8039.9219 | 0.9712 | 2.2692 | 0.976 |
| bed_llm | ported | ported | subset | 208 | 6.9423 | 20076.5962 | 1429.2452 | 32.8837 | 6839.8141 | 0.6659 | 2.7933 | 0.6318 |
| bed_llm | ported | ported | full_test | 771 | 6.9092 | 20151.6783 | 1409.4747 | 32.6181 | 25148.5455 | 0.6122 | 2.7613 | 0.7346 |
| random | random |  | full_test | 771 | 0 | 0 | 0 | 0 | 0 | 1.834 | 5.1245 | 2.6461 |
| random | random |  | subset | 208 | 0 | 0 | 0 | 0 | 0 | 1.8546 | 5.3077 | 2.655 |
| fixed | fixed_order |  | full_test | 771 | 0 | 0 | 0 | 0 | 0 | 1.8911 | 2.3307 | 1.8871 |
| fixed | fixed_order |  | subset | 208 | 0 | 0 | 0 | 0 | 0 | 1.8798 | 2.5673 | 1.8665 |
| eig | eig |  | full_test | 771 | 0 | 0 | 0 | 0 | 0 | 1.4426 | 8.7977 | 3.7446 |
| eig | eig |  | subset | 208 | 0 | 0 | 0 | 0 | 0 | 1.3498 | 8.8846 | 3.6859 |
| learned | learned_policy |  | full_test | 771 | 0 | 0 | 0 | 0 | 0 | 1.4458 | 7.9883 | 3.752 |
| learned | learned_policy |  | subset | 208 | 0 | 0 | 0 | 0 | 0 | 1.3606 | 8.774 | 3.7852 |
| enumerator | enumerator |  | full_test | 771 | 0 | 0 | 0 | 0 | 0 | 0.6868 | 1.7782 | 1.3982 |
| enumerator | enumerator |  | subset | 208 | 0 | 0 | 0 | 0 | 0 | 0.6743 | 1.6635 | 1.2718 |
| raw_share | raw_share |  | subset | 208 | 0 | 0 | 0 | 0 | 0 | 0 | 1.8894 | 0 |
| raw_share | raw_share |  | full_test | 771 | 0 | 0 | 0 | 0 | 0 | 0 | 1.9857 | 0 |
| rewoo_adapted | rewoo_adapted | ported | subset | 208 | 0.8702 | 1736.8846 | 62.6875 | 4.1195 | 856.8639 | 0.8462 | 0.8894 | 0.8445 |
| dspy_react_adapted | dspy_react_adapted | ported | subset | 208 | 1.3606 | 2855.9808 | 179.0865 | 7.4187 | 1543.0864 | 1.6286 | 1.2981 | 1.0754 |
