## Table 5b: ablations (paired on the alerts each arm shares with the reference)

| group | arm | n | brier | brier 95% CI | gap closed vs gfp | brier minus ours_compose | questions/alert | eps spent | no-verdict rate | note |
|---|---|---|---|---|---|---|---|---|---|---|
| reference | One bank only | 771 | 0.1668 | [0.1481, 0.1865] | 0.000 [0.000, 0.000] | -0.0005 [-0.0026, 0.0017] |  |  |  |  |
| reference | See everything (GFP) | 771 | 0.0966 | [0.0728, 0.1227] | 1.000 [1.000, 1.000] | -0.0707 [-0.0973, -0.0435] |  |  |  |  |
| reference | See everything (best SOTA) | 0 |  |  |  |  |  |  |  | not_run: score table 'best_sota_see_everything' is not provided (inputs.yaml scores.best_sota_see_everything.path) |
| raw | Share raw records (3 banks, no noise) | 771 | 0.164 | [0.1480, 0.1804] | 0.040 [-0.107, 0.161] | -0.0033 [-0.0116, 0.0064] | 1.9857 | 0 | 0 |  |
| planners | Random planner | 771 | 0.1698 | [0.1508, 0.1896] | -0.042 [-0.102, 0.005] | 0.0025 [-0.0012, 0.0063] | 5.1245 | 1.834 | 0 |  |
| planners | Fixed-order planner | 771 | 0.1694 | [0.1508, 0.1892] | -0.036 [-0.072, -0.012] | 0.0021 [-0.0005, 0.0048] | 2.3307 | 1.8911 | 0 |  |
| planners | EIG planner | 771 | 0.1823 | [0.1627, 0.2026] | -0.220 [-0.392, -0.114] | 0.0150 [0.0086, 0.0214] | 8.7977 | 1.4426 | 0 |  |
| planners | Learned policy | 771 | 0.1814 | [0.1616, 0.2016] | -0.207 [-0.369, -0.103] | 0.0141 [0.0077, 0.0206] | 7.9883 | 1.4458 | 0 |  |
| planners | Enumerator (no LLM) | 771 | 0.1708 | [0.1514, 0.1907] | -0.057 [-0.178, 0.055] | 0.0035 [-0.0041, 0.0112] | 1.7782 | 0.6868 | 0 |  |
| llm | LLM, fixed list | 208 | 0.1656 | [0.1342, 0.2005] | -0.088 [-0.176, -0.028] | 0.0080 [0.0031, 0.0131] | 1.1827 | 1.5228 | 0 |  |
| llm | LLM, act only | 208 | 0.1592 | [0.1281, 0.1939] | -0.014 [-0.078, 0.047] | 0.0016 [-0.0026, 0.0059] | 1.4567 | 1.4832 | 0 |  |
| llm | LLM, compose (ours, eps 2) | 771 | 0.1673 | [0.1486, 0.1868] | -0.007 [-0.038, 0.026] | 0.0000 [0.0000, 0.0000] | 1.5538 | 1.4569 | 0 |  |
| backtest | No back-testing | 208 | 0.1626 | [0.1312, 0.1973] | -0.053 [-0.110, -0.019] | 0.0049 [0.0003, 0.0099] | 2.7067 | 1.3245 | 0 |  |
| backtest | LLM decides | 208 | 0.1577 | [0.1265, 0.1921] | 0.004 [-0.027, 0.037] | 0.0001 [-0.0035, 0.0038] | 1.524 | 1.4555 | 0 |  |
| backtest | Realistic vs exact back-test LRs | 0 |  |  |  |  |  |  |  | not_run: no run-log rows match {"system": "eig", "variant": ["eig_exact_label_lrs"]} (pending arm: manifest job nonllm.eig_eps_0.5.full_test, nonllm.eig_eps_0.5.subset, nonllm.eig_eps_1.full_test, nonllm.eig_eps_1.subset is not done yet) |
| views | Evidence view: top 40 by amount | 0 |  |  |  |  |  |  |  | not_run: no run-log rows match {"system": "ours_compose", "variant": ["harness_view_top40"]} |
| views | Evidence view: scorer + PCST | 0 |  |  |  |  |  |  |  | not_run: no run-log rows match {"system": "ours_compose", "variant": ["harness_view_scorer_pcst"]} |
| e16 | Fine-tuned planner, seed 0 (E16) | 0 |  |  |  |  |  |  |  | not_run: no run-log rows match {"system": "ours_compose", "variant": ["student_s0"]} |
| e16 | Fine-tuned planner, seed 1 (E16) | 0 |  |  |  |  |  |  |  | not_run: no run-log rows match {"system": "ours_compose", "variant": ["student_s1"]} |
| e16 | Fine-tuned planner, seed 2 (E16) | 0 |  |  |  |  |  |  |  | not_run: no run-log rows match {"system": "ours_compose", "variant": ["student_s2"]} |
| e16 | Same model prompted (E16) | 0 |  |  |  |  |  |  |  | not_run: no run-log rows match {"system": "ours_compose", "variant": ["prompted_qwen3_8b"]} |
| e16 | Teacher (E16) | 0 |  |  |  |  |  |  |  | not_run: no run-log rows match {"system": "ours_compose", "variant": ["harness_teacher"]} |
| e11 | Library by shape, frozen | 771 | 0.1673 | [0.1486, 0.1868] | -0.007 [-0.038, 0.026] | 0.0000 [0.0000, 0.0000] | 1.5538 | 1.4569 | 0 |  |
| e11 | Library by account | 0 |  |  |  |  |  |  |  | not_run: no run-log rows match {"system": "ours_compose", "variant": ["library_account"]} |
| e11 | Library by shape, continual | 0 |  |  |  |  |  |  |  | not_run: no run-log rows match {"system": "ours_compose", "variant": ["harness_lib_shape_continual"]} |
| fusion | Summed log-LRs (naive fusion) | 771 | 0.1823 | [0.1627, 0.2026] | -0.220 [-0.392, -0.114] | 0.0150 [0.0086, 0.0214] | 8.7977 | 1.4426 | 0 |  |
| fusion | Fitted fusion, shared weight | 0 |  |  |  |  |  |  |  | not_run: no run-log rows match {"system": "eig", "variant": ["eig_fusion_shared"]} (pending arm: manifest job nonllm.eig_eps_0.5.full_test, nonllm.eig_eps_0.5.subset, nonllm.eig_eps_1.full_test, nonllm.eig_eps_1.subset is not done yet) |
| fusion | Fitted fusion, per family | 0 |  |  |  |  |  |  |  | not_run: no run-log rows match {"system": "eig", "variant": ["eig_fusion_per_family"]} (pending arm: manifest job nonllm.eig_eps_0.5.full_test, nonllm.eig_eps_0.5.subset, nonllm.eig_eps_1.full_test, nonllm.eig_eps_1.subset is not done yet) |
| switches | Per-account cap 8 | 0 |  |  |  |  |  |  |  | not_run: no run-log rows match {"system": "eig", "variant": ["eig_cap_8"]} (pending arm: manifest job nonllm.eig_eps_0.5.full_test, nonllm.eig_eps_0.5.subset, nonllm.eig_eps_1.full_test, nonllm.eig_eps_1.subset is not done yet) |
| switches | Per-account cap 2 | 0 |  |  |  |  |  |  |  | not_run: no run-log rows match {"system": "eig", "variant": ["eig_cap_2"]} (pending arm: manifest job nonllm.eig_eps_0.5.full_test, nonllm.eig_eps_0.5.subset, nonllm.eig_eps_1.full_test, nonllm.eig_eps_1.subset is not done yet) |
| switches | Question repeated within family | 0 |  |  |  |  |  |  |  | not_run: no run-log rows match {"system": "eig", "variant": ["eig_repeat_family"]} (pending arm: manifest job nonllm.eig_eps_0.5.full_test, nonllm.eig_eps_0.5.subset, nonllm.eig_eps_1.full_test, nonllm.eig_eps_1.subset is not done yet) |
| switches | Stop rule at 0.3 | 0 |  |  |  |  |  |  |  | not_run: no run-log rows match {"system": "eig", "variant": ["eig_stop_rule"]} (pending arm: manifest job nonllm.eig_eps_0.5.full_test, nonllm.eig_eps_0.5.subset, nonllm.eig_eps_1.full_test, nonllm.eig_eps_1.subset is not done yet) |
| switches | Batched 2 then 1 (E13) | 0 |  |  |  |  |  |  |  | not_run: no run-log rows match {"system": "eig", "variant": ["eig_batched_2_1"]} (pending arm: manifest job nonllm.eig_eps_0.5.full_test, nonllm.eig_eps_0.5.subset, nonllm.eig_eps_1.full_test, nonllm.eig_eps_1.subset is not done yet) |
| switches | 20% declines | 0 |  |  |  |  |  |  |  | not_run: no run-log rows match {"system": "eig", "variant": ["eig_decline_20"]} (pending arm: manifest job nonllm.eig_eps_0.5.full_test, nonllm.eig_eps_0.5.subset, nonllm.eig_eps_1.full_test, nonllm.eig_eps_1.subset is not done yet) |
| switches | 50% declines | 0 |  |  |  |  |  |  |  | not_run: no run-log rows match {"system": "eig", "variant": ["eig_decline_50"]} (pending arm: manifest job nonllm.eig_eps_0.5.full_test, nonllm.eig_eps_0.5.subset, nonllm.eig_eps_1.full_test, nonllm.eig_eps_1.subset is not done yet) |
| switches | 20% timeouts | 0 |  |  |  |  |  |  |  | not_run: no run-log rows match {"system": "eig", "variant": ["eig_timeout_20"]} (pending arm: manifest job nonllm.eig_eps_0.5.full_test, nonllm.eig_eps_0.5.subset, nonllm.eig_eps_1.full_test, nonllm.eig_eps_1.subset is not done yet) |
