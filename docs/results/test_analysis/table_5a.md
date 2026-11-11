## Table 5a: external systems (end-to-end and questions-only gap closed, 95% cluster-bootstrap CI)

| group | system | tier | mode | alerts | n | brier e2e | gap e2e vs gfp | gap questions-only vs gfp | no-verdict rate | LLM calls | tokens | GPU-s | eps spent | questions/alert |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours | ours_compose | core | ported | subset | 208 | 0.1576 | 0.005 [-0.044, 0.058] | -1.180 [-2.483, -0.545] | 0 | 2.4279 | 12936.2452 | 28.4252 | 1.4675 | 1.5288 |
| ours | ours_compose | core | ported | full_test | 771 | 0.1673 | -0.007 [-0.038, 0.026] | -1.379 [-2.428, -0.856] | 0 | 2.3969 | 12867.3398 | 29.0178 | 1.4569 | 1.5538 |
| C | co_investigator | secondary | ported | subset | 208 | 0.2334 | -0.878 [-1.709, -0.471] | -1.065 [-2.246, -0.485] | 0.7115 | 7.5721 | 16863.1779 | 38.6535 | 0.9712 | 2.2692 |
| D | bed_llm | secondary | ported | subset | 208 | 0.1805 | -0.262 [-0.797, 0.026] | -1.061 [-2.255, -0.474] | 0 | 6.9423 | 21505.8413 | 32.8837 | 0.6659 | 2.7933 |
| D | bed_llm | secondary | ported | full_test | 771 | 0.1956 | -0.410 [-0.831, -0.176] | -1.181 [-2.116, -0.708] | 0 | 6.9092 | 21561.153 | 32.6181 | 0.6122 | 2.7613 |
| D | dspy_react_adapted | new_arm | ported | subset | 208 | 0.1623 | -0.051 [-0.149, 0.033] | -1.607 [-3.241, -0.822] | 0 | 1.3606 | 3035.0673 | 7.4187 | 1.6286 | 1.2981 |
| D | rewoo_adapted | new_arm | ported | subset | 208 | 0.1638 | -0.068 [-0.145, -0.007] | -1.494 [-2.994, -0.767] | 0 | 0.8702 | 1799.5721 | 4.1195 | 0.8462 | 0.8894 |
