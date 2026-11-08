# GPU RAG matrix report (eval, 4000 bootstrap resamples)

**val `eval`, scored once for the frozen winners: descriptive, nothing is selected from it.**

## Group planner (n=155, threshold 0.322699)

| arm | brier | log_loss | auc | pr_auc | ece | cal_slope | precision | recall | f1 | escalation_rate | catch_top10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| planner_default (default) | 0.194 | 0.646 | 0.759 | 0.658 | 0.110 | 0.392 | 0.494 | 0.750 | 0.595 | 0.510 | 0.288 |
| planner_combined | 0.194 | 0.646 | 0.760 | 0.658 | 0.105 | 0.395 | 0.494 | 0.750 | 0.595 | 0.510 | 0.288 |
| p0 | 0.195 | 0.645 | 0.753 | 0.655 | 0.114 | 0.386 | 0.474 | 0.692 | 0.562 | 0.490 | 0.288 |

Paired difference (arm - the default (planner_default)), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| planner_combined | -0.0001 [-0.0011, +0.0008] p_holm 0.885 | -0.0007 [-0.0043, +0.0021] p_holm 0.737 | +0.0009 [-0.0051, +0.0088] p_holm 0.886 | -0.0001 [-0.0031, +0.0029] p_holm 0.947 | -0.0048 [-0.0212, +0.0361] p_holm 0.644 | +0.0000 [+0.0000, +0.0000] p_holm 1.000 |

Paired difference (arm - p0), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| planner_default | -0.0004 [-0.0061, +0.0053] p_holm 1.000 | +0.0008 [-0.0128, +0.0145] p_holm 1.000 | +0.0054 [-0.0099, +0.0212] p_holm 0.830 | +0.0028 [-0.0115, +0.0171] p_holm 1.000 | -0.0047 [-0.0533, +0.0386] p_holm 1.000 | +0.0329 [-0.0035, +0.0781] p_holm 0.170 |
| planner_combined | -0.0005 [-0.0064, +0.0052] p_holm 1.000 | +0.0001 [-0.0138, +0.0142] p_holm 1.000 | +0.0063 [-0.0087, +0.0215] p_holm 0.830 | +0.0027 [-0.0115, +0.0169] p_holm 1.000 | -0.0094 [-0.0483, +0.0473] p_holm 1.000 | +0.0329 [-0.0035, +0.0781] p_holm 0.170 |

Flipped calls vs p0 at the threshold (right-to-wrong / wrong-to-right), no-verdict rate, collapse checks:

| arm | flips | no verdict | var(p)/var(p0) | AUC | AUC p0 | collapsed |
|---|---|---|---|---|---|---|
| planner_default | 2 / 5 | 0.000 | 0.985 | 0.759 | 0.753 | False |
| planner_combined | 2 / 5 | 0.000 | 0.983 | 0.760 | 0.753 | False |

Per alert (means): questions, accepted-question share, eps, LLM calls, prompt tokens in / per call, tokens out, median / p90 s, GPU h.

| arm | questions | accepted | eps | calls | tok in | tok/call | tok out | median s | p90 s | GPU h |
|---|---|---|---|---|---|---|---|---|---|---|
| planner_default | 1.64 | 0.63 | 1.50 | 2.40 | 12240 | 5100 | 317 | 84.2 | 183.6 | 4.08 |
| planner_combined | 1.62 | 0.63 | 1.48 | 2.37 | 12101 | 5097 | 315 | 83.4 | 186.7 | 4.05 |


## Group sharma_native (n=155, threshold 0.322699)

| arm | brier | log_loss | auc | pr_auc | ece | cal_slope | precision | recall | f1 | escalation_rate | catch_top10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sharma_native_mem_on (default) | 0.237 | 0.729 | 0.672 | 0.469 | 0.197 | 0.359 | 0.583 | 0.538 | 0.560 | 0.310 | 0.269 |
| p0 | 0.195 | 0.645 | 0.753 | 0.655 | 0.114 | 0.386 | 0.474 | 0.692 | 0.562 | 0.490 | 0.288 |

Paired difference (arm - p0), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| sharma_native_mem_on | +0.0425 [+0.0064, +0.0803] p_holm 0.022 | +0.0836 [-0.1264, +0.2401] p_holm 0.341 | -0.0811 [-0.1433, -0.0161] p_holm 0.016 | -0.1858 [-0.2713, -0.1039] p_holm 0.000 | +0.0823 [-0.0394, +0.0950] p_holm 0.431 | -0.0025 [-0.0980, +0.0884] p_holm 0.934 |

Flipped calls vs p0 at the threshold (right-to-wrong / wrong-to-right), no-verdict rate, collapse checks:

| arm | flips | no verdict | var(p)/var(p0) | AUC | AUC p0 | collapsed |
|---|---|---|---|---|---|---|
| sharma_native_mem_on | 10 / 22 | 0.000 | 1.568 | 0.672 | 0.753 | True |

Per alert (means): questions, accepted-question share, eps, LLM calls, prompt tokens in / per call, tokens out, median / p90 s, GPU h.

| arm | questions | accepted | eps | calls | tok in | tok/call | tok out | median s | p90 s | GPU h |
|---|---|---|---|---|---|---|---|---|---|---|
| sharma_native_mem_on | 0.00 | - | 0.00 | 5.00 | 3691 | 738 | 351 | 15.8 | 17.0 | 0.67 |


## Group sharma_ported (n=155, threshold 0.322699)

| arm | brier | log_loss | auc | pr_auc | ece | cal_slope | precision | recall | f1 | escalation_rate | catch_top10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sharma_ported_mem_on (default) | 0.233 | 0.717 | 0.666 | 0.466 | 0.198 | 0.365 | 0.605 | 0.500 | 0.547 | 0.277 | 0.231 |
| p0 | 0.195 | 0.645 | 0.753 | 0.655 | 0.114 | 0.386 | 0.474 | 0.692 | 0.562 | 0.490 | 0.288 |

Paired difference (arm - p0), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| sharma_ported_mem_on | +0.0381 [-0.0028, +0.0794] p_holm 0.069 | +0.0717 [-0.1388, +0.2363] p_holm 0.439 | -0.0874 [-0.1562, -0.0143] p_holm 0.022 | -0.1892 [-0.2787, -0.1045] p_holm 0.000 | +0.0841 [-0.0489, +0.0906] p_holm 0.583 | -0.0151 [-0.1225, +0.0864] p_holm 0.781 |

Flipped calls vs p0 at the threshold (right-to-wrong / wrong-to-right), no-verdict rate, collapse checks:

| arm | flips | no verdict | var(p)/var(p0) | AUC | AUC p0 | collapsed |
|---|---|---|---|---|---|---|
| sharma_ported_mem_on | 13 / 26 | 0.000 | 1.453 | 0.666 | 0.753 | True |

Per alert (means): questions, accepted-question share, eps, LLM calls, prompt tokens in / per call, tokens out, median / p90 s, GPU h.

| arm | questions | accepted | eps | calls | tok in | tok/call | tok out | median s | p90 s | GPU h |
|---|---|---|---|---|---|---|---|---|---|---|
| sharma_ported_mem_on | 0.65 | 0.44 | 0.28 | 5.60 | 16416 | 2931 | 389 | 21.2 | 27.7 | 0.96 |


## Group torres_native (n=155, threshold 0.322699)

| arm | brier | log_loss | auc | pr_auc | ece | cal_slope | precision | recall | f1 | escalation_rate | catch_top10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torres_native_mem_on (default) | 0.413 | 1.170 | 0.611 | 0.413 | 0.430 | 115630967.188 | 0.347 | 0.981 | 0.513 | 0.948 | 0.212 |
| p0 | 0.195 | 0.645 | 0.753 | 0.655 | 0.114 | 0.386 | 0.474 | 0.692 | 0.562 | 0.490 | 0.288 |

Paired difference (arm - p0), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| torres_native_mem_on | +0.2184 [+0.1492, +0.2860] p_holm 0.000 | +0.5242 [+0.2687, +0.7356] p_holm 0.000 | -0.1426 [-0.2388, -0.0465] p_holm 0.005 | -0.2418 [-0.3598, -0.1301] p_holm 0.001 | +0.3160 [+0.1812, +0.3383] p_holm 0.000 | -0.0499 [-0.1365, +0.0411] p_holm 0.274 |

Flipped calls vs p0 at the threshold (right-to-wrong / wrong-to-right), no-verdict rate, collapse checks:

| arm | flips | no verdict | var(p)/var(p0) | AUC | AUC p0 | collapsed |
|---|---|---|---|---|---|---|
| torres_native_mem_on | 56 / 15 | 0.110 | 0.515 | 0.611 | 0.753 | True |

Per alert (means): questions, accepted-question share, eps, LLM calls, prompt tokens in / per call, tokens out, median / p90 s, GPU h.

| arm | questions | accepted | eps | calls | tok in | tok/call | tok out | median s | p90 s | GPU h |
|---|---|---|---|---|---|---|---|---|---|---|
| torres_native_mem_on | 0.00 | - | 0.00 | 3.01 | 3738 | 1243 | 1187 | 52.4 | 78.9 | 2.35 |


## Group torres_ported (n=155, threshold 0.322699)

| arm | brier | log_loss | auc | pr_auc | ece | cal_slope | precision | recall | f1 | escalation_rate | catch_top10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torres_ported_mem_on (default) | 0.434 | 1.230 | 0.555 | 0.366 | 0.431 | 100815568.924 | 0.342 | 0.962 | 0.505 | 0.942 | 0.192 |
| p0 | 0.195 | 0.645 | 0.753 | 0.655 | 0.114 | 0.386 | 0.474 | 0.692 | 0.562 | 0.490 | 0.288 |

Paired difference (arm - p0), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| torres_ported_mem_on | +0.2395 [+0.1722, +0.3052] p_holm 0.000 | +0.5844 [+0.3274, +0.8026] p_holm 0.000 | -0.1978 [-0.2897, -0.1034] p_holm 0.001 | -0.2884 [-0.4017, -0.1722] p_holm 0.000 | +0.3166 [+0.1921, +0.3490] p_holm 0.000 | -0.0574 [-0.1419, +0.0329] p_holm 0.202 |

Flipped calls vs p0 at the threshold (right-to-wrong / wrong-to-right), no-verdict rate, collapse checks:

| arm | flips | no verdict | var(p)/var(p0) | AUC | AUC p0 | collapsed |
|---|---|---|---|---|---|---|
| torres_ported_mem_on | 56 / 14 | 0.103 | 0.548 | 0.555 | 0.753 | True |

Per alert (means): questions, accepted-question share, eps, LLM calls, prompt tokens in / per call, tokens out, median / p90 s, GPU h.

| arm | questions | accepted | eps | calls | tok in | tok/call | tok out | median s | p90 s | GPU h |
|---|---|---|---|---|---|---|---|---|---|---|
| torres_ported_mem_on | 2.50 | 0.31 | 0.88 | 3.99 | 7723 | 1937 | 1391 | 85.2 | 125.8 | 3.78 |


## Group co_investigator_native (n=155, threshold 0.322699)

| arm | brier | log_loss | auc | pr_auc | ece | cal_slope | precision | recall | f1 | escalation_rate | catch_top10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| co_investigator_native_mem_on (default) | 0.257 | 0.725 | 0.706 | 0.566 | 0.210 | 0.463 | 0.422 | 0.827 | 0.558 | 0.658 | 0.269 |
| p0 | 0.195 | 0.645 | 0.753 | 0.655 | 0.114 | 0.386 | 0.474 | 0.692 | 0.562 | 0.490 | 0.288 |

Paired difference (arm - p0), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| co_investigator_native_mem_on | +0.0622 [+0.0185, +0.1052] p_holm 0.005 | +0.0792 [-0.1305, +0.2407] p_holm 0.381 | -0.0472 [-0.1290, +0.0395] p_holm 0.269 | -0.0891 [-0.1963, +0.0159] p_holm 0.087 | +0.0957 [+0.0084, +0.1379] p_holm 0.029 | -0.0041 [-0.0741, +0.0705] p_holm 0.897 |

Flipped calls vs p0 at the threshold (right-to-wrong / wrong-to-right), no-verdict rate, collapse checks:

| arm | flips | no verdict | var(p)/var(p0) | AUC | AUC p0 | collapsed |
|---|---|---|---|---|---|---|
| co_investigator_native_mem_on | 22 / 10 | 0.606 | 1.001 | 0.706 | 0.753 | True |

Per alert (means): questions, accepted-question share, eps, LLM calls, prompt tokens in / per call, tokens out, median / p90 s, GPU h.

| arm | questions | accepted | eps | calls | tok in | tok/call | tok out | median s | p90 s | GPU h |
|---|---|---|---|---|---|---|---|---|---|---|
| co_investigator_native_mem_on | 0.00 | - | 0.00 | 6.52 | 9814 | 1506 | 1682 | 82.3 | 107.8 | 3.51 |


## Group co_investigator_ported (n=155, threshold 0.322699)

| arm | brier | log_loss | auc | pr_auc | ece | cal_slope | precision | recall | f1 | escalation_rate | catch_top10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| co_investigator_ported_mem_on (default) | 0.272 | 0.763 | 0.655 | 0.507 | 0.240 | 0.344 | 0.389 | 0.712 | 0.503 | 0.613 | 0.192 |
| p0 | 0.195 | 0.645 | 0.753 | 0.655 | 0.114 | 0.386 | 0.474 | 0.692 | 0.562 | 0.490 | 0.288 |

Paired difference (arm - p0), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| co_investigator_ported_mem_on | +0.0777 [+0.0393, +0.1163] p_holm 0.000 | +0.1173 [-0.0938, +0.2715] p_holm 0.213 | -0.0983 [-0.1701, -0.0227] p_holm 0.012 | -0.1475 [-0.2473, -0.0483] p_holm 0.004 | +0.1251 [+0.0302, +0.1443] p_holm 0.003 | -0.0591 [-0.1172, -0.0009] p_holm 0.049 |

Flipped calls vs p0 at the threshold (right-to-wrong / wrong-to-right), no-verdict rate, collapse checks:

| arm | flips | no verdict | var(p)/var(p0) | AUC | AUC p0 | collapsed |
|---|---|---|---|---|---|---|
| co_investigator_ported_mem_on | 21 / 4 | 0.626 | 1.016 | 0.655 | 0.753 | True |

Per alert (means): questions, accepted-question share, eps, LLM calls, prompt tokens in / per call, tokens out, median / p90 s, GPU h.

| arm | questions | accepted | eps | calls | tok in | tok/call | tok out | median s | p90 s | GPU h |
|---|---|---|---|---|---|---|---|---|---|---|
| co_investigator_ported_mem_on | 2.28 | 0.26 | 0.98 | 7.57 | 13721 | 1812 | 1822 | 105.1 | 135.7 | 4.50 |


## Group finfre_rag_native (n=155, threshold 0.322699)

| arm | brier | log_loss | auc | pr_auc | ece | cal_slope | precision | recall | f1 | escalation_rate | catch_top10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| finfre_rag_native_pool_bal (default) | 0.283 | 0.863 | 0.699 | 0.470 | 0.265 | 0.441 | 0.449 | 0.846 | 0.587 | 0.632 | 0.173 |
| p0 | 0.195 | 0.645 | 0.753 | 0.655 | 0.114 | 0.386 | 0.474 | 0.692 | 0.562 | 0.490 | 0.288 |

Paired difference (arm - p0), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| finfre_rag_native_pool_bal | +0.0881 [+0.0344, +0.1445] p_holm 0.001 | +0.2179 [-0.0042, +0.4164] p_holm 0.054 | -0.0542 [-0.1342, +0.0275] p_holm 0.173 | -0.1848 [-0.2949, -0.0825] p_holm 0.000 | +0.1504 [+0.0374, +0.1729] p_holm 0.006 | +0.0242 [-0.0621, +0.1107] p_holm 0.597 |

Flipped calls vs p0 at the threshold (right-to-wrong / wrong-to-right), no-verdict rate, collapse checks:

| arm | flips | no verdict | var(p)/var(p0) | AUC | AUC p0 | collapsed |
|---|---|---|---|---|---|---|
| finfre_rag_native_pool_bal | 29 / 23 | 0.000 | 1.281 | 0.699 | 0.753 | True |

Per alert (means): questions, accepted-question share, eps, LLM calls, prompt tokens in / per call, tokens out, median / p90 s, GPU h.

| arm | questions | accepted | eps | calls | tok in | tok/call | tok out | median s | p90 s | GPU h |
|---|---|---|---|---|---|---|---|---|---|---|
| finfre_rag_native_pool_bal | 0.00 | - | 0.00 | 1.00 | 956 | 956 | 172 | 7.0 | 19.5 | 0.41 |


## Group finfre_rag_ported (n=155, threshold 0.322699)

| arm | brier | log_loss | auc | pr_auc | ece | cal_slope | precision | recall | f1 | escalation_rate | catch_top10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| finfre_rag_ported_pool_bal (default) | 0.302 | 0.887 | 0.711 | 0.487 | 0.293 | 0.501 | 0.429 | 0.865 | 0.573 | 0.677 | 0.192 |
| p0 | 0.195 | 0.645 | 0.753 | 0.655 | 0.114 | 0.386 | 0.474 | 0.692 | 0.562 | 0.490 | 0.288 |

Paired difference (arm - p0), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| finfre_rag_ported_pool_bal | +0.1076 [+0.0502, +0.1655] p_holm 0.000 | +0.2413 [+0.0152, +0.4447] p_holm 0.036 | -0.0423 [-0.1261, +0.0398] p_holm 0.323 | -0.1679 [-0.2841, -0.0659] p_holm 0.001 | +0.1785 [+0.0588, +0.2027] p_holm 0.001 | +0.0107 [-0.0824, +0.1024] p_holm 0.828 |

Flipped calls vs p0 at the threshold (right-to-wrong / wrong-to-right), no-verdict rate, collapse checks:

| arm | flips | no verdict | var(p)/var(p0) | AUC | AUC p0 | collapsed |
|---|---|---|---|---|---|---|
| finfre_rag_ported_pool_bal | 34 / 23 | 0.006 | 1.214 | 0.711 | 0.753 | True |

Per alert (means): questions, accepted-question share, eps, LLM calls, prompt tokens in / per call, tokens out, median / p90 s, GPU h.

| arm | questions | accepted | eps | calls | tok in | tok/call | tok out | median s | p90 s | GPU h |
|---|---|---|---|---|---|---|---|---|---|---|
| finfre_rag_ported_pool_bal | 2.52 | 0.29 | 1.09 | 2.03 | 3700 | 1827 | 361 | 27.4 | 45.2 | 1.36 |

