# GPU RAG matrix report (fit+cal, 4000 bootstrap resamples)

## Group finfre_rag_native (n=300, threshold 0.322699)

| arm | brier | log_loss | auc | pr_auc | ece | cal_slope | precision | recall | f1 | escalation_rate | catch_top10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| finfre_rag_native_pool_bal (default) | 0.247 | 0.720 | 0.724 | 0.709 | 0.199 | 0.440 | 0.636 | 0.763 | 0.694 | 0.623 | 0.186 |
| finfre_rag_native_pool_nat | 0.257 | 0.763 | 0.729 | 0.699 | 0.206 | 0.434 | 0.739 | 0.526 | 0.614 | 0.370 | 0.167 |
| finfre_rag_native_pool_033 | 0.222 | 0.667 | 0.748 | 0.708 | 0.161 | 0.526 | 0.665 | 0.827 | 0.737 | 0.647 | 0.179 |
| finfre_rag_native_zero_shot | 0.316 | 0.900 | 0.644 | 0.635 | 0.279 | 0.512 | 0.816 | 0.199 | 0.320 | 0.127 | 0.167 |
| p0 | 0.157 | 0.623 | 0.855 | 0.838 | 0.064 | 0.427 | 0.725 | 0.878 | 0.794 | 0.630 | 0.179 |

Paired difference (arm - the default (finfre_rag_native_pool_bal)), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| finfre_rag_native_pool_nat | +0.0092 [-0.0363, +0.0540] p_holm 0.719 | +0.0431 [-0.0812, +0.1678] p_holm 0.556 | +0.0054 [-0.0576, +0.0679] p_holm 0.846 | -0.0096 [-0.0680, +0.0479] p_holm 1.000 | +0.0073 [-0.0473, +0.0812] p_holm 0.616 | -0.0796 [-0.1505, -0.0081] p_holm 0.060 |
| finfre_rag_native_pool_033 | -0.0249 [-0.0617, +0.0111] p_holm 0.336 | -0.0524 [-0.1475, +0.0409] p_holm 0.556 | +0.0241 [-0.0272, +0.0782] p_holm 0.734 | -0.0006 [-0.0452, +0.0448] p_holm 1.000 | -0.0387 [-0.0935, +0.0160] p_holm 0.304 | +0.0433 [-0.0070, +0.0953] p_holm 0.090 |
| finfre_rag_native_zero_shot | +0.0683 [+0.0131, +0.1241] p_holm 0.054 | +0.1802 [+0.0315, +0.3311] p_holm 0.061 | -0.0802 [-0.1517, -0.0080] p_holm 0.106 | -0.0734 [-0.1349, -0.0100] p_holm 0.079 | +0.0797 [+0.0157, +0.1767] p_holm 0.066 | -0.3743 [-0.4716, -0.2747] p_holm 0.001 |

Paired difference (arm - p0), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| finfre_rag_native_pool_bal | +0.0902 [+0.0537, +0.1252] p_holm 0.002 | +0.0968 [-0.0911, +0.2561] p_holm 0.555 | -0.1314 [-0.1875, -0.0772] p_holm 0.002 | -0.1294 [-0.1881, -0.0745] p_holm 0.002 | +0.1350 [+0.0269, +0.1437] p_holm 0.008 | -0.1003 [-0.1558, -0.0453] p_holm 0.002 |
| finfre_rag_native_pool_nat | +0.0994 [+0.0682, +0.1296] p_holm 0.002 | +0.1398 [-0.0559, +0.2992] p_holm 0.406 | -0.1260 [-0.1690, -0.0833] p_holm 0.002 | -0.1390 [-0.1912, -0.0876] p_holm 0.002 | +0.1423 [+0.0525, +0.1567] p_holm 0.002 | -0.1800 [-0.2448, -0.1150] p_holm 0.002 |
| finfre_rag_native_pool_033 | +0.0652 [+0.0330, +0.0988] p_holm 0.002 | +0.0444 [-0.1310, +0.2001] p_holm 0.581 | -0.1073 [-0.1577, -0.0591] p_holm 0.002 | -0.1300 [-0.1852, -0.0802] p_holm 0.002 | +0.0963 [-0.0040, +0.1052] p_holm 0.072 | -0.0571 [-0.1045, -0.0130] p_holm 0.012 |
| finfre_rag_native_zero_shot | +0.1585 [+0.1194, +0.1986] p_holm 0.002 | +0.2770 [+0.0520, +0.4710] p_holm 0.072 | -0.2115 [-0.2690, -0.1542] p_holm 0.002 | -0.2029 [-0.2689, -0.1360] p_holm 0.002 | +0.2146 [+0.1249, +0.2373] p_holm 0.002 | -0.4746 [-0.5669, -0.3853] p_holm 0.002 |

Flipped calls vs p0 at the threshold (right-to-wrong / wrong-to-right), no-verdict rate, collapse checks:

| arm | flips | no verdict | var(p)/var(p0) | AUC | AUC p0 | collapsed |
|---|---|---|---|---|---|---|
| finfre_rag_native_pool_bal | 64 / 30 | 0.000 | 1.097 | 0.724 | 0.855 | True |
| finfre_rag_native_pool_nat | 65 / 33 | 0.000 | 1.171 | 0.729 | 0.855 | True |
| finfre_rag_native_pool_033 | 48 / 27 | 0.000 | 1.086 | 0.748 | 0.855 | True |
| finfre_rag_native_zero_shot | 108 / 47 | 0.000 | 0.543 | 0.644 | 0.855 | True |

Per alert (means): questions, accepted-question share, eps, LLM calls, prompt tokens in / per call, tokens out, median / p90 s, GPU h.

| arm | questions | accepted | eps | calls | tok in | tok/call | tok out | median s | p90 s | GPU h |
|---|---|---|---|---|---|---|---|---|---|---|
| finfre_rag_native_pool_bal | 0.00 | - | 0.00 | 1.01 | 1046 | 1036 | 179 | 7.6 | 19.3 | 0.84 |
| finfre_rag_native_pool_nat | 0.00 | - | 0.00 | 1.00 | 1047 | 1047 | 155 | 7.3 | 19.2 | 0.78 |
| finfre_rag_native_pool_033 | 0.00 | - | 0.00 | 1.01 | 1162 | 1150 | 181 | 8.0 | 20.0 | 0.90 |
| finfre_rag_native_zero_shot | 0.00 | - | 0.00 | 1.00 | 181 | 181 | 156 | 3.9 | 6.2 | 0.35 |


## Group finfre_rag_ported (n=300, threshold 0.322699)

| arm | brier | log_loss | auc | pr_auc | ece | cal_slope | precision | recall | f1 | escalation_rate | catch_top10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| finfre_rag_ported_pool_bal (default) | 0.253 | 0.742 | 0.710 | 0.689 | 0.202 | 0.426 | 0.623 | 0.814 | 0.706 | 0.680 | 0.192 |
| finfre_rag_ported_pool_nat | 0.254 | 0.783 | 0.704 | 0.688 | 0.195 | 0.397 | 0.739 | 0.564 | 0.640 | 0.397 | 0.160 |
| finfre_rag_ported_pool_033 | 0.220 | 0.654 | 0.753 | 0.717 | 0.153 | 0.559 | 0.660 | 0.846 | 0.742 | 0.667 | 0.179 |
| finfre_rag_ported_zero_shot | 0.298 | 0.834 | 0.597 | 0.604 | 0.255 | 0.600 | 0.857 | 0.231 | 0.364 | 0.140 | 0.173 |
| p0 | 0.157 | 0.623 | 0.855 | 0.838 | 0.064 | 0.427 | 0.725 | 0.878 | 0.794 | 0.630 | 0.179 |

Paired difference (arm - the default (finfre_rag_ported_pool_bal)), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| finfre_rag_ported_pool_nat | +0.0015 [-0.0500, +0.0545] p_holm 0.991 | +0.0412 [-0.1026, +0.1927] p_holm 0.598 | -0.0063 [-0.0776, +0.0646] p_holm 0.887 | -0.0013 [-0.0642, +0.0635] p_holm 0.991 | -0.0070 [-0.0730, +0.0735] p_holm 0.975 | -0.0656 [-0.1424, +0.0095] p_holm 0.174 |
| finfre_rag_ported_pool_033 | -0.0334 [-0.0710, +0.0035] p_holm 0.213 | -0.0882 [-0.1858, +0.0082] p_holm 0.219 | +0.0434 [-0.0108, +0.0990] p_holm 0.228 | +0.0286 [-0.0170, +0.0757] p_holm 0.413 | -0.0489 [-0.0967, +0.0064] p_holm 0.261 | +0.0360 [-0.0124, +0.0853] p_holm 0.174 |
| finfre_rag_ported_zero_shot | +0.0449 [-0.0095, +0.0993] p_holm 0.234 | +0.0920 [-0.0507, +0.2364] p_holm 0.429 | -0.1125 [-0.1769, -0.0463] p_holm 0.006 | -0.0854 [-0.1431, -0.0277] p_holm 0.018 | +0.0532 [-0.0265, +0.1473] p_holm 0.364 | -0.3419 [-0.4376, -0.2417] p_holm 0.001 |

Paired difference (arm - p0), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| finfre_rag_ported_pool_bal | +0.0958 [+0.0575, +0.1334] p_holm 0.002 | +0.1193 [-0.0659, +0.2821] p_holm 0.385 | -0.1451 [-0.2007, -0.0895] p_holm 0.002 | -0.1494 [-0.2057, -0.0964] p_holm 0.002 | +0.1373 [+0.0260, +0.1485] p_holm 0.011 | -0.0886 [-0.1435, -0.0345] p_holm 0.002 |
| finfre_rag_ported_pool_nat | +0.0972 [+0.0630, +0.1320] p_holm 0.002 | +0.1605 [-0.0516, +0.3364] p_holm 0.385 | -0.1514 [-0.2023, -0.0998] p_holm 0.002 | -0.1507 [-0.2118, -0.0869] p_holm 0.002 | +0.1303 [+0.0373, +0.1429] p_holm 0.004 | -0.1542 [-0.2209, -0.0900] p_holm 0.002 |
| finfre_rag_ported_pool_033 | +0.0624 [+0.0305, +0.0962] p_holm 0.002 | +0.0311 [-0.1504, +0.1867] p_holm 0.692 | -0.1017 [-0.1513, -0.0538] p_holm 0.002 | -0.1208 [-0.1730, -0.0695] p_holm 0.002 | +0.0884 [-0.0101, +0.1001] p_holm 0.116 | -0.0526 [-0.1007, -0.0069] p_holm 0.023 |
| finfre_rag_ported_zero_shot | +0.1407 [+0.1037, +0.1778] p_holm 0.002 | +0.2114 [-0.0095, +0.3958] p_holm 0.242 | -0.2576 [-0.3077, -0.2052] p_holm 0.002 | -0.2348 [-0.2993, -0.1675] p_holm 0.002 | +0.1905 [+0.0911, +0.2058] p_holm 0.002 | -0.4306 [-0.5235, -0.3406] p_holm 0.002 |

Flipped calls vs p0 at the threshold (right-to-wrong / wrong-to-right), no-verdict rate, collapse checks:

| arm | flips | no verdict | var(p)/var(p0) | AUC | AUC p0 | collapsed |
|---|---|---|---|---|---|---|
| finfre_rag_ported_pool_bal | 66 / 31 | 0.000 | 1.020 | 0.710 | 0.855 | True |
| finfre_rag_ported_pool_nat | 62 / 34 | 0.000 | 1.180 | 0.704 | 0.855 | True |
| finfre_rag_ported_pool_033 | 50 / 29 | 0.007 | 1.035 | 0.753 | 0.855 | True |
| finfre_rag_ported_zero_shot | 102 / 47 | 0.000 | 0.405 | 0.597 | 0.855 | True |

Per alert (means): questions, accepted-question share, eps, LLM calls, prompt tokens in / per call, tokens out, median / p90 s, GPU h.

| arm | questions | accepted | eps | calls | tok in | tok/call | tok out | median s | p90 s | GPU h |
|---|---|---|---|---|---|---|---|---|---|---|
| finfre_rag_ported_pool_bal | 2.55 | 0.24 | 1.03 | 2.06 | 3851 | 1872 | 376 | 27.4 | 46.5 | 2.69 |
| finfre_rag_ported_pool_nat | 2.41 | 0.25 | 1.08 | 2.03 | 3795 | 1872 | 344 | 27.6 | 45.0 | 2.59 |
| finfre_rag_ported_pool_033 | 2.43 | 0.27 | 1.10 | 2.07 | 3961 | 1917 | 393 | 32.9 | 45.3 | 2.76 |
| finfre_rag_ported_zero_shot | 2.55 | 0.24 | 1.03 | 2.00 | 2881 | 1441 | 293 | 23.9 | 29.8 | 1.97 |


## Selection (rule of docs/t3_rag_bench.md)

| component | default | winner | reason |
|---|---|---|---|
| finfre_rag_native/finfre_pool | finfre_rag_native_pool_bal | **finfre_rag_native_pool_bal** | default kept |
| finfre_rag_ported/finfre_pool | finfre_rag_ported_pool_bal | **finfre_rag_ported_pool_bal** | default kept |
