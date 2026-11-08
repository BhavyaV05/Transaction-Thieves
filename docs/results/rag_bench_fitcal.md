# GPU RAG matrix report (fit+cal, 4000 bootstrap resamples)

## Group planner (n=298, threshold 0.322699)

| arm | brier | log_loss | auc | pr_auc | ece | cal_slope | precision | recall | f1 | escalation_rate | catch_top10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| planner_default (default) | 0.159 | 0.628 | 0.851 | 0.837 | 0.072 | 0.417 | 0.727 | 0.872 | 0.793 | 0.628 | 0.179 |
| planner_default_seed1 | 0.162 | 0.634 | 0.845 | 0.835 | 0.073 | 0.405 | 0.726 | 0.885 | 0.798 | 0.638 | 0.179 |
| ev_none | 0.159 | 0.626 | 0.851 | 0.837 | 0.083 | 0.420 | 0.723 | 0.885 | 0.795 | 0.641 | 0.179 |
| ev_amount_1hop | 0.159 | 0.629 | 0.848 | 0.836 | 0.063 | 0.415 | 0.723 | 0.885 | 0.795 | 0.641 | 0.179 |
| ev_amount | 0.160 | 0.632 | 0.847 | 0.836 | 0.085 | 0.410 | 0.734 | 0.885 | 0.802 | 0.631 | 0.179 |
| ev_scorer_v2 | 0.159 | 0.629 | 0.848 | 0.837 | 0.081 | 0.416 | 0.741 | 0.878 | 0.804 | 0.621 | 0.179 |
| ev_pcst_v1 | 0.157 | 0.624 | 0.855 | 0.840 | 0.090 | 0.424 | 0.727 | 0.872 | 0.793 | 0.628 | 0.179 |
| rules_none | 0.158 | 0.626 | 0.852 | 0.838 | 0.077 | 0.421 | 0.721 | 0.878 | 0.792 | 0.638 | 0.179 |
| rules_fixed | 0.160 | 0.632 | 0.848 | 0.837 | 0.057 | 0.409 | 0.727 | 0.872 | 0.793 | 0.628 | 0.179 |
| rules_whole | 0.160 | 0.631 | 0.847 | 0.836 | 0.062 | 0.411 | 0.726 | 0.885 | 0.798 | 0.638 | 0.179 |
| menu_topk_8 | 0.158 | 0.626 | 0.853 | 0.838 | 0.076 | 0.422 | 0.723 | 0.872 | 0.791 | 0.631 | 0.179 |
| menu_topk_16 | 0.159 | 0.630 | 0.847 | 0.836 | 0.068 | 0.414 | 0.739 | 0.891 | 0.808 | 0.631 | 0.179 |
| menu_topk_all | 0.157 | 0.624 | 0.856 | 0.840 | 0.077 | 0.425 | 0.730 | 0.885 | 0.800 | 0.634 | 0.179 |
| evidence_min_0 | 0.158 | 0.625 | 0.853 | 0.838 | 0.089 | 0.423 | 0.729 | 0.878 | 0.797 | 0.631 | 0.179 |
| evidence_min_10 | 0.159 | 0.628 | 0.850 | 0.836 | 0.087 | 0.418 | 0.727 | 0.872 | 0.793 | 0.628 | 0.179 |
| evidence_min_25 | 0.158 | 0.626 | 0.852 | 0.838 | 0.076 | 0.420 | 0.723 | 0.872 | 0.791 | 0.631 | 0.179 |
| lib_none | 0.159 | 0.628 | 0.850 | 0.837 | 0.084 | 0.417 | 0.729 | 0.878 | 0.797 | 0.631 | 0.179 |
| lib_account | 0.156 | 0.622 | 0.854 | 0.841 | 0.088 | 0.427 | 0.734 | 0.885 | 0.802 | 0.631 | 0.179 |
| lib_shape_bt | 0.160 | 0.631 | 0.845 | 0.835 | 0.060 | 0.412 | 0.723 | 0.872 | 0.791 | 0.631 | 0.179 |
| planner_combined | 0.159 | 0.627 | 0.851 | 0.838 | 0.084 | 0.418 | 0.727 | 0.872 | 0.793 | 0.628 | 0.179 |
| p0 | 0.158 | 0.626 | 0.853 | 0.838 | 0.069 | 0.421 | 0.725 | 0.878 | 0.794 | 0.634 | 0.179 |

Paired difference (arm - the default (planner_default)), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| planner_default_seed1 | +0.0028 [-0.0025, +0.0082] p_holm 1.000 | +0.0066 [-0.0053, +0.0183] p_holm 1.000 | -0.0060 [-0.0167, +0.0049] p_holm 1.000 | -0.0027 [-0.0098, +0.0042] p_holm 1.000 | +0.0011 [-0.0383, +0.0317] p_holm 1.000 | +0.0047 [-0.0146, +0.0249] p_holm 1.000 |
| ev_none | +0.0000 [-0.0047, +0.0045] p_holm 1.000 | -0.0015 [-0.0134, +0.0103] p_holm 1.000 | -0.0004 [-0.0119, +0.0110] p_holm 1.000 | +0.0003 [-0.0066, +0.0074] p_holm 1.000 | +0.0107 [-0.0362, +0.0371] p_holm 1.000 | +0.0024 [-0.0132, +0.0187] p_holm 1.000 |
| ev_amount_1hop | +0.0004 [-0.0021, +0.0031] p_holm 1.000 | +0.0013 [-0.0057, +0.0080] p_holm 1.000 | -0.0030 [-0.0106, +0.0052] p_holm 1.000 | -0.0010 [-0.0050, +0.0030] p_holm 1.000 | -0.0086 [-0.0306, +0.0302] p_holm 1.000 | +0.0024 [-0.0086, +0.0150] p_holm 1.000 |
| ev_amount | +0.0014 [-0.0026, +0.0057] p_holm 1.000 | +0.0042 [-0.0058, +0.0151] p_holm 1.000 | -0.0042 [-0.0141, +0.0052] p_holm 1.000 | -0.0016 [-0.0074, +0.0041] p_holm 1.000 | +0.0131 [-0.0343, +0.0354] p_holm 1.000 | +0.0093 [-0.0060, +0.0258] p_holm 1.000 |
| ev_scorer_v2 | -0.0005 [-0.0045, +0.0037] p_holm 1.000 | +0.0010 [-0.0097, +0.0121] p_holm 1.000 | -0.0029 [-0.0133, +0.0072] p_holm 1.000 | +0.0002 [-0.0055, +0.0059] p_holm 1.000 | +0.0086 [-0.0332, +0.0360] p_holm 1.000 | +0.0105 [-0.0071, +0.0301] p_holm 1.000 |
| ev_pcst_v1 | -0.0016 [-0.0038, +0.0004] p_holm 1.000 | -0.0038 [-0.0096, +0.0015] p_holm 1.000 | +0.0041 [-0.0018, +0.0110] p_holm 1.000 | +0.0025 [-0.0007, +0.0062] p_holm 1.000 | +0.0181 [-0.0246, +0.0343] p_holm 1.000 | +0.0000 [-0.0068, +0.0069] p_holm 1.000 |
| rules_none | -0.0012 [-0.0046, +0.0021] p_holm 1.000 | -0.0023 [-0.0112, +0.0066] p_holm 1.000 | +0.0009 [-0.0084, +0.0101] p_holm 1.000 | +0.0009 [-0.0040, +0.0061] p_holm 1.000 | +0.0050 [-0.0308, +0.0360] p_holm 1.000 | -0.0011 [-0.0148, +0.0130] p_holm 1.000 |
| rules_fixed | +0.0014 [-0.0020, +0.0050] p_holm 1.000 | +0.0043 [-0.0045, +0.0141] p_holm 1.000 | -0.0031 [-0.0127, +0.0055] p_holm 1.000 | -0.0006 [-0.0059, +0.0046] p_holm 1.000 | -0.0148 [-0.0369, +0.0236] p_holm 1.000 | +0.0000 [-0.0111, +0.0119] p_holm 1.000 |
| rules_whole | +0.0010 [-0.0028, +0.0050] p_holm 1.000 | +0.0030 [-0.0069, +0.0134] p_holm 1.000 | -0.0040 [-0.0149, +0.0062] p_holm 1.000 | -0.0017 [-0.0075, +0.0038] p_holm 1.000 | -0.0099 [-0.0368, +0.0297] p_holm 1.000 | +0.0047 [-0.0103, +0.0212] p_holm 1.000 |
| menu_topk_8 | -0.0010 [-0.0022, +0.0001] p_holm 1.000 | -0.0022 [-0.0053, +0.0003] p_holm 1.000 | +0.0016 [-0.0004, +0.0038] p_holm 1.000 | +0.0011 [-0.0004, +0.0028] p_holm 1.000 | +0.0038 [-0.0158, +0.0201] p_holm 1.000 | -0.0023 [-0.0072, +0.0000] p_holm 1.000 |
| menu_topk_16 | +0.0002 [-0.0039, +0.0045] p_holm 1.000 | +0.0017 [-0.0087, +0.0130] p_holm 1.000 | -0.0040 [-0.0148, +0.0055] p_holm 1.000 | -0.0011 [-0.0073, +0.0044] p_holm 1.000 | -0.0043 [-0.0378, +0.0304] p_holm 1.000 | +0.0151 [-0.0059, +0.0366] p_holm 1.000 |
| menu_topk_all | -0.0021 [-0.0065, +0.0025] p_holm 1.000 | -0.0041 [-0.0155, +0.0075] p_holm 1.000 | +0.0044 [-0.0076, +0.0168] p_holm 1.000 | +0.0033 [-0.0032, +0.0101] p_holm 1.000 | +0.0053 [-0.0295, +0.0348] p_holm 1.000 | +0.0070 [-0.0088, +0.0242] p_holm 1.000 |
| evidence_min_0 | -0.0010 [-0.0026, +0.0003] p_holm 1.000 | -0.0030 [-0.0071, +0.0004] p_holm 1.000 | +0.0021 [-0.0009, +0.0054] p_holm 1.000 | +0.0009 [-0.0010, +0.0029] p_holm 1.000 | +0.0167 [-0.0145, +0.0327] p_holm 1.000 | +0.0035 [+0.0000, +0.0114] p_holm 1.000 |
| evidence_min_10 | -0.0001 [-0.0018, +0.0016] p_holm 1.000 | -0.0002 [-0.0051, +0.0053] p_holm 1.000 | -0.0017 [-0.0086, +0.0032] p_holm 1.000 | -0.0007 [-0.0036, +0.0016] p_holm 1.000 | +0.0151 [-0.0155, +0.0322] p_holm 1.000 | +0.0000 [+0.0000, +0.0000] p_holm 1.000 |
| evidence_min_25 | -0.0006 [-0.0022, +0.0008] p_holm 1.000 | -0.0016 [-0.0058, +0.0019] p_holm 1.000 | +0.0003 [-0.0027, +0.0036] p_holm 1.000 | +0.0003 [-0.0013, +0.0021] p_holm 1.000 | +0.0040 [-0.0162, +0.0288] p_holm 1.000 | -0.0023 [-0.0073, +0.0000] p_holm 1.000 |
| lib_none | -0.0004 [-0.0038, +0.0030] p_holm 1.000 | +0.0000 [-0.0079, +0.0088] p_holm 1.000 | -0.0007 [-0.0097, +0.0071] p_holm 1.000 | +0.0003 [-0.0046, +0.0052] p_holm 1.000 | +0.0119 [-0.0298, +0.0349] p_holm 1.000 | +0.0035 [-0.0047, +0.0139] p_holm 1.000 |
| lib_account | -0.0031 [-0.0074, +0.0011] p_holm 1.000 | -0.0054 [-0.0158, +0.0053] p_holm 1.000 | +0.0029 [-0.0071, +0.0123] p_holm 1.000 | +0.0043 [-0.0018, +0.0110] p_holm 1.000 | +0.0161 [-0.0341, +0.0385] p_holm 1.000 | +0.0093 [-0.0069, +0.0275] p_holm 1.000 |
| lib_shape_bt | +0.0008 [-0.0021, +0.0038] p_holm 1.000 | +0.0031 [-0.0047, +0.0116] p_holm 1.000 | -0.0058 [-0.0140, +0.0009] p_holm 1.000 | -0.0022 [-0.0064, +0.0017] p_holm 1.000 | -0.0118 [-0.0339, +0.0237] p_holm 1.000 | -0.0023 [-0.0072, +0.0000] p_holm 1.000 |
| planner_combined | -0.0003 [-0.0015, +0.0009] p_holm 1.000 | -0.0007 [-0.0031, +0.0019] p_holm 1.000 | +0.0002 [-0.0021, +0.0023] p_holm 1.000 | +0.0005 [-0.0012, +0.0022] p_holm 1.000 | +0.0116 [-0.0167, +0.0204] p_holm 1.000 | +0.0000 [+0.0000, +0.0000] p_holm 1.000 |

Paired difference (arm - p0), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| planner_default | +0.0009 [-0.0034, +0.0051] p_holm 1.000 | +0.0019 [-0.0084, +0.0119] p_holm 1.000 | -0.0022 [-0.0127, +0.0080] p_holm 1.000 | -0.0013 [-0.0076, +0.0050] p_holm 1.000 | +0.0027 [-0.0330, +0.0311] p_holm 1.000 | -0.0012 [-0.0171, +0.0140] p_holm 1.000 |
| planner_default_seed1 | +0.0037 [-0.0002, +0.0077] p_holm 1.000 | +0.0085 [-0.0008, +0.0177] p_holm 1.000 | -0.0082 [-0.0177, +0.0013] p_holm 1.000 | -0.0039 [-0.0101, +0.0020] p_holm 1.000 | +0.0038 [-0.0378, +0.0304] p_holm 1.000 | +0.0035 [-0.0099, +0.0175] p_holm 1.000 |
| ev_none | +0.0009 [-0.0039, +0.0059] p_holm 1.000 | +0.0004 [-0.0109, +0.0119] p_holm 1.000 | -0.0026 [-0.0125, +0.0066] p_holm 1.000 | -0.0010 [-0.0081, +0.0057] p_holm 1.000 | +0.0134 [-0.0361, +0.0355] p_holm 1.000 | +0.0012 [-0.0172, +0.0199] p_holm 1.000 |
| ev_amount_1hop | +0.0014 [-0.0026, +0.0053] p_holm 1.000 | +0.0032 [-0.0058, +0.0121] p_holm 1.000 | -0.0051 [-0.0141, +0.0038] p_holm 1.000 | -0.0022 [-0.0082, +0.0035] p_holm 1.000 | -0.0059 [-0.0345, +0.0319] p_holm 1.000 | +0.0012 [-0.0125, +0.0156] p_holm 1.000 |
| ev_amount | +0.0023 [-0.0017, +0.0066] p_holm 1.000 | +0.0061 [-0.0036, +0.0165] p_holm 1.000 | -0.0063 [-0.0160, +0.0028] p_holm 1.000 | -0.0028 [-0.0086, +0.0027] p_holm 1.000 | +0.0158 [-0.0325, +0.0335] p_holm 1.000 | +0.0081 [-0.0104, +0.0271] p_holm 1.000 |
| ev_scorer_v2 | +0.0005 [-0.0045, +0.0052] p_holm 1.000 | +0.0028 [-0.0088, +0.0144] p_holm 1.000 | -0.0051 [-0.0155, +0.0050] p_holm 1.000 | -0.0011 [-0.0077, +0.0056] p_holm 1.000 | +0.0113 [-0.0347, +0.0357] p_holm 1.000 | +0.0093 [-0.0106, +0.0300] p_holm 1.000 |
| ev_pcst_v1 | -0.0007 [-0.0052, +0.0036] p_holm 1.000 | -0.0019 [-0.0123, +0.0080] p_holm 1.000 | +0.0019 [-0.0069, +0.0116] p_holm 1.000 | +0.0012 [-0.0045, +0.0075] p_holm 1.000 | +0.0209 [-0.0307, +0.0367] p_holm 1.000 | -0.0012 [-0.0160, +0.0127] p_holm 1.000 |
| rules_none | -0.0003 [-0.0040, +0.0035] p_holm 1.000 | -0.0004 [-0.0093, +0.0087] p_holm 1.000 | -0.0012 [-0.0092, +0.0064] p_holm 1.000 | -0.0003 [-0.0058, +0.0050] p_holm 1.000 | +0.0077 [-0.0319, +0.0353] p_holm 1.000 | -0.0023 [-0.0171, +0.0117] p_holm 1.000 |
| rules_fixed | +0.0023 [-0.0018, +0.0061] p_holm 1.000 | +0.0062 [-0.0038, +0.0157] p_holm 1.000 | -0.0053 [-0.0131, +0.0025] p_holm 1.000 | -0.0019 [-0.0071, +0.0035] p_holm 1.000 | -0.0121 [-0.0384, +0.0250] p_holm 1.000 | -0.0012 [-0.0162, +0.0136] p_holm 1.000 |
| rules_whole | +0.0019 [-0.0019, +0.0057] p_holm 1.000 | +0.0049 [-0.0037, +0.0136] p_holm 1.000 | -0.0062 [-0.0147, +0.0018] p_holm 1.000 | -0.0029 [-0.0088, +0.0024] p_holm 1.000 | -0.0071 [-0.0359, +0.0266] p_holm 1.000 | +0.0035 [-0.0071, +0.0158] p_holm 1.000 |
| menu_topk_8 | -0.0000 [-0.0043, +0.0042] p_holm 1.000 | -0.0004 [-0.0106, +0.0098] p_holm 1.000 | -0.0006 [-0.0109, +0.0093] p_holm 1.000 | -0.0002 [-0.0063, +0.0059] p_holm 1.000 | +0.0065 [-0.0318, +0.0346] p_holm 1.000 | -0.0035 [-0.0189, +0.0110] p_holm 1.000 |
| menu_topk_16 | +0.0011 [-0.0026, +0.0050] p_holm 1.000 | +0.0036 [-0.0052, +0.0129] p_holm 1.000 | -0.0062 [-0.0149, +0.0019] p_holm 1.000 | -0.0024 [-0.0080, +0.0027] p_holm 1.000 | -0.0016 [-0.0376, +0.0291] p_holm 1.000 | +0.0139 [-0.0039, +0.0329] p_holm 1.000 |
| menu_topk_all | -0.0012 [-0.0029, +0.0003] p_holm 1.000 | -0.0023 [-0.0068, +0.0022] p_holm 1.000 | +0.0023 [-0.0026, +0.0079] p_holm 1.000 | +0.0020 [-0.0004, +0.0048] p_holm 1.000 | +0.0080 [-0.0245, +0.0270] p_holm 1.000 | +0.0058 [+0.0000, +0.0151] p_holm 1.000 |
| evidence_min_0 | -0.0001 [-0.0043, +0.0040] p_holm 1.000 | -0.0011 [-0.0108, +0.0086] p_holm 1.000 | -0.0001 [-0.0100, +0.0096] p_holm 1.000 | -0.0004 [-0.0066, +0.0055] p_holm 1.000 | +0.0194 [-0.0281, +0.0410] p_holm 1.000 | +0.0023 [-0.0122, +0.0166] p_holm 1.000 |
| evidence_min_10 | +0.0008 [-0.0033, +0.0049] p_holm 1.000 | +0.0017 [-0.0079, +0.0118] p_holm 1.000 | -0.0039 [-0.0154, +0.0070] p_holm 1.000 | -0.0020 [-0.0086, +0.0044] p_holm 1.000 | +0.0178 [-0.0305, +0.0412] p_holm 1.000 | -0.0012 [-0.0171, +0.0140] p_holm 1.000 |
| evidence_min_25 | +0.0003 [-0.0038, +0.0044] p_holm 1.000 | +0.0003 [-0.0094, +0.0102] p_holm 1.000 | -0.0019 [-0.0119, +0.0081] p_holm 1.000 | -0.0009 [-0.0070, +0.0052] p_holm 1.000 | +0.0067 [-0.0314, +0.0387] p_holm 1.000 | -0.0035 [-0.0188, +0.0112] p_holm 1.000 |
| lib_none | +0.0005 [-0.0029, +0.0039] p_holm 1.000 | +0.0019 [-0.0063, +0.0101] p_holm 1.000 | -0.0029 [-0.0119, +0.0056] p_holm 1.000 | -0.0010 [-0.0066, +0.0044] p_holm 1.000 | +0.0147 [-0.0316, +0.0333] p_holm 1.000 | +0.0023 [-0.0133, +0.0178] p_holm 1.000 |
| lib_account | -0.0022 [-0.0062, +0.0016] p_holm 1.000 | -0.0036 [-0.0130, +0.0058] p_holm 1.000 | +0.0007 [-0.0085, +0.0098] p_holm 1.000 | +0.0030 [-0.0032, +0.0096] p_holm 1.000 | +0.0188 [-0.0333, +0.0361] p_holm 1.000 | +0.0081 [-0.0059, +0.0231] p_holm 1.000 |
| lib_shape_bt | +0.0017 [-0.0026, +0.0061] p_holm 1.000 | +0.0050 [-0.0053, +0.0154] p_holm 1.000 | -0.0080 [-0.0186, +0.0014] p_holm 1.000 | -0.0034 [-0.0100, +0.0027] p_holm 1.000 | -0.0090 [-0.0377, +0.0271] p_holm 1.000 | -0.0035 [-0.0198, +0.0125] p_holm 1.000 |
| planner_combined | +0.0006 [-0.0035, +0.0047] p_holm 1.000 | +0.0012 [-0.0087, +0.0111] p_holm 1.000 | -0.0020 [-0.0124, +0.0078] p_holm 1.000 | -0.0008 [-0.0069, +0.0052] p_holm 1.000 | +0.0143 [-0.0323, +0.0348] p_holm 1.000 | -0.0012 [-0.0171, +0.0140] p_holm 1.000 |

Flipped calls vs p0 at the threshold (right-to-wrong / wrong-to-right), no-verdict rate, collapse checks:

| arm | flips | no verdict | var(p)/var(p0) | AUC | AUC p0 | collapsed |
|---|---|---|---|---|---|---|
| planner_default | 4 / 4 | 0.000 | 0.991 | 0.851 | 0.853 | True |
| planner_default_seed1 | 2 / 3 | 0.000 | 0.981 | 0.845 | 0.853 | True |
| ev_none | 5 / 5 | 0.000 | 0.985 | 0.851 | 0.853 | True |
| ev_amount_1hop | 3 / 3 | 0.000 | 0.990 | 0.848 | 0.853 | True |
| ev_amount | 4 / 7 | 0.000 | 1.008 | 0.847 | 0.853 | True |
| ev_scorer_v2 | 4 / 8 | 0.000 | 0.998 | 0.848 | 0.853 | True |
| ev_pcst_v1 | 3 / 3 | 0.000 | 0.989 | 0.855 | 0.853 | False |
| rules_none | 4 / 3 | 0.000 | 0.980 | 0.852 | 0.853 | True |
| rules_fixed | 3 / 3 | 0.000 | 0.989 | 0.848 | 0.853 | True |
| rules_whole | 2 / 3 | 0.000 | 0.989 | 0.847 | 0.853 | True |
| menu_topk_8 | 4 / 3 | 0.000 | 0.992 | 0.853 | 0.853 | True |
| menu_topk_16 | 3 / 8 | 0.000 | 0.996 | 0.847 | 0.853 | True |
| menu_topk_all | 0 / 2 | 0.000 | 1.002 | 0.856 | 0.853 | False |
| evidence_min_0 | 3 / 4 | 0.000 | 0.991 | 0.853 | 0.853 | True |
| evidence_min_10 | 4 / 4 | 0.000 | 0.993 | 0.850 | 0.853 | True |
| evidence_min_25 | 4 / 3 | 0.000 | 0.989 | 0.852 | 0.853 | True |
| lib_none | 4 / 5 | 0.000 | 1.002 | 0.850 | 0.853 | True |
| lib_account | 3 / 6 | 0.000 | 1.004 | 0.854 | 0.853 | False |
| lib_shape_bt | 5 / 4 | 0.000 | 0.991 | 0.845 | 0.853 | True |
| planner_combined | 4 / 4 | 0.000 | 0.993 | 0.851 | 0.853 | True |

Per alert (means): questions, accepted-question share, eps, LLM calls, prompt tokens in / per call, tokens out, median / p90 s, GPU h.

| arm | questions | accepted | eps | calls | tok in | tok/call | tok out | median s | p90 s | GPU h |
|---|---|---|---|---|---|---|---|---|---|---|
| planner_default | 1.43 | 0.63 | 1.32 | 1.94 | 10281 | 5292 | 257 | 83.5 | 153.9 | 6.51 |
| planner_default_seed1 | 1.38 | 0.63 | 1.28 | 2.00 | 10491 | 5254 | 266 | 82.2 | 179.0 | 6.66 |
| ev_none | 1.25 | 0.89 | 1.44 | 1.36 | 5391 | 3967 | 170 | 32.4 | 66.6 | 3.17 |
| ev_amount_1hop | 1.37 | 0.64 | 1.28 | 1.99 | 10172 | 5103 | 262 | 80.8 | 159.8 | 6.46 |
| ev_amount | 1.47 | 0.61 | 1.27 | 2.11 | 13957 | 6623 | 289 | 116.9 | 237.3 | 9.61 |
| ev_scorer_v2 | 1.43 | 0.81 | 1.32 | 1.70 | 11097 | 6510 | 227 | 92.7 | 178.8 | 7.59 |
| ev_pcst_v1 | 1.40 | 0.63 | 1.29 | 1.94 | 10272 | 5287 | 257 | 87.1 | 173.0 | 6.83 |
| rules_none | 1.34 | 0.56 | 1.31 | 1.79 | 8970 | 5006 | 231 | 74.6 | 145.2 | 5.81 |
| rules_fixed | 1.36 | 0.64 | 1.27 | 2.04 | 10835 | 5311 | 274 | 84.9 | 181.0 | 7.04 |
| rules_whole | 1.31 | 0.56 | 1.24 | 2.15 | 12543 | 5822 | 288 | 97.3 | 217.1 | 8.27 |
| menu_topk_8 | 1.41 | 0.62 | 1.30 | 1.94 | 10264 | 5301 | 256 | 83.1 | 158.3 | 6.56 |
| menu_topk_16 | 1.48 | 0.64 | 1.21 | 2.50 | 14366 | 5746 | 331 | 99.3 | 253.6 | 9.68 |
| menu_topk_all | 0.58 | 0.43 | 0.43 | 1.27 | 7913 | 6222 | 174 | 0.4 | 248.4 | 5.37 |
| evidence_min_0 | 1.43 | 0.63 | 1.31 | 1.97 | 10411 | 5294 | 260 | 86.0 | 175.6 | 6.79 |
| evidence_min_10 | 1.41 | 0.63 | 1.30 | 1.94 | 10255 | 5296 | 256 | 84.5 | 170.9 | 6.71 |
| evidence_min_25 | 1.42 | 0.63 | 1.30 | 1.93 | 10226 | 5291 | 256 | 84.3 | 160.5 | 6.67 |
| lib_none | 1.36 | 0.58 | 1.34 | 1.78 | 9388 | 5278 | 237 | 80.6 | 146.5 | 6.13 |
| lib_account | 1.34 | 0.65 | 1.33 | 1.85 | 9885 | 5356 | 246 | 85.7 | 156.8 | 6.55 |
| lib_shape_bt | 1.40 | 0.64 | 1.29 | 1.94 | 10187 | 5261 | 255 | 85.5 | 164.6 | 6.72 |
| planner_combined | 1.42 | 0.62 | 1.31 | 1.91 | 10106 | 5293 | 253 | 83.6 | 152.2 | 6.48 |


## Group sharma_native (n=298, threshold 0.322699)

| arm | brier | log_loss | auc | pr_auc | ece | cal_slope | precision | recall | f1 | escalation_rate | catch_top10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sharma_native_mem_on (default) | 0.230 | 0.710 | 0.729 | 0.700 | 0.183 | 0.487 | 0.808 | 0.622 | 0.703 | 0.403 | 0.173 |
| sharma_native_mem_off | 0.214 | 0.666 | 0.752 | 0.731 | 0.170 | 0.587 | 0.864 | 0.609 | 0.714 | 0.369 | 0.179 |
| p0 | 0.158 | 0.626 | 0.853 | 0.838 | 0.069 | 0.421 | 0.725 | 0.878 | 0.794 | 0.634 | 0.179 |

Paired difference (arm - the default (sharma_native_mem_on)), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| sharma_native_mem_off | -0.0162 [-0.0430, +0.0106] p_holm 0.209 | -0.0445 [-0.1183, +0.0289] p_holm 0.208 | +0.0223 [-0.0100, +0.0562] p_holm 0.174 | +0.0310 [-0.0045, +0.0685] p_holm 0.084 | -0.0137 [-0.0475, +0.0195] p_holm 0.462 | +0.0114 [-0.0275, +0.0513] p_holm 0.562 |

Paired difference (arm - p0), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| sharma_native_mem_on | +0.0722 [+0.0428, +0.1035] p_holm 0.001 | +0.0843 [-0.1002, +0.2422] p_holm 0.702 | -0.1240 [-0.1666, -0.0833] p_holm 0.001 | -0.1388 [-0.1882, -0.0910] p_holm 0.001 | +0.1142 [+0.0306, +0.1205] p_holm 0.003 | -0.0913 [-0.1554, -0.0323] p_holm 0.006 |
| sharma_native_mem_off | +0.0560 [+0.0310, +0.0822] p_holm 0.001 | +0.0398 [-0.1445, +0.1912] p_holm 0.702 | -0.1017 [-0.1334, -0.0695] p_holm 0.001 | -0.1078 [-0.1462, -0.0682] p_holm 0.001 | +0.1005 [+0.0180, +0.1056] p_holm 0.012 | -0.0799 [-0.1420, -0.0212] p_holm 0.007 |

Flipped calls vs p0 at the threshold (right-to-wrong / wrong-to-right), no-verdict rate, collapse checks:

| arm | flips | no verdict | var(p)/var(p0) | AUC | AUC p0 | collapsed |
|---|---|---|---|---|---|---|
| sharma_native_mem_on | 49 / 38 | 0.000 | 1.457 | 0.729 | 0.853 | True |
| sharma_native_mem_off | 42 / 37 | 0.000 | 1.412 | 0.752 | 0.853 | True |

Per alert (means): questions, accepted-question share, eps, LLM calls, prompt tokens in / per call, tokens out, median / p90 s, GPU h.

| arm | questions | accepted | eps | calls | tok in | tok/call | tok out | median s | p90 s | GPU h |
|---|---|---|---|---|---|---|---|---|---|---|
| sharma_native_mem_on | 0.00 | - | 0.00 | 5.00 | 3692 | 738 | 352 | 22.7 | 31.9 | 2.00 |
| sharma_native_mem_off | 0.00 | - | 0.00 | 5.00 | 3591 | 718 | 335 | 25.9 | 31.4 | 2.08 |


## Group sharma_ported (n=298, threshold 0.322699)

| arm | brier | log_loss | auc | pr_auc | ece | cal_slope | precision | recall | f1 | escalation_rate | catch_top10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sharma_ported_mem_on (default) | 0.255 | 0.779 | 0.701 | 0.676 | 0.206 | 0.445 | 0.804 | 0.551 | 0.654 | 0.359 | 0.154 |
| sharma_ported_mem_off | 0.238 | 0.734 | 0.722 | 0.701 | 0.197 | 0.521 | 0.840 | 0.571 | 0.679 | 0.356 | 0.167 |
| p0 | 0.158 | 0.626 | 0.853 | 0.838 | 0.069 | 0.421 | 0.725 | 0.878 | 0.794 | 0.634 | 0.179 |

Paired difference (arm - the default (sharma_ported_mem_on)), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| sharma_ported_mem_off | -0.0166 [-0.0373, +0.0033] p_holm 0.105 | -0.0442 [-0.1010, +0.0113] p_holm 0.115 | +0.0214 [-0.0043, +0.0479] p_holm 0.104 | +0.0246 [-0.0051, +0.0549] p_holm 0.100 | -0.0095 [-0.0414, +0.0135] p_holm 0.372 | +0.0254 [-0.0027, +0.0565] p_holm 0.083 |

Paired difference (arm - p0), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| sharma_ported_mem_on | +0.0970 [+0.0657, +0.1301] p_holm 0.001 | +0.1526 [-0.0400, +0.3183] p_holm 0.254 | -0.1526 [-0.1933, -0.1115] p_holm 0.001 | -0.1625 [-0.2134, -0.1097] p_holm 0.001 | +0.1370 [+0.0559, +0.1557] p_holm 0.001 | -0.1402 [-0.2072, -0.0768] p_holm 0.001 |
| sharma_ported_mem_off | +0.0804 [+0.0514, +0.1102] p_holm 0.001 | +0.1083 [-0.0794, +0.2665] p_holm 0.254 | -0.1312 [-0.1683, -0.0934] p_holm 0.001 | -0.1379 [-0.1830, -0.0925] p_holm 0.001 | +0.1276 [+0.0457, +0.1394] p_holm 0.001 | -0.1148 [-0.1786, -0.0514] p_holm 0.001 |

Flipped calls vs p0 at the threshold (right-to-wrong / wrong-to-right), no-verdict rate, collapse checks:

| arm | flips | no verdict | var(p)/var(p0) | AUC | AUC p0 | collapsed |
|---|---|---|---|---|---|---|
| sharma_ported_mem_on | 55 / 35 | 0.000 | 1.384 | 0.701 | 0.853 | True |
| sharma_ported_mem_off | 48 / 35 | 0.007 | 1.325 | 0.722 | 0.853 | True |

Per alert (means): questions, accepted-question share, eps, LLM calls, prompt tokens in / per call, tokens out, median / p90 s, GPU h.

| arm | questions | accepted | eps | calls | tok in | tok/call | tok out | median s | p90 s | GPU h |
|---|---|---|---|---|---|---|---|---|---|---|
| sharma_ported_mem_on | 0.88 | 0.54 | 0.48 | 5.79 | 17048 | 2943 | 403 | 29.3 | 39.1 | 2.44 |
| sharma_ported_mem_off | 1.30 | 0.51 | 0.72 | 6.20 | 18208 | 2936 | 421 | 35.1 | 46.3 | 2.87 |


## Group torres_native (n=298, threshold 0.322699)

| arm | brier | log_loss | auc | pr_auc | ece | cal_slope | precision | recall | f1 | escalation_rate | catch_top10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torres_native_mem_on (default) | 0.316 | 0.952 | 0.606 | 0.598 | 0.268 | 379097534.117 | 0.544 | 0.994 | 0.703 | 0.956 | 0.135 |
| torres_native_mem_off | 0.261 | 0.762 | 0.639 | 0.628 | 0.185 | 0.363 | 0.568 | 0.968 | 0.716 | 0.893 | 0.160 |
| p0 | 0.158 | 0.626 | 0.853 | 0.838 | 0.069 | 0.421 | 0.725 | 0.878 | 0.794 | 0.634 | 0.179 |

Paired difference (arm - the default (torres_native_mem_on)), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| torres_native_mem_off | -0.0548 [-0.0901, -0.0206] p_holm 0.003 | -0.1891 [-0.3173, -0.0738] p_holm 0.002 | +0.0331 [-0.0296, +0.0959] p_holm 0.294 | +0.0295 [-0.0208, +0.0798] p_holm 0.248 | -0.0838 [-0.1561, -0.0469] p_holm 0.000 | +0.0127 [-0.0098, +0.0352] p_holm 0.254 |

Paired difference (arm - p0), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| torres_native_mem_on | +0.1575 [+0.1142, +0.2012] p_holm 0.001 | +0.3256 [+0.1368, +0.4875] p_holm 0.005 | -0.2470 [-0.3050, -0.1889] p_holm 0.001 | -0.2401 [-0.2986, -0.1802] p_holm 0.001 | +0.1990 [+0.1001, +0.2335] p_holm 0.001 | -0.0913 [-0.1325, -0.0495] p_holm 0.001 |
| torres_native_mem_off | +0.1027 [+0.0680, +0.1369] p_holm 0.001 | +0.1365 [-0.0605, +0.2988] p_holm 0.159 | -0.2140 [-0.2738, -0.1540] p_holm 0.001 | -0.2107 [-0.2764, -0.1448] p_holm 0.001 | +0.1153 [+0.0098, +0.1215] p_holm 0.026 | -0.0786 [-0.1202, -0.0364] p_holm 0.001 |

Flipped calls vs p0 at the threshold (right-to-wrong / wrong-to-right), no-verdict rate, collapse checks:

| arm | flips | no verdict | var(p)/var(p0) | AUC | AUC p0 | collapsed |
|---|---|---|---|---|---|---|
| torres_native_mem_on | 78 / 18 | 0.111 | 0.401 | 0.606 | 0.853 | True |
| torres_native_mem_off | 70 / 21 | 0.064 | 0.683 | 0.639 | 0.853 | True |

Per alert (means): questions, accepted-question share, eps, LLM calls, prompt tokens in / per call, tokens out, median / p90 s, GPU h.

| arm | questions | accepted | eps | calls | tok in | tok/call | tok out | median s | p90 s | GPU h |
|---|---|---|---|---|---|---|---|---|---|---|
| torres_native_mem_on | 0.00 | - | 0.00 | 2.92 | 3917 | 1340 | 1232 | 57.3 | 86.8 | 4.92 |
| torres_native_mem_off | 0.00 | - | 0.00 | 3.14 | 3930 | 1253 | 1284 | 57.2 | 88.9 | 4.85 |


## Group torres_ported (n=298, threshold 0.322699)

| arm | brier | log_loss | auc | pr_auc | ece | cal_slope | precision | recall | f1 | escalation_rate | catch_top10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torres_ported_mem_on (default) | 0.299 | 0.873 | 0.602 | 0.586 | 0.267 | 0.460 | 0.551 | 1.000 | 0.711 | 0.950 | 0.109 |
| torres_ported_mem_off | 0.254 | 0.769 | 0.639 | 0.627 | 0.159 | 0.318 | 0.549 | 0.974 | 0.702 | 0.930 | 0.167 |
| p0 | 0.158 | 0.626 | 0.853 | 0.838 | 0.069 | 0.421 | 0.725 | 0.878 | 0.794 | 0.634 | 0.179 |

Paired difference (arm - the default (torres_ported_mem_on)), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| torres_ported_mem_off | -0.0455 [-0.0807, -0.0099] p_holm 0.013 | -0.1044 [-0.2191, +0.0293] p_holm 0.110 | +0.0372 [-0.0277, +0.0991] p_holm 0.245 | +0.0414 [-0.0069, +0.0923] p_holm 0.088 | -0.1072 [-0.1476, -0.0478] p_holm 0.001 | -0.0086 [-0.0286, +0.0104] p_holm 0.382 |

Paired difference (arm - p0), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| torres_ported_mem_on | +0.1410 [+0.0984, +0.1832] p_holm 0.001 | +0.2473 [+0.0433, +0.4220] p_holm 0.040 | -0.2516 [-0.3138, -0.1917] p_holm 0.001 | -0.2526 [-0.3164, -0.1878] p_holm 0.001 | +0.1972 [+0.0784, +0.2099] p_holm 0.001 | -0.0835 [-0.1249, -0.0420] p_holm 0.001 |
| torres_ported_mem_off | +0.0955 [+0.0638, +0.1273] p_holm 0.001 | +0.1430 [-0.0275, +0.2828] p_holm 0.091 | -0.2145 [-0.2719, -0.1559] p_holm 0.001 | -0.2111 [-0.2712, -0.1460] p_holm 0.001 | +0.0900 [-0.0104, +0.1007] p_holm 0.118 | -0.0921 [-0.1335, -0.0521] p_holm 0.001 |

Flipped calls vs p0 at the threshold (right-to-wrong / wrong-to-right), no-verdict rate, collapse checks:

| arm | flips | no verdict | var(p)/var(p0) | AUC | AUC p0 | collapsed |
|---|---|---|---|---|---|---|
| torres_ported_mem_on | 75 / 19 | 0.117 | 0.449 | 0.602 | 0.853 | True |
| torres_ported_mem_off | 75 / 17 | 0.067 | 0.545 | 0.639 | 0.853 | True |

Per alert (means): questions, accepted-question share, eps, LLM calls, prompt tokens in / per call, tokens out, median / p90 s, GPU h.

| arm | questions | accepted | eps | calls | tok in | tok/call | tok out | median s | p90 s | GPU h |
|---|---|---|---|---|---|---|---|---|---|---|
| torres_ported_mem_on | 2.40 | 0.34 | 0.87 | 3.95 | 8118 | 2054 | 1445 | 88.2 | 129.9 | 7.58 |
| torres_ported_mem_off | 2.20 | 0.26 | 1.07 | 3.99 | 7600 | 1906 | 1434 | 86.9 | 125.0 | 7.25 |


## Group co_investigator_native (n=298, threshold 0.322699)

| arm | brier | log_loss | auc | pr_auc | ece | cal_slope | precision | recall | f1 | escalation_rate | catch_top10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| co_investigator_native_mem_on (default) | 0.183 | 0.693 | 0.804 | 0.774 | 0.094 | 0.368 | 0.673 | 0.910 | 0.774 | 0.708 | 0.167 |
| co_investigator_native_mem_off | 0.194 | 0.863 | 0.786 | 0.772 | 0.112 | 732006014.708 | 0.667 | 0.885 | 0.760 | 0.695 | 0.167 |
| p0 | 0.158 | 0.626 | 0.853 | 0.838 | 0.069 | 0.421 | 0.725 | 0.878 | 0.794 | 0.634 | 0.179 |

Paired difference (arm - the default (co_investigator_native_mem_on)), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| co_investigator_native_mem_off | +0.0102 [-0.0190, +0.0384] p_holm 0.512 | +0.1702 [+0.0018, +0.3782] p_holm 0.047 | -0.0177 [-0.0676, +0.0319] p_holm 0.501 | -0.0027 [-0.0469, +0.0407] p_holm 0.921 | +0.0181 [-0.0327, +0.0635] p_holm 0.545 | -0.0135 [-0.0475, +0.0206] p_holm 0.450 |

Paired difference (arm - p0), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| co_investigator_native_mem_on | +0.0254 [+0.0035, +0.0480] p_holm 0.023 | +0.0666 [+0.0125, +0.1215] p_holm 0.016 | -0.0495 [-0.0865, -0.0144] p_holm 0.007 | -0.0642 [-0.1025, -0.0278] p_holm 0.001 | +0.0249 [-0.0328, +0.0560] p_holm 0.608 | -0.0204 [-0.0478, +0.0073] p_holm 0.147 |
| co_investigator_native_mem_off | +0.0356 [+0.0095, +0.0623] p_holm 0.011 | +0.2368 [+0.0713, +0.4423] p_holm 0.003 | -0.0672 [-0.1126, -0.0222] p_holm 0.005 | -0.0669 [-0.1077, -0.0282] p_holm 0.001 | +0.0430 [-0.0236, +0.0770] p_holm 0.596 | -0.0339 [-0.0686, +0.0010] p_holm 0.115 |

Flipped calls vs p0 at the threshold (right-to-wrong / wrong-to-right), no-verdict rate, collapse checks:

| arm | flips | no verdict | var(p)/var(p0) | AUC | AUC p0 | collapsed |
|---|---|---|---|---|---|---|
| co_investigator_native_mem_on | 23 / 11 | 0.661 | 0.914 | 0.804 | 0.853 | True |
| co_investigator_native_mem_off | 31 / 15 | 0.597 | 1.005 | 0.786 | 0.853 | True |

Per alert (means): questions, accepted-question share, eps, LLM calls, prompt tokens in / per call, tokens out, median / p90 s, GPU h.

| arm | questions | accepted | eps | calls | tok in | tok/call | tok out | median s | p90 s | GPU h |
|---|---|---|---|---|---|---|---|---|---|---|
| co_investigator_native_mem_on | 0.00 | - | 0.00 | 6.51 | 10498 | 1612 | 1698 | 86.5 | 106.7 | 7.00 |
| co_investigator_native_mem_off | 0.00 | - | 0.00 | 6.68 | 9908 | 1483 | 1706 | 90.7 | 125.6 | 7.74 |


## Group co_investigator_ported (n=298, threshold 0.322699)

| arm | brier | log_loss | auc | pr_auc | ece | cal_slope | precision | recall | f1 | escalation_rate | catch_top10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| co_investigator_ported_mem_on (default) | 0.175 | 0.631 | 0.822 | 0.796 | 0.104 | 0.500 | 0.661 | 0.923 | 0.770 | 0.732 | 0.173 |
| co_investigator_ported_mem_off | 0.186 | 0.693 | 0.803 | 0.802 | 0.097 | 0.334 | 0.654 | 0.910 | 0.761 | 0.728 | 0.179 |
| p0 | 0.158 | 0.626 | 0.853 | 0.838 | 0.069 | 0.421 | 0.725 | 0.878 | 0.794 | 0.634 | 0.179 |

Paired difference (arm - the default (co_investigator_ported_mem_on)), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| co_investigator_ported_mem_off | +0.0112 [-0.0168, +0.0384] p_holm 0.445 | +0.0628 [-0.0887, +0.2318] p_holm 0.440 | -0.0192 [-0.0646, +0.0292] p_holm 0.445 | +0.0062 [-0.0378, +0.0517] p_holm 0.806 | -0.0074 [-0.0396, +0.0533] p_holm 0.774 | -0.0087 [-0.0392, +0.0226] p_holm 0.602 |

Paired difference (arm - p0), 95% cluster-bootstrap CI, Holm-adjusted p within this group:

| arm | brier | log_loss | auc | pr_auc | ece | f1 |
|---|---|---|---|---|---|---|
| co_investigator_ported_mem_on | +0.0167 [-0.0090, +0.0428] p_holm 0.198 | +0.0046 [-0.1064, +0.0937] p_holm 0.916 | -0.0313 [-0.0730, +0.0100] p_holm 0.148 | -0.0425 [-0.0848, +0.0002] p_holm 0.104 | +0.0349 [-0.0401, +0.0582] p_holm 1.000 | -0.0241 [-0.0545, +0.0065] p_holm 0.133 |
| co_investigator_ported_mem_off | +0.0279 [+0.0025, +0.0529] p_holm 0.064 | +0.0675 [-0.1118, +0.2515] p_holm 0.916 | -0.0505 [-0.0921, -0.0083] p_holm 0.044 | -0.0363 [-0.0766, +0.0051] p_holm 0.104 | +0.0274 [-0.0337, +0.0663] p_holm 1.000 | -0.0328 [-0.0661, +0.0025] p_holm 0.133 |

Flipped calls vs p0 at the threshold (right-to-wrong / wrong-to-right), no-verdict rate, collapse checks:

| arm | flips | no verdict | var(p)/var(p0) | AUC | AUC p0 | collapsed |
|---|---|---|---|---|---|---|
| co_investigator_ported_mem_on | 26 / 11 | 0.628 | 0.878 | 0.822 | 0.853 | True |
| co_investigator_ported_mem_off | 30 / 12 | 0.628 | 0.944 | 0.803 | 0.853 | True |

Per alert (means): questions, accepted-question share, eps, LLM calls, prompt tokens in / per call, tokens out, median / p90 s, GPU h.

| arm | questions | accepted | eps | calls | tok in | tok/call | tok out | median s | p90 s | GPU h |
|---|---|---|---|---|---|---|---|---|---|---|
| co_investigator_ported_mem_on | 2.30 | 0.26 | 1.00 | 7.55 | 14439 | 1911 | 1829 | 137.5 | 176.8 | 11.45 |
| co_investigator_ported_mem_off | 2.24 | 0.23 | 0.99 | 7.60 | 13547 | 1782 | 1815 | 110.0 | 136.2 | 8.93 |


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
| planner/evidence | planner_default | **planner_default** | default kept |
| planner/evidence_floor | planner_default | **planner_default** | default kept |
| planner/library | planner_default | **planner_default** | default kept |
| planner/menu | planner_default | **planner_default** | default kept |
| planner/rules | planner_default | **planner_default** | default kept |
| sharma_native/memory | sharma_native_mem_on | **sharma_native_mem_on** | default kept |
| sharma_ported/memory | sharma_ported_mem_on | **sharma_ported_mem_on** | default kept |
| torres_native/memory | torres_native_mem_on | **torres_native_mem_on** | default kept |
| torres_ported/memory | torres_ported_mem_on | **torres_ported_mem_on** | default kept |
| co_investigator_native/memory | co_investigator_native_mem_on | **co_investigator_native_mem_on** | default kept |
| co_investigator_ported/memory | co_investigator_ported_mem_on | **co_investigator_ported_mem_on** | default kept |
| finfre_rag_native/finfre_pool | finfre_rag_native_pool_bal | **finfre_rag_native_pool_bal** | default kept |
| finfre_rag_ported/finfre_pool | finfre_rag_ported_pool_bal | **finfre_rag_ported_pool_bal** | default kept |

**Recorded overrides (the pre-written selection rule was not applied to these arms):**

* `menu_topk_all`: invalid: failed the pre-written call-share check (29% < 30% of alerts made an LLM call on fit/c001); most alerts return p0 because the full menu overflows the prompt (PromptTooLong 0.79 per alert on fit). Reported descriptively; user decision 8 Nov 2026.
