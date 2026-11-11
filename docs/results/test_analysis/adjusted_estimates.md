## S2: p0-adjusted paired Brier differences of H1 and H3 (sensitivity, unadjusted is primary)

| hypothesis | comparison | n | clusters | unadjusted | unadjusted 95% CI | unadjusted CI width | adjusted | adjusted 95% CI | adjusted CI width | width ratio (adjusted / unadjusted) | slope on p0 | slope on p0 95% CI | slope on answered questions | slope on answered 95% CI | R2 (in sample) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1 | Brier(ours) - Brier(one bank) | 771 | 402 | 0.000465 | [-0.0017, 0.0026] | 0.0043 | 0.000465 | [-0.0017, 0.0025] | 0.0043 | 0.9974 | 0.0014 | [-0.0098, 0.0120] | 0.000476 | [-0.0047, 0.0055] | 0.000127 | sensitivity, unadjusted is primary |
| H3 | Brier(ours) - Brier(best simple (fixed_order)) | 208 | 167 | -0.0013 | [-0.0063, 0.0034] | 0.0096 | -0.0013 | [-0.0062, 0.0034] | 0.0096 | 0.9973 | -0.0021 | [-0.0289, 0.0204] | -0.0025 | [-0.0140, 0.0077] | 0.0019 | sensitivity, unadjusted is primary |
| H3 | Brier(ours) - Brier(enumerator) | 208 | 167 | 0.0007171 | [-0.0143, 0.0156] | 0.0299 | 0.0007171 | [-0.0142, 0.0158] | 0.0299 | 1.0005 | -0.0005769 | [-0.0625, 0.0602] | -0.0071 | [-0.0369, 0.0219] | 0.0022 | sensitivity, unadjusted is primary |
| H3 | Brier(ours) - Brier(llm fixed list) | 208 | 167 | -0.008 | [-0.0131, -0.0031] | 0.01 | -0.008 | [-0.0132, -0.0030] | 0.0102 | 1.0183 | -0.019 | [-0.0436, 0.0021] | -0.0083 | [-0.0189, 0.0010] | 0.0208 | sensitivity, unadjusted is primary |

S2 (sensitivity, unadjusted is primary; outside the Holm families, never changes a verdict). Per-alert Brier difference (negative = ours better) regressed by OLS on p0 and the number of answered questions of our system; the adjusted estimate is the fit at the original covariate means, so it equals the unadjusted mean by construction and only its CI differs; cluster bootstrap refits the regression on every resample of whole clusters.
