## Detection models on the headline alerts (reproduction checks: T1's baselines_val_table.py)

| model | n | positives | brier | brier skill | pr_auc | ece | look-ahead |
|---|---|---|---|---|---|---|---|
| GFP + LightGBM, one bank | 3873 | 975 | 0.1585 | 0.1587 | 0.6015 | 0.1187 | none |
| GFP + LightGBM, see everything | 3873 | 975 | 0.0854 | 0.5466 | 0.8477 | 0.0406 | minor |
