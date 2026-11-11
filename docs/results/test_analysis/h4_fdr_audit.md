## H4 FDR audit: accepted questions asked at least 30 times

| question | eps | asked | realised_log_lr_in_claimed_direction | useless |
|---|---|---|---|---|
| n_counterparties(ALERT.sender, dir=in, window=before(72)) | 1 | 40 | -0.0528 | True |
| n_counterparties(ALERT.sender, dir=out, window=after(24)) | 1 | 32 | -0.0712 | True |
| new_account(ALERT.receiver, days=2) | 1 | 118 | -0.1332 | True |
| new_account(ALERT.sender, days=2) | 1 | 87 | 0.0951 | False |
| matching_amount_out(ALERT.sender, tol=0.05) | 1 | 30 | -0.0319 | True |
| matching_amount_out(ALERT.receiver, tol=0.05) | 1 | 34 | 0.8144 | False |
| n_counterparties(ALERT.receiver, dir=in, window=before(72)) | 1 | 157 | -0.0312 | True |
