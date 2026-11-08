# System 1 report (eval; n=160, positives 52; clusters: attempt+accounts (clusters file))

## Detection metrics per arm (threshold 0.3227)

| arm | Brier | AUC | PR-AUC | log loss | ECE | cal slope | precision | recall | F1 | escalation |
|---|---|---|---|---|---|---|---|---|---|---|
| off | 0.1893 | 0.7667 | 0.6593 | 0.6324 | 0.1537 | 0.4145 | 0.5000 | 0.7692 | 0.6061 | 0.5000 |
| selected | 0.1868 | 0.7692 | 0.6624 | 0.6266 | 0.1390 | 0.4258 | 0.4938 | 0.7692 | 0.6015 | 0.5062 |

## Paired against OFF (95% cluster-bootstrap CIs)

| arm | d Brier [CI] | p non-inf (Holm below) | d AUC [CI] | d log loss [CI] | flips r->w / w->r | calls/alert d | tokens_in d | questions d | eps d | median s | p90 s | speed-up median / mean [CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| selected | -0.0025 [-0.0061, 0.0005] | 0.0010 | 0.0025 [-0.0028, 0.0086] | -0.0058 [-0.0138, 0.0013] | 1 / 0 | 0.1000 [-0.0468, 0.2551] | 498.2500 [-291.6048, 1340.3890] | 0.0625 [-0.0135, 0.1395] | 0.0625 [-0.0148, 0.1416] | 13.88 (OFF 13.80) | 30.55 (OFF 29.56) | 0.99 / 0.95 [0.89, 1.02] |

## Tier shares per decision (live)

| arm | decision | n | fast | large | default | median s fast / large / default |
|---|---|---|---|---|---|---|
| selected | typology | 160 | 0.000 | 0.000 | 1.000 | n/a / n/a / 0.000 |

## Cascade on the decision-set eval rows (realised error vs the 2 alpha bound)

| decision | n | kept fast | to large | to default | error_marginal | bound | within | fast retired |
|---|---|---|---|---|---|---|---|---|
| injection_present | 960 | 0.759 | 0.000 | 0.241 | 0.0948 | 0.10 | True | False |
| rule_applies | 1280 | 0.000 | 0.177 | 0.823 | 0.0359 | 0.10 | True | False |
| typology | 159 | 0.000 | 0.000 | 1.000 | 0.0000 | 0.10 | True | True |
| worth_asking | 1280 | 1.000 | 0.000 | 0.000 | 0.0531 | 0.10 | True | False |

## Injection detection on the held-out val families (with the screen; without it recall is 0 by construction)

| family | n | attacks | recall [Wilson] | precision | false quarantine [Wilson] | settled share |
|---|---|---|---|---|---|---|
| all | 960 | 456 | 0.800 [0.761, 0.835] | 0.910 | 0.071 [0.052, 0.097] | 0.759 |
| encoded_base64 | 32 | 32 | 1.000 [0.893, 1.000] | 1.000 | n/a | 0.906 |
| encoded_hex | 29 | 29 | 0.655 [0.473, 0.801] | 1.000 | n/a | 0.345 |
| encoded_leet | 29 | 29 | 0.966 [0.828, 0.994] | 1.000 | n/a | 0.586 |
| encoded_rot13 | 39 | 39 | 0.692 [0.536, 0.814] | 1.000 | n/a | 0.385 |
| encoded_spaced | 31 | 31 | 0.903 [0.751, 0.967] | 1.000 | n/a | 0.355 |
| lookalike | 252 | 0 | n/a | 0.000 | 0.143 [0.105, 0.191] | 0.857 |
| multilingual | 111 | 111 | 0.802 [0.718, 0.865] | 1.000 | n/a | 0.640 |
| paraphrase | 185 | 185 | 0.768 [0.702, 0.823] | 1.000 | n/a | 0.584 |
| plain | 252 | 0 | n/a | n/a | 0.000 [-0.000, 0.015] | 1.000 |

## Selection (frozen from fit+cal, applied here)

functions on: typology; tiers: fast; per function: {'worth': False, 'inject': False, 'typology': True}
