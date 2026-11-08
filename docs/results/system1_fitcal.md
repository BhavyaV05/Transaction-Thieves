# System 1 report (fitcal; n=300, positives 156; clusters: attempt+accounts (clusters file))

## Detection metrics per arm (threshold 0.3227)

| arm | Brier | AUC | PR-AUC | log loss | ECE | cal slope | precision | recall | F1 | escalation |
|---|---|---|---|---|---|---|---|---|---|---|
| off | 0.1592 | 0.8523 | 0.8362 | 0.6277 | 0.0809 | 0.4174 | 0.7211 | 0.8782 | 0.7919 | 0.6333 |
| worth | 0.1586 | 0.8509 | 0.8369 | 0.6261 | 0.0647 | 0.4201 | 0.7249 | 0.8782 | 0.7942 | 0.6300 |
| inject | 0.1582 | 0.8551 | 0.8381 | 0.6250 | 0.0873 | 0.4226 | 0.7166 | 0.8590 | 0.7813 | 0.6233 |
| typology | 0.1574 | 0.8537 | 0.8383 | 0.6239 | 0.0708 | 0.4244 | 0.7287 | 0.8782 | 0.7965 | 0.6267 |
| all_fast | 0.1577 | 0.8530 | 0.8363 | 0.6234 | 0.0939 | 0.4252 | 0.7354 | 0.8910 | 0.8058 | 0.6300 |
| all | 0.1581 | 0.8551 | 0.8383 | 0.6242 | 0.0760 | 0.4238 | 0.7283 | 0.8590 | 0.7882 | 0.6133 |

## Paired against OFF (95% cluster-bootstrap CIs)

| arm | d Brier [CI] | p non-inf (Holm below) | d AUC [CI] | d log loss [CI] | flips r->w / w->r | calls/alert d | tokens_in d | questions d | eps d | median s | p90 s | speed-up median / mean [CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| worth | -0.0006 [-0.0037, 0.0024] | 0.0439 | -0.0013 [-0.0079, 0.0046] | -0.0016 [-0.0094, 0.0057] | 3 / 4 | -0.0500 [-0.1197, 0.0224] | -323.9200 [-694.1811, 64.4611] | 0.0133 [-0.0350, 0.0605] | -0.0042 [-0.0505, 0.0401] | 13.39 (OFF 12.93) | 29.09 (OFF 29.22) | 0.97 / 1.01 [0.97, 1.05] |
| inject | -0.0009 [-0.0040, 0.0019] | 0.0214 | 0.0028 [-0.0027, 0.0089] | -0.0028 [-0.0095, 0.0037] | 3 / 0 | -0.0667 [-0.1502, 0.0270] | -388.3333 [-842.5934, 120.2635] | -0.0067 [-0.0586, 0.0441] | -0.0042 [-0.0459, 0.0367] | 12.89 (OFF 12.93) | 28.63 (OFF 29.22) | 1.00 / 1.03 [0.98, 1.08] |
| typology | -0.0017 [-0.0048, 0.0011] | 0.0048 | 0.0014 [-0.0063, 0.0083] | -0.0038 [-0.0117, 0.0037] | 2 / 4 | -0.0300 [-0.1175, 0.0664] | -158.7033 [-667.9330, 391.3916] | 0.0167 [-0.0345, 0.0705] | 0.0067 [-0.0302, 0.0449] | 13.08 (OFF 12.93) | 29.44 (OFF 29.22) | 0.99 / 1.01 [0.96, 1.06] |
| all_fast | -0.0014 [-0.0046, 0.0013] | 0.0055 | 0.0008 [-0.0044, 0.0070] | -0.0043 [-0.0121, 0.0020] | 1 / 6 | -0.0133 [-0.1016, 0.0821] | -97.7267 [-583.5137, 429.9659] | 0.0300 [-0.0135, 0.0769] | 0.0008 [-0.0349, 0.0351] | 13.43 (OFF 12.93) | 29.89 (OFF 29.22) | 0.96 / 0.99 [0.94, 1.03] |
| all | -0.0011 [-0.0046, 0.0019] | 0.0209 | 0.0028 [-0.0032, 0.0102] | -0.0035 [-0.0117, 0.0038] | 3 / 3 | -0.0400 [-0.1203, 0.0510] | -294.4067 [-720.8754, 177.2417] | 0.0033 [-0.0439, 0.0544] | -0.0033 [-0.0379, 0.0325] | 13.45 (OFF 12.93) | 28.08 (OFF 29.22) | 0.96 / 1.01 [0.96, 1.05] |

## Tier shares per decision (live)

| arm | decision | n | fast | large | default | median s fast / large / default |
|---|---|---|---|---|---|---|
| worth | worth_asking | 587 | 1.000 | 0.000 | 0.000 | 0.056 / n/a / n/a |
| inject | injection_present | 109 | 1.000 | 0.000 | 0.000 | 0.048 / n/a / n/a |
| typology | typology | 300 | 0.000 | 0.000 | 1.000 | n/a / n/a / 0.000 |
| all_fast | injection_present | 106 | 1.000 | 0.000 | 0.000 | 0.048 / n/a / n/a |
| all_fast | typology | 300 | 0.000 | 0.000 | 1.000 | n/a / n/a / 0.000 |
| all_fast | worth_asking | 598 | 1.000 | 0.000 | 0.000 | 0.056 / n/a / n/a |
| all | injection_present | 108 | 1.000 | 0.000 | 0.000 | 0.049 / n/a / n/a |
| all | typology | 300 | 0.000 | 0.000 | 1.000 | n/a / n/a / 0.000 |
| all | worth_asking | 590 | 1.000 | 0.000 | 0.000 | 0.056 / n/a / n/a |

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

## H5, per clause (exploratory, uncorrected)

| clause | decision | estimate | limit | met |
|---|---|---|---|---|
| bound | injection_present | 0.0948 | 0.1000 | True |
| bound | rule_applies | 0.0359 | 0.1000 | True |
| bound | typology | 0.0000 | 0.1000 | True |
| bound | worth_asking | 0.0531 | 0.1000 | True |
| most routine decisions (live: share settled by the fast tier) |  | 0.6994 | 0.5000 | True |
| most routine decisions (live: share settled by any tier) |  | 0.6994 | 0.5000 | True |
| >= 10x faster (end-to-end seconds per alert, median OFF / ON) |  | 0.9613 | 10.0000 | False |
| no detection loss (paired Brier ON - OFF, 95% CI upper bound <= delta) |  | -0.0011 | 0.0020 | True |
| no detection loss (flipped calls: right-to-wrong <= wrong-to-right) |  | {"right_to_wrong": 3, "wrong_to_right": 3, "n": 300, "changed_calls": 6, "net": 0} | n/a | True |

## Selection (derived on fit+cal)

functions on: typology; tiers: fast; per function: {'worth': False, 'inject': False, 'typology': True}
