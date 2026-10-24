# E12: held-out-cycles rerun (week 8, T1)

Owner: T1 (Abhishek). Spec: tracks.md 3.3 item 7, 3.9 week 8. Train and val only, no test row. T4 refits fusion and T2 reruns the transfer analysis from these files; T1 did the data, GFP, conformal intake and ran T2's held-out back-test with T2's scripts, unchanged.

## What was run

- **Switch.** `configs/data/<ds>_heldout.yaml` (`heldout_cycles: true`, alert sets in `data/alerts/<ds>_heldout/`) and `ftm.data.io(heldout_cycles=True)`: the cycle attempts' rows are removed from the **train** rows of every view; val is untouched, so cycle alerts can still be evaluated. `scripts/train_gfp.py --data <ds>_heldout` honours the config. Rows dropped: all train transactions of the attempts typed `cycle` in the pattern file (HI-Small 36 attempts, 122 transactions of the 1,312 train pattern transactions; HI-Medium 202 attempts, 958 of 8,744).
- **GFP one-bank** on all 30 views for both datasets (same settings, seed 0, `neg_frac` 0.3 on HI-Medium); **see-everything for HI-Small** (`--threads 6`). Scores `data/scores/<ds>_heldout/`.
- **Conformal intake** with the same alpha_launder as the full run (0.60 / 0.63; `configs/intake/conformal.yaml`), thresholds recomputed on the held-out model's val `cal` part: `data/alerts/<ds>_heldout/{train,val}.parquet` (same schema, with `direction`, `counterparty_bank`; `top_features` not built for these sets).
- **T2's held-out back-test** (`scripts/backtest.py --data <ds>_heldout --split train --heldout-cycles --eps 0.25 0.5 1 2`, 110 expert programs, 440 hypotheses, then `--assemble`) and `ftm.backtest.heldout.transfer_report` on val, via `scripts/heldout_report.py`. Output `data/runs/heldout_<ds>.json`.

## Detection (GFP one-bank p0, val `eval` part, laundering rows by typology)

| | HI-Small full | HI-Small held-out | HI-Medium full | HI-Medium held-out |
|---|---|---|---|---|
| alert threshold t_alert (p0) | 0.0323 | 0.0323 | 0.1292 | 0.1468 |
| clean rows alerted | 0.30 % | 0.34 % | 0.19 % | 0.15 % |
| **cycle** laundering rows (n) | 6 | 6 | 35 | 35 |
| cycle: mean p0 | 0.342 | 0.392 | 0.407 | 0.369 |
| cycle: share of rows alerted | 0.833 | 0.833 | 0.657 | 0.629 |
| cycle: AUC against clean rows | 0.9974 | 0.9985 | 0.9832 | 0.9846 |
| non-cycle typed laundering rows, share alerted (71 / 232 rows) | 0.59 | 0.58 | 0.74 | 0.71 |
| Brier one-bank (val eval rows) | 0.001167 | 0.001152 | 0.001451 | 0.001474 |
| PR-AUC one-bank | 0.202 | 0.207 | 0.233 | 0.219 |
| Brier see-everything / gap one-bank minus see | 0.000906 / +0.000261 | 0.000920 / +0.000232 | n/a (blocked) | n/a |

Per-typology rows for every typology are in `data/runs/heldout_<ds>.json` (`detection`).

## Intake (val alert sets)

| | HI-Small full | HI-Small held-out | HI-Medium full | HI-Medium held-out |
|---|---|---|---|---|
| val alerts / positives | 1,284 / 177 | 1,307 / 173 | 2,238 / 798 | 1,950 / 764 |
| train alerts / positives | 5,198 / 255 | 5,403 / 250 | 10,962 / 1,448 | 9,060 / 1,152 |
| **cycle attempts alerted** (attempt recall) | 7 of 24 (0.29) | 7 of 24 (0.29) | 34 of 161 (0.21) | 32 of 161 (0.20) |
| cycle laundering tx alerted | 0.15 | 0.15 | 0.15 | 0.14 |
| other typologies, attempt recall (range over 7 typologies) | 0.33 to 0.67 | 0.33 to 0.67 | 0.21 to 0.38 | 0.20 to 0.39 |

## Back-test transfer (T2's `transfer_report`; separation = mean log-LR of a laundering alert's answer minus a clean one's, 0 = no information)

Accepted questions (e-BH, FDR 10 %): HI-Small full 1, held-out 5; HI-Medium full 23, held-out 22. Both tables are scored on the **same** val alerts (the held-out-world val set; askable alerts only).

| val alerts | HI-Small full table | HI-Small held-out table | HI-Medium full table | HI-Medium held-out table |
|---|---|---|---|---|
| cycle vs clean (alerts; laundering cycle alerts) | no question scoreable (2 cycle launderers) | same | 23 questions, separation **0.060** (61 % positive) | 22 questions, **0.063** (59 % positive) |
| other typologies vs clean | 1 question, 0.378 | 5 questions, 0.479 | 23 questions, 0.068 (52 % positive) | 22 questions, 0.109 (59 % positive) |

## Reading (limitations first)

- **Samples are tiny for cycles.** HI-Small val has 2 askable cycle laundering alerts and 6 cycle rows in the eval part; nothing can be concluded there. HI-Medium has 26 askable cycle launderers (35 cycle rows in eval, 161 cycle attempts in val); everything below is about HI-Medium and is descriptive (no CI was computed; one seed).
- **Removing cycles from training did not hurt cycle detection.** The cycle rows' AUC against clean rows is unchanged (0.983 to 0.985 HI-Medium; 0.997 to 0.999 HI-Small); the alerted share moves from 0.66 to 0.63 (HI-Medium), the mean p0 from 0.41 to 0.37. The reason is on the feature side: the GFP features include temporal and length-constrained cycle counts (`temp-cycle`, `lc-cycle`) that a tree model can use on cycles it never saw as labelled examples, because the same graph structure raises the score of laundering in other typologies (scatter-gather, stack) and clean cycles are rare.
- **Other typologies change as much as cycles** (HI-Medium mean share alerted: fan-out 0.78 to 0.70, random 0.79 to 0.57 with 14 rows, scatter-gather 0.60 to 0.65, bipartite 0.76 to 0.72): the differences are of the size of the retraining noise on a few dozen rows, not a cycle-specific effect. The held-out model's threshold moved (0.129 to 0.147), so alert sets are not nested.
- **Intake recall for cycle attempts is low in both runs** (21 % of cycle attempts have an alerted transaction on HI-Medium vs 34 to 38 % for bipartite, fan-out, scatter-gather and stack) and holding cycles out lowers it by 1 point (34 to 32 attempts). Cycles are the hardest typology for this intake with or without training examples.
- **The accepted questions transfer.** The held-out table (trained without cycles) separates the held-out typology as well as the full one: 0.063 vs 0.060 mean separation over 22 and 23 questions, and 59 % vs 61 % of the questions have a positive separation on cycle alerts (HI-Medium). The questions that survive FDR control are almost the same set (22 of 23). Outside cycles the held-out table separates better (0.109 vs 0.068), which is a different accepted set on the same alerts, not an effect of cycles.
- For T4 (fusion refit) and T2: the held-out alert sets and p0 are ready; the back-test caches are `data/backtest/<ds>_heldout/<namespace>/` (HI-Small `6743490a410f`, HI-Medium `729f6c4c0a11`). Not done: fusion refit (T4), the LLM planner under held-out cycles (T3).
