# Headroom and tuning of the non-LLM question pipeline (headroom_v1)

Worker A1, rewrite of 1 Nov.  Code: `ftm/fusion/headroom.py`, `scripts/t3_headroom.py` (`select`, then `eval`), tests
`tests/test_headroom.py`.  Frozen files: `configs/fusion/headroom_v1*.{json,jsonl}`, `configs/fusion/headroom_v1_decisions.json`,
`configs/decision/headroom_v1.yaml`, `configs/final/non_llm.yaml`.  CPU only; test never touched.

## Protocol (anti-overfitting)

* Every choice (fusion form, ridge, question set, eps allocation, base calibration, shrink, thresholds, stop bounds) was made on
  TRAIN (5 interleaved time-block folds, features out-of-fold) and/or val `fit`+`cal` (300 askable alerts, scored cross-part:
  fit on `fit`, score `cal`, and the reverse).  Val `eval` rows are dropped on load in `select`.
* `select` wrote `configs/fusion/headroom_v1_decisions.json` (sha256 `3ca483b6...37a52c3`) and committed it (d269338) BEFORE `eval`
  ran.  `eval` refuses to run if any frozen file's sha256 changed and prints the decisions hash.  The eval numbers below are from
  that single run (160 askable alerts, 133 clusters, cluster bootstrap of `ftm/eval/bootstrap.py`, 10,000 resamples; clusters are
  laundering attempts, alerts without one are their own cluster; account-component merging was not used).
* Data: train back-test 7613b89b5dcc (K=4 time blocks, history ledgers, 440 hypotheses = 110 programs x eps 0.25/0.5/1/2; the
  re-assemble now covers all 440, e-BH accepts 6, all at eps 2); the same 440 hypotheses run on every askable val alert (namespace
  f8b676b7d1fc; 460 alerts: fit 191, cal 109, eval 160).  Val answers are re-binned with the TRAIN table's bins.  p0 = v2
  (`data/alerts/hi_medium_v2`, train cross-fitted isotonic); v1 reported for reference.
* Offline simulation: each (alert, hypothesis) has one stored DP answer from the right bank, so any policy is replayed on the stored
  answers.  Not modelled: the per-account cap (an alert's eps budget 2 already bounds it; cross-alert account reuse is ignored), and
  the harness gate.

## What the data looks like

* TRAIN askable alerts: 1983, label rate 0.187; p0 v2 is weak here (AUC 0.58; Brier 0.173 vs 0.152 for a constant).  VAL fit+cal: 300
  alerts, label rate 0.52 (fit 0.57, cal 0.43), eval 0.325; p0 v2 AUC 0.855 on fit+cal, 0.76 on eval.  So p0 on val is a different,
  much stronger object than p0 on train, and the label rate drifts hard across splits.
* Consequence: a base (a, b) fitted on train is useless on val (Brier 0.24), and recalibrating p0 on val does not help under drift:
  `recal` fitted on `fit` and scored on `cal` (and reverse) gives 0.172 vs 0.157 for raw p0 (the exact-1.0 scores of val p0 wreck a
  logit fit; logit(p0) is clipped to [0.1, 0.9] everywhere, `ConditionalFusion.p0_clip`).  Three bases were compared on val:
  `recal` (a, b fitted), `intercept` (a only), `p0` (none).  `p0` won cross-part, so the frozen model adds evidence to clipped raw p0
  with no fitted (a, b).

## Fusion forms (TRAIN, cross-fitted time blocks; Brier, lower is better; recalibrated p0 alone = 0.15145)

| form | eps-2 single ask | 2 x eps-1 | 4 x eps-0.5 | any eps |
|---|---|---|---|---|
| (a) ConditionalFusion form, 6 e-BH-accepted hypotheses only (ridge 1-100) | 0.1511 | 0.1514 | 0.1514 | 0.1512 |
| (b) per-family weights over ALL 440, prior 1, L2 = 3 (CHOSEN) | 0.1487 | 0.1509 | 0.1501 | 0.1500 |
| (b) same, L2 = 1 / 10 / 30 / 100 | 0.1487 / 0.1489 / 0.1491 / 0.1492 | | | |
| (b) per-family, prior 0 (weights shrunk to zero) L2 = 3 / 30 | 0.1493 / 0.1508 | | | |
| (b) per-hypothesis weights, prior 1, L2 = 3 | 0.1492 | 0.1508 | 0.1500 | 0.1503 |
| (b) answer-bin effects (one-hot, ridge to 0) L2 = 3 / 30 | 0.1510 / 0.1513 | 0.1540 / 0.1514 | 0.1574 / 0.1515 | 0.1557 / 0.1513 |
| (b) answer-bin effects, elastic net (one setting, l1 = 100) | 0.1515 (shrunk to nothing) | | | |
| (c) family weights x logit-p0 interaction | 0.1487 | 0.1509 | 0.1501 | 0.1500 |
| every answer observed (dense, prior 0, L2 = 10) | 0.0997 (information bound on train) | | | |

Reading: the e-BH acceptance does discard usable weak signal.  The accepted six give nothing on train (0.0003), the full set gives
0.0027 (eps-2) / 0.0013 (any eps) over recalibrated p0.  The gain is spread over many families, not a few lucky ones, and it needs
shrinkage toward naive Bayes (prior 1; prior 0 loses most of it).  Answer-bin effects and the p0 interaction do not help (interaction
rejected: it changes Brier by < 0.0001; `interaction_supported = false`).  With every answer observed the train information bound is
0.0997, so the budget (2 eps per alert), not the question set, is the bottleneck.

## Val fit+cal: base, question set, allocation, shrink (cross-part Brier; 96 combinations, 300 alerts)

Floors on fit+cal (in-sample): constant 0.2496; p0 v1 raw 0.1554; p0 v2 raw 0.1571; p0 v2 clipped [0.1, 0.9] 0.1557; recalibrated
(a, b) 0.1538; intercept-only 0.1540.  Cross-part (the honest recalibration floors): recal 0.1720, intercept 0.1710, clipped raw p0
0.1557 (nothing fitted).  Best combinations: all 440 / p0 base / any eps / shrink 0.5 = 0.1483; accepted-only / p0 / eps-2 / 0.75 =
0.1493; all / p0 / eps-2 / 0.5 = 0.1501.  The accepted-only form therefore also helps on val (0.0064 over the clipped-p0 floor) while it
did not on train: the val sample is small (SE of a Brier difference is about 0.005), so the two disagree within noise; the full set was
chosen by the pre-set rule (lowest cross-part val Brier) and it also has the train support.

## Frozen choice (headroom_v1)

* Fusion: `logit P = logit(clip(p0, 0.1, 0.9)) + 0.5 * sum_f w_f * logLR_f` (ConditionalFusion, base "p0", p0_clip 0.1, shrink
  0.5).  w_f from the train fit (L2 = 3 toward 1, out-of-fold LRs, random-policy ask draws), 28 families, weights 0.23-1.51 (mean 0.7).
  Every one of the 440 hypotheses is usable: `configs/fusion/headroom_v1_table.jsonl` is the train table with `accepted` = true for them
  (`accepted_ebh` keeps the e-BH verdict).  File `configs/fusion/headroom_v1.json`, content hash `fd0dee423a098b48`, sha256
  `62edba610664063280f17d142958d99d83c36051fa7d61927fbefd0282e76f9a`; table sha256 `9b828439...b8e1`.
* Secondary, the "ConditionalFusion as is" variant: accepted six only, eps-2 only, shrink 0.75
  (`headroom_v1_accepted.json`, content hash `bae009e41a7f849a`, sha256 `fdbc67b4...2ef7`, table `headroom_v1_accepted_table.jsonl`).
* Eps allocation: no fixed split.  EIG per unit eps over the whole menu (0.25-2) beats one eps-2 ask, two eps-1 and four eps-0.5 on
  fit+cal (0.1483 vs 0.1501 / 0.1557 / 0.1523), because a cheap question can be asked several times.  The planner should offer
  all four eps levels (`eps_menu` in `configs/final/non_llm.yaml`).
* Decision (`configs/decision/headroom_v1.yaml`): escalate when posterior >= 0.3227 (maximises F1 on fit+cal; the 1:1 cost optimum
  is the same point; the 10% review-capacity quantile is 0.9156 and is listed, not chosen; the pre-registered objective stays
  `capacity`, default.yaml is untouched).  `cannot_cross` bounds derived by `stop_bounds` from the 440 table rows and the weights:
  `max_delta` 2.0 per eps, `max_weight` 0.756 (= max|w| x shrink).  On fit+cal the rule cuts asks per alert from 3.69 to 1.64 and costs
  0.0016 Brier (0.1483 -> 0.1499): useful for saving budget, not for Brier.

## Ceilings (Brier; cross-checks in the notes)

All rows use the frozen model, eps budget 2 per alert, right-bank answers, one question per family.  "fit+cal" is in-sample for the
frozen shrink (and for the recalibration floors, which are refit); the honest cross-part values are marked.  "eval" is the single
run after the freeze (160 alerts).

| row | fit+cal (300) | eval (160) |
|---|---|---|
| constant base rate (fit+cal rate) | 0.2496 | |
| p0 v1 raw | 0.1554 | 0.1985 |
| p0 v2 raw | 0.1571 | 0.1896 |
| p0 v2 recalibrated, (a, b) on fit+cal | 0.1538 in-sample / 0.1720 cross-part | 0.1960 |
| p0 v2 intercept-only | 0.1540 / 0.1710 | 0.1949 |
| p0 v2 clipped [0.1, 0.9], nothing fitted ("same base") | 0.1557 | 0.1888 |
| information ceiling, all 110 programs at eps 2 (budget ignored, dense model) | 0.1429 | 0.1862 |
| information ceiling, all 440 answers (budget ignored) | 0.1411 | 0.1844 |
| budgeted EIG, label-free, any eps (FROZEN) | 0.1483 (cross-part 0.1483) | 0.1799 |
| EIG, one eps-2 ask | 0.1501 | 0.1818 |
| EIG, two eps-1 | 0.1557 | 0.1856 |
| EIG, four eps-0.5 | 0.1523 | 0.1952 |
| random questions (30 seeds) | 0.1554 | 0.1888 |
| fixed best order (train-ranked, one ask) | 0.1552 | 0.1962 |
| EIG, accepted-six variant | 0.1493 | 0.1812 |
| UNATTAINABLE oracle: label-aware greedy eps-feasible set | 0.1076 | 0.1318 |
| UNATTAINABLE oracle: label-aware best single ask | 0.1120 | 0.1388 |

The oracle rows look at the label and pick, per alert, the answer that happens to push toward it (selection on DP noise); they say
how much an omniscient picker could extract from the stored answers, not what a planner can reach.  The information-ceiling rows are
a dense model fitted on train and rescaled on the rows shown: on fit+cal they are the lowest attainable-looking numbers (0.141), on
eval they are WORSE than the budgeted EIG (0.184 vs 0.180), i.e. they are not true ceilings out of sample; read them as "fit+cal says
all answers would be worth ~0.014 over clipped p0, eval shows no extra gain".

### Eval (once): paired differences, cluster bootstrap 95%

| Brier(EIG fused) minus | mean | 95% CI |
|---|---|---|
| p0 recalibrated (frozen a, b from fit+cal) | -0.0161 | [-0.0270, -0.0056] |
| p0 clipped, nothing fitted (same base) | -0.0089 | [-0.0192, +0.0017] |
| p0 v2 raw | -0.0097 | [-0.0201, +0.0011] |
| p0 v1 raw | -0.0186 | [-0.0304, -0.0073] |

Eval Brier: fused 0.1799 [0.1451, 0.2179]; p0 v2 0.1896 [0.1529, 0.2295]; recalibrated 0.1960.  At the F1 threshold 0.3227 the fused
posterior escalates 80 of 160 alerts, precision 0.50, recall 0.77 (base rate 0.325).  The accepted-six variant: -0.0148 vs recalibrated
[-0.0281, -0.0026], -0.0084 vs raw p0 v2 [-0.0213, +0.0041].

## Components vs their ceilings

* p0: the honest floor is raw/clipped p0 (0.1557 fit+cal, 0.1888 eval); fitted recalibration is worse than leaving p0 alone under the
  fit-to-cal and val-to-eval drift.  p0 v1 is slightly better than v2 on fit+cal (0.1554 vs 0.1571) and worse on eval (0.1985 vs 0.1896).
* Fusion: tuned fusion closes about 0.0074 of the 0.0481 Brier between clipped p0 and the label-aware oracle on fit+cal (15%) and
  0.0089 of 0.0570 on eval (16%); against the dense information bound it closes about 0.0074 of 0.0146 (51%) on fit+cal.
* Planner: label-free EIG beats random by 0.007 (fit+cal) and 0.009 (eval) and fixed-order by 0.007 / 0.016; random and fixed order
  are no better than p0 alone.  The e-BH-accepted-only EIG gets within 0.001 of full EIG, but only 57-60% of alerts have a
  right-bank answer for those six questions.

## Collapse checks

Frozen model on fit+cal / eval: slope of the posterior's logit on logit(clipped p0) 1.01 / 1.07 (not shrunk to a constant; the base
is p0 itself, b = 1 fixed), posterior variance 1.03x / 1.10x that of the p0 floor, AUC 0.870 / 0.789 vs floor 0.855 / 0.757,
calibration slope 1.10 / 0.76 (eval over-confident, as raw p0 is: 0.71), mean posterior 0.50 / 0.41 vs base rate 0.52 / 0.325.  The
gain is not shrinkage: it beats the same-base floor, ranks better (AUC +0.015 / +0.032), and the fusion also beats the recalibrated
floor, which the variants with fitted (a, b) never did cross-part.  Mean posterior vs base rate on eval (0.41 vs 0.325) shows the
prior-shift problem is untouched: no variant here fixes the label-rate drift between splits.

## Honest conclusion

At eps 2 per alert, questions do add signal over p0, but it is small relative to noise: about 0.007-0.009 Brier over clipped p0
(5%), on both fit+cal and eval, and the eval gain over the same-base p0 and over raw p0 v2 has a CI that includes zero
([-0.019, +0.002]).  The gain over recalibrated p0 is clearly significant, but that comparison is soft (recalibration hurts under
drift).  Most of the benefit comes from using many weakly informative questions (the e-BH-accepted six alone give almost as much on
fit+cal and slightly less on eval) and from letting the planner spend the budget on several cheap asks.  The budget, not the
question set, is the limit (train information bound 0.0997 vs 0.149 achieved).  A planner cannot approach the oracle (0.108 / 0.132):
that gap is selection on noise.  H1's "30% of the gap" must be judged against the one-bank p0 and the raw-share upper bound, which
this study does not recompute.

## Not done

Elastic net was run for one setting on answer-bin effects only (saga is slow: > 10 min per grid point); L1 on the family form was
not run.  The per-account cap and gate are not simulated.  The p0 interaction was tested on train only.  The harness-level
(LangGraph) pipeline was not run with the frozen files: load them with `--fusion-params configs/fusion/headroom_v1.json --backtest`
pointing at `configs/fusion/headroom_v1_table.jsonl` (ConditionalLR; `FusedLR.from_params` in `run_llm_agent.py` expects the older
`Fusion.params()` format and needs the `kind == "conditional_fusion"` branch the graph worker owns).
