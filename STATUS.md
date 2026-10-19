# Status

One line per track per week (newest at the top). Say what is done, what is blocked, and what you need. Slipping? Say so in the week it happens (cut order: `plan.md` section 14).

| Week | T1 Abhishek | T2 Bhavya | T3 Akshat | T4 Shreyas |
|---|---|---|---|---|
| 2-3 | Done: GFP+LightGBM reproduced on the paper split (F1 62.95 LightGBM / 62.51 XGBoost vs 62.86 / 63.23 published); Multi-GNN Multi-PNA+EU reproduced on its split (test F1 67.64 vs paper 68.16, 5 of 100 epochs, 2.5 h); p0 (hand-offs #5, #17) for HI-Small one-bank + see-everything (Brier 0.001167 vs 0.000906, constant 0.001307: the gap is real) and HI-Medium one-bank; conformal intake with achieved-vs-target miss rates; draft train/val alert sets (HI-Medium val 2,238, train 10,962, projected test 3,475 alerts / 1,011 positives, a projection); edge-scorer data prep. Blocked: HI-Medium see-everything (compute, command in `ftm/baselines/README.md`); Multi-GNN on our split not started. Needs a team decision: the plan 5.2 drop rules empty the val set, the selective-risk bound conflicts with the alert budget, the 1,000-alert subset needs test labels (`ftm/intake/README.md`). Branch `t1/week2-4-models-intake` | - | - | Eval library done (metrics, cluster bootstrap, Holm; hand-off #16) and fusion v0 code done (naive/shared/per-family, shuffled-answers test; hand-off #18), synthetic tests only. Needs: T1 to run `scripts/eval_report.py` on real p0 (no data on this machine); T2 back-test table to fit fusion and run G2; team call on `docs/t4_decisions_week2.md`. Branch `t4/week2-eval-fusion` |
| 0 | Done except conversion benchmark (needs T3 script). Data hand-off #1 (HI-Small and HI-Medium) already built, see `ftm/data/README.md`; snapml/PyG OK; needs: TA OK on snapml licence, licence answer from Tian et al. authors. Branch `t1/week0-week1-data` | - | - | - |

## Week-0 machine report (everyone, before G0)

Fill in `docs/hardware.md` section 2 and tick your week-0 smoke tests.
