# Final analysis: run g5 (test)

- alerts: 3873, positives 975, clusters 2552 (largest share 0.012)
- git_commits: ['479d1d44855a0f0b03ccfaa790fde113fb64d9f1', '4b727cc66e2afe455adcb2d00a69b545bcc7e6b9', '878d55ceb87e729ccd94b8a7989c10d898677021', '9c9cc1e6288172003213756978749e5a997c7ea9', 'bfb5165573d0d9ef18b5edbbb46d5aa0ba772f7b', 'ff3a5ab5c1f5025473657c9807a6734f8ff855cd']
- freeze_sha256: b094561b57d2f8e3da6bd6945e8146904c8baec4b44910c633fce0b04722b97f
- subset_sha256: 2faf01a4c6aa9db53622aff6577e0d6aade0fd95551dd872bf63c6c5921f94bd
- full_test_sha256: 4643ebea8a61f7899669750911975b1b619566a8e0431368f528220970174d84
- arms: 25 present, 86 missing (partial run: missing arms are marked not_run below)

## Hypotheses

| hypothesis | tier | estimate | 95% CI | p (Holm) | verdict |
|---|---|---|---|---|---|
| H1 | primary | -0.007 | [-0.038, 0.026] | 1.0000 | not_supported |
| H2 | primary | -0.165 | [-2.882, 0.750] | 1.0000 | not_supported |
| H3 | primary | -0.001 | [-0.006, 0.003] | 0.3309 | not_supported |
| H4 | primary | -0.005 | [-0.010, -0.000] | 0.0720 | not_supported |
| H9 | primary | 0.266 | [-0.022, 0.799] | 0.0040 | supported |
| H6 | secondary |  |  |  | not_run |
| H7 | secondary | -0.000 | [-0.004, 0.004] | 1.0000 | not_supported |
| H10 | secondary |  |  |  | not_run |
| H5 | exploratory (uncorrected) |  |  |  | not_run |
| H8 | exploratory (uncorrected) |  |  |  | not_run |

With 3873 alerts the plan's rule of thumb gives a detectable Brier difference of about 0.0112 (0.011 at 4000 alerts, scaled by 1/sqrt(n)). On the 1000-alert external subset it is about 0.0220. From the cluster-bootstrap standard error of the paired difference between our system and one bank on this run, the smallest difference detectable with 80% power at two-sided 5% is 0.0030 Brier. At this run's own paired standard error (ours minus one bank, H1) of 0.0011 Brier, detecting a difference of 0.01 (80% power, two-sided 5%) needs about 70 alerts, against 771 here; at the projected test size of 3475 alerts the detectable difference is about 0.0014 Brier. Where the target is far below the detectable difference, 'not met' and 'not detectable' look the same.

## Warnings

- only MANIFEST.partial.json exists: the run is not complete, missing arms are marked
- 86 job(s) of the manifest are not done and are marked missing: multi_gnn.one_bank.subset, multi_gnn.one_bank.full_test, multi_gnn.see_everything.subset, multi_gnn.see_everything.full_test, fraudgt.one_bank.subset, fraudgt.one_bank.full_test, fraudgt.see_everything.subset, fraudgt.see_everything.full_test, mega_gnn.one_bank.subset, mega_gnn.one_bank.full_test, mega_gnn.see_everything.subset, mega_gnn.see_everything.full_test ...
- 9 subset job(s) cover only the 208 askable alerts of the 1000 frozen ids (20.8%); headline comparisons use the askable alerts, the all_alerts table scores the rest with p0 (no verdict = no change)
- 4 full_test job(s) cover only the 892 askable alerts of the 4478 frozen ids (19.9%); headline comparisons use the askable alerts, the all_alerts table scores the rest with p0 (no verdict = no change)
- 4114 verdict rows are for alerts outside the headline alert set (dropped)

## Produced

- hypotheses: hypotheses.md
- arms_status: arms_status.md
- hypotheses_components: hypotheses_components.md
- h4_fdr_audit: h4_fdr_audit.md
- power: power.txt
- table_5a: table_5a.md
- table_5b: table_5b.md
- privacy_curve: figures/privacy_curve.csv, figures/privacy_curve.png
- breakdown_typology: breakdown_typology.md
- breakdown_bank: breakdown_bank.md
- n_alerts: n_alerts.md
- job_coverage: job_coverage.md
- all_alerts: all_alerts.md
- stop_reasons: stop_reasons.md
- compute: compute.md
- compute_by_machine: compute_by_machine.md
- e18_sweep: e18_sweep.md
- table_baselines: table_baselines.md
- case_studies: case_studies.md
- error_tags: error_tags.md
- fusion_calibration: fusion_calibration.md
- shuffled_answers: shuffled_answers.md
- review_capacity: review_capacity.md
- adjusted_estimates: adjusted_estimates.md
- pareto: figures/pareto.csv, figures/pareto.png
- external_gap: figures/external_gap.csv, figures/external_gap.png
- reliability: figures/reliability.csv, figures/reliability.png

## Not produced

- end_to_end_all_rows: the all-rows table is not provided (inputs.yaml rows.path: label, p0, alert_id, attempt_id, src_id, dst_id)
- injection: no step row carries an injection_id: the E15 injection runs are not in this run
- eps_infinity: eps-sweep JSON not provided (inputs.yaml eps_sweep.path, e.g. data/runs/eps_sweep_hi_medium.json from scripts/eps_sweep.py)
- p0_recalibration: inputs.yaml p0_recal.path is not set (scripts/fit_p0_recal.py writes it)
