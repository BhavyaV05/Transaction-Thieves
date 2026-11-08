# Final analysis: run val_v3 (val)

Validation dry run: the test tables are built the same way once the final run is analysed with `--split test`.

- alerts: 806, positives 198, clusters 650 (largest share 0.015)
- subset_sha256: 2a681a5eff23a8f39fc0333eecd9588e413f263a8bb627605ee8cd25737b4785
- arms: 57 present, 0 missing

## Hypotheses

| hypothesis | tier | estimate | 95% CI | p (Holm) | verdict |
|---|---|---|---|---|---|
| H1 | primary | -0.015 | [-0.142, 0.091] | 1.0000 | not_supported |
| H2 | primary | -0.067 | [-0.663, 0.242] | 1.0000 | not_supported |
| H3 | primary | -0.003 | [-0.008, 0.003] | 1.0000 | not_supported |
| H4 | primary | 0.000 | [-0.006, 0.007] | 1.0000 | not_supported |
| H9 | primary | -0.044 | [-0.298, 0.087] | 1.0000 | not_supported |
| H6 | secondary |  |  |  | not_run |
| H7 | secondary | -0.001 | [-0.006, 0.005] | 1.0000 | not_supported |
| H10 | secondary |  |  |  | not_run |
| H5 | exploratory (uncorrected) |  |  |  | not_run |
| H8 | exploratory (uncorrected) | n/a | [-1.934, 47.683] |  | not met (exploratory, uncorrected) |

With 806 alerts the plan's rule of thumb gives a detectable Brier difference of about 0.0245 (0.011 at 4000 alerts, scaled by 1/sqrt(n)). On the 160-alert external subset it is about 0.0550. From the cluster-bootstrap standard error of the paired difference between our system and one bank on this run, the smallest difference detectable with 80% power at two-sided 5% is 0.0070 Brier. At this run's own paired standard error (ours minus one bank, H1) of 0.0025 Brier, detecting a difference of 0.01 (80% power, two-sided 5%) needs about 79 alerts, against 160 here; at the projected test size of 3475 alerts the detectable difference is about 0.0015 Brier. Where the target is far below the detectable difference, 'not met' and 'not detectable' look the same.

## Produced

- hypotheses: hypotheses.md
- arms_status: arms_status.md
- hypotheses_components: hypotheses_components.md
- h4_fdr_audit: h4_fdr_audit.md
- power: power.txt
- table_5a: table_5a.md
- table_5b: table_5b.md
- privacy_curve: figures/privacy_curve.csv
- breakdown_typology: breakdown_typology.md
- breakdown_bank: breakdown_bank.md
- n_alerts: n_alerts.md
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
- pareto: figures/pareto.csv
- external_gap: figures/external_gap.csv
- reliability: figures/reliability.csv

## Not produced

- end_to_end_all_rows: the all-rows table is not provided (inputs.yaml rows.path: label, p0, alert_id, attempt_id, src_id, dst_id)
- injection: no step row carries an injection_id: the E15 injection runs are not in this run
- eps_infinity: eps-sweep JSON not provided (inputs.yaml eps_sweep.path, e.g. data/runs/eps_sweep_hi_medium.json from scripts/eps_sweep.py)
- p0_recalibration: inputs.yaml p0_recal.path is not set (scripts/fit_p0_recal.py writes it)
