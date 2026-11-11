# Test-split results (final run g5)

Copy of `outputs/final/g5/analysis/` as written by
`FTM_FINAL_RUN=1 python scripts/final_analysis.py --split test --run outputs/final/g5 --inputs configs/analysis/inputs_test.yaml --fixed outputs/analysis/val/fixed_on_val.json`
(freeze hash `b094561b57d2f8e3da6bd6945e8146904c8baec4b44910c633fce0b04722b97f`). Start with `index.md` and `hypotheses.md`.

`all_systems_metrics.json` holds the extra descriptive metrics (ROC-AUC, PR-AUC, ECE, precision and recall at 10%) per system, from `scripts/report_all_systems.py`; they feed the big comparison table of the report and never change a verdict.
