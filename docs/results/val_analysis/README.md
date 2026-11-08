# Validation dry run of the final analysis

`scripts/final_analysis.py --split val` on the validation logs, packaged by `scripts/build_val_run.py` into
`outputs/analysis/val_run_v3` (inputs: `configs/analysis/inputs_val.yaml`). Every arm ran on the same 160 askable eval
alerts; p0 is the v2 one-bank score (`FTM_P0_VARIANT=v2`).

- `fixed_on_val.json`: the values the test analysis takes as fixed (`--fixed`). Equivalence margin 0.01, margin gap 0.166,
  best simple planner `fixed_order`, review capacity top 10%, no hub cap (largest cluster share 0.015).
  The best ported external agent is `bed_llm`, the frozen G4 pick in `configs/final/top_external.json`. The gap-closed rule
  in `ftm.eval.final.best_ported_external` picks `rewoo_adapted` on these 160 alerts; that name is kept in the file for
  information only, and H9 in `hypotheses.md` below was computed with it.
- `hypotheses.md`, `index.md`, `arms_status.md`: the dry run's tables. No hypothesis is supported on validation;
  H6, H10 and H5 are `not_run` here (no evidence-view arm, DP-FedAvg score or fast-model results on validation).
