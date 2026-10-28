# T4: the final analysis pipeline

Owner: T4 (Shreyas). Spec: `prereg/hypotheses.yaml`, `plan.md` 7-9, `docs/schedule_nov10.md` (phase C-D). Code: `scripts/final_analysis.py`, `ftm/eval/final.py` (loading, scoring, hypotheses), `ftm/eval/final_tables.py` (tables, figures, numbers, report fill-in), `ftm/eval/final_synth.py` (synthetic run for dry runs), `configs/analysis/{inputs,systems}.yaml`, tests `tests/test_final_analysis.py`.

The script reads one run directory written by `scripts/run_final.py` (`MANIFEST.json`), attaches labels from the test alert table (the run logs carry none; a `label` column in a log is dropped), and writes every pre-registered number, table and figure into `<run dir>/analysis/`. It never sets `FTM_FINAL_RUN`; `--split test` refuses to start unless the caller set it.

## Dry runs (before any test number exists)

```bash
# 1. nothing but code: a synthetic world of systems with planted skills (no data, no LLM, seconds on a laptop)
python scripts/final_analysis.py --synthetic --outdir outputs/analysis/synthetic

# 2. the rehearsal run of the orchestrator (mock LLM, synthetic alerts), same code path as the real run
python scripts/run_final.py --synthetic 40 --only ours.compose        # writes outputs/final_synthetic/<id>/
python scripts/final_analysis.py --split val --run outputs/final_synthetic/<id>

# 3. validation dry run on real val run logs and T1's val alert table: also writes fixed_on_val.json
python scripts/final_analysis.py --split val --run outputs/final/<val_run_id> --inputs configs/analysis/inputs.yaml
```

The additions (`power.txt` needed-n line, `eps_infinity`, `review_capacity`, `adjusted_estimates`; code `ftm/eval/final_additions.py`, tests `tests/test_final_additions.py`) all come out of any of the three commands above; on real val logs add the sweep file to a copy of `inputs.yaml` (`eps_sweep: {path: data/runs/eps_sweep_hi_medium.json}`). S1 and S2 are listed under `sensitivity:` in `prereg/hypotheses.yaml`, not under `hypotheses:`: the file's tests tie that block to the three tiers and the Holm families (`tests/test_prereg.py`), and these two analyses are neither.

A synthetic result is stamped on top of `index.md` and is never a result. A manifest with `synthetic: true` is analysed like val (its logs say `split=synthetic`); its labels come from the logs themselves, because no alert table exists for it. Most tables of a mock-LLM rehearsal come out `not_run` (no upper limits, no per-alert score tables): that is the point, the reasons are listed.

## The final run

```bash
# the caller sets FTM_FINAL_RUN=1; the script never does. Needs the values fixed on validation.
FTM_FINAL_RUN=1 python scripts/final_analysis.py --split test --run outputs/final/<run_id> \
    --inputs <inputs.yaml with alerts.path etc. filled> --fixed <val analysis>/fixed_on_val.json
# numbers into copies of the report (never in place); unfilled placeholders are listed in unfilled.json
python scripts/final_analysis.py --fill docs/report --numbers outputs/final/<run_id>/analysis/numbers.json --out outputs/final/<run_id>/analysis/report_filled
```

`fixed_on_val.json` (written by every `--split val` run) carries what validation fixes before test: the best simple planner, the equivalence margin (0.5 x |Brier(random) - Brier(best simple)|), the H9 margin in gap-closed units, the best ported external agent, and the hub cap (flagged if the largest cluster share passes 0.2 and none is chosen yet). Test never derives these; without the file `--split test` stops.

## What the manifest may look like

`MANIFEST.json` as written by `ftm/final/runner.py` (`manifest_version` 1): top-level `run_id`, `synthetic`, `freeze_sha256`, `subset_sha256`, `full_test_sha256`, `subset_ids_path`, `full_test_ids_path` (relative to the run dir), `jobs[]` with `job_id, system, variant, alerts (subset|full_test), kind, mode, seeds, machine_class, status, runlog_path, runlog_sha256, n_alerts`.

Checked before anything is computed (the script prints nothing and exits 2 on any of these):
- a job with status done whose run log is missing, or whose sha256 differs from `runlog_sha256`;
- an id file that does not hash to the manifest's `subset_sha256` / `full_test_sha256`; a subset (full test) job whose verdict alerts are not exactly the frozen ids (injection jobs are exempt: they run on their own attack items);
- logs holding rows of another split than `--split`; manifest seeds that differ from the log's seeds; `n_alerts` that differ; two verdict rows for the same (system, variant, seed, alert); a different one-bank `p0` for the same alert in two jobs;
- `--split test` on a variant listed under `never_results` in `systems.yaml` (random answers, mock banks).

**Partial runs** (arms finish one at a time): if only `MANIFEST.partial.json` exists, or jobs have a status other than done (the runner writes `{job_id, status: missing}` rows), the analysis runs on what is there. Missing arms are listed in `arms_status.{md,csv}` and in `index.md`; every table or hypothesis that needs one comes out `not_run` with the reason and the name of the pending job ("pending arm: manifest job react.subset is not done yet"). Holm keeps the slot of a `not_run` member in its family (the family size is fixed by the pre-registration, so one missing arm cannot make the other tests easier).

## What is produced (`<run dir>/analysis/`)

Every table is `.md` + `.csv`; every figure is `figures/<name>.png` + `figures/<name>.csv` (the plotted data; the PNG is skipped with a note when matplotlib is missing or with `--no-figures`). Colors are the colour-blind-safe Okabe-Ito set, one hue per system family, direct labels, light grid, no chart junk.

| File | Content |
|---|---|
| `index.md` | front page: run, commit and hashes, arms present/missing, hypotheses, warnings, produced / not produced with reasons |
| `hypotheses` (+ `_components`) | H1-H4, H9 (primary, one Holm family); H6, H7, H10 (secondary, own Holm family); H5, H8 (exploratory, uncorrected, labelled). Estimate, 95% cluster-bootstrap CI, raw and Holm-adjusted p, verdict with the rule quoted | tab_hypotheses |
| `h4_fdr_audit` | per accepted question asked >= 30 times: realised log-LR in the claimed direction |
| `table_5a` | external agents, native vs ported, on the subset and (top 2-3) on the full test, gap closed end-to-end and questions-only against both upper limits, costs | tab_5a |
| `table_5b` | ours, each ablation, planners, raw sharing, references; arms with no result are listed as `not_run`, not dropped | tab_5b |
| `privacy_curve` + `figures/privacy_curve` | gap closed against eps per alert, with the theoretical ceiling (an oracle moving the log-odds by eps) | fig privacy |
| `figures/pareto`, `figures/external_gap`, `figures/reliability` | gap vs eps / LLM calls / GPU-seconds; per external agent; p0 vs fused score |
| `breakdown_typology`, `breakdown_bank` | ours vs one bank vs best external agent, per group |
| `end_to_end_all_rows`, `n_alerts` | end-to-end over all rows with the intake miss rate beside it; counts per set and tail (reported separately, never pooled) |
| `stop_reasons`, `compute`, `compute_by_machine` | stop reasons per system; LLM calls, tokens, GPU-seconds, wall clock |
| `injection`, `injection_summary` | checker on vs off, attack success per kind |
| `case_studies`, `error_tags` | worst and best alerts with their question chains; error tags in pipeline order |
| `fusion_calibration`, `shuffled_answers`, `table_baselines`, `e18_sweep` | calibration of the fused score; shuffled-answers control; Group A/B detection models one-bank vs see-everything; E18 table (T3's CSV) |
| `power.txt` (+ `status.json` key `power`) | the plan's rule of thumb (0.011 at 4,000 alerts, scaled by 1/sqrt(n)), the detectable difference from this run's own bootstrap standard error, **the number of alerts needed to detect a Brier difference of 0.01** (80% power, two-sided 5%) at that standard error, and the detectable difference at the projected test size (`inputs.yaml power.projected_test_alerts`, default 3,475). `status.json` holds the numbers: `se_paired_difference`, `mde_at_n`, `alerts_needed_for_target`, `mde_at_projected`, `source`. Id `power_statement` |
| `eps_infinity` | Brier gain against eps per alert **including the no-noise row**, from the eps-sweep JSON of `scripts/eps_sweep.py` (`inputs.yaml eps_sweep.path`, e.g. `data/runs/eps_sweep_hi_medium.json`; null = `not_run` with the reason). Under the table, one machine-generated line: "Noise is / is not the limiter", from the no-noise gain against the eps 2 gain (limiter only if the no-noise gain beats the eps 2 gain by more than 0.005; descriptive, no CI, the sweep is a validation file). `status.json` keeps the line under `eps_infinity.reading` |
| `review_capacity` (S1) | precision and recall among the top-K alerts by score (K = capacity fraction x alerts of the comparison, ties by alert order), per system, with paired cluster-bootstrap CIs of the difference to the one-bank score. Capacity = `review_capacity` in `fixed_on_val.json`; a `--split val` run writes the default `{kind: top_fraction, fraction: 0.10, status: placeholder}` when absent (a placeholder to be pre-registered), `--split test` refuses without it. Secondary / sensitivity, outside the Holm families, no verdict |
| `adjusted_estimates` (S2) | for H1 (ours minus one bank) and the three H3 comparisons: the per-alert Brier difference regressed (OLS) on p0 and the number of answered questions of our system; unadjusted and adjusted estimate and CI width side by side, slopes with CIs, in-sample R2. Cluster bootstrap refits the regression on every resample of whole clusters. The adjusted estimate is the fit at the original covariate means, so it equals the unadjusted mean by construction (OLS with an intercept) and only its CI differs. Labelled "sensitivity, unadjusted is primary"; constant covariates (e.g. no step rows) are dropped and noted |
| `numbers.json`, `status.json`, `hypotheses.json` | every number the report placeholders `{{TBD:id}}` ask for; per-output status; full hypothesis records |

`numbers.json` keys are the report placeholder ids (`hypothesis_verdicts`, `gap_closed_gfp`, `tab_5a`, `tab_5b`, `fig_privacy_curve`, `fig_pareto`, `compute_table`, `inj_results`, ...); anything the outline asks for that has no producer is `unmapped`.

Rules built in: gap closed = (Brier(one bank) - Brier(system)) / (Brier(one bank) - Brier(upper)) against both upper limits, a raw Brier difference when the denominator is below 1e-5; both end-to-end (verdict, no verdict scored as p0) and questions-only (the fused log-LRs of the answered questions, no free-text confidence) scores; no-verdict rate per system beside every number; bootstrap resamples attempt+account clusters (hub cap from `fixed_on_val.json`), 10,000 draws, seed fixed in `ftm/eval/bootstrap.py`; the headline set is the bulk of the test days, spans-split alerts are flagged not dropped.

## Inputs that are not run logs (`configs/analysis/inputs.yaml`)

`eps_sweep.path` (the eps = infinity row) and `power.{target_brier_difference, projected_test_alerts}` (power.txt). The alert table (`label`, cluster keys, typology, home bank; build it once from T1's alert parquet with `--build-alert-table`), the one-bank `p0`, per-alert probabilities of the upper limits (GFP and best SOTA with see-everything), DP-FedAvg, Group A/B models, the all-rows table, the E18 summary, the fast-model table, the retrieval-view pattern-edge shares (H6) and the fusion parameters. A missing path makes only what needs it `not_run`.

## Not done / open

- No real test or validation number was produced: all checks ran on synthetic data. The first run on real val logs will show which `configs/analysis/systems.yaml` selectors (marked ASSUMED there: raw-sharing arm, eps variants of the LLM system, retrieval-view variants, injection checker on/off) do not match the names the run logs actually carry; a wrong selector gives a `not_run` with the selector and the names found, not a wrong number.
- Manifest `system`/`variant` are the plan's names (`ours`/`compose`), the run logs carry their own (`ours_compose`/`harness`); roles are matched on the logs' columns, the manifest names are used for the arms table only.
- The `--fill` step covers the placeholders the outline lists that this pipeline can produce; the rest (`abstract`, `heldout_cycles`, `gate_rag`, `three_way`, `ft_planner`, `library_shape`) are listed in `unfilled.json` with their reason.
- `ruff format` is not enforced by CI (only `ruff check`), and the new files follow the repo's long-line style.
