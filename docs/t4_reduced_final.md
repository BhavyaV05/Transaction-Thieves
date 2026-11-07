# The reduced final run: a recorded deviation from the pre-registered plan (T4, 7 Nov 2026)

Owner: T4 (Shreyas). Status: **proposed**; the arm list becomes binding when `prereg/FREEZE.json` is committed (test runs once). Nothing here touches the test split.

## Why

`python scripts/run_final.py --plan` estimates about **188 GPU hours on the 5070 Ti** for the full plan (110 jobs). About 77 hours are left before the 10 Nov deadline, and the laptop (3070 Ti) cannot serve Gemma 4 12B. The pre-registered plan (`plan.md` 7.2, 8) cannot finish, so we run a subset of it and say exactly which parts were not run.

Nothing about a hypothesis, a threshold or a statistic is changed by this note except where the "Other deviations" list says so. Arms that are not run are reported as **not run** in every table; no arm is dropped silently, and nothing is replaced by a validation number in a test column.

## What runs

Selection (`scripts/run_final.py --only ...`, sets are the frozen 1,000-alert `subset` and the `full_test` set; estimates from `--plan`, 7.5 s per alert replaced by the validation medians):

| Where | Job ids | Est. hours |
|---|---|---|
| 5070 Ti (S3), LLM | `ours.compose.full_test` 5.25, `ours.compose.subset` 1.75, `ours.no_backtest.subset` 1.75, `ours.act_only.subset` 1.75, `ours.fixed_list.subset` 1.75, `ours.llm_decides.subset` 1.75 (**added by T4, needs T3's OK**: H7 compares compose with llm_decides), `bed_llm.ported.subset` 3.47, `bed_llm.ported.full_test` 10.42, `co_investigator.ported.subset` 6.47, `rewoo_adapted.rewoo_adapted.subset` 3.0, `dspy_react_adapted.dspy_react_adapted.subset` 3.0 | 40.4 |
| 5070 Ti (S3), T1 | `t1.edge_scorer_home` (the planner's evidence view reads its test scores) 0.1, `t1.gfp_see_everything_medium` (the HI-Medium upper limit of H1, H2 and H9; it was only in the optional row 12; CPU, needs >= 10 GB RAM) 1.5 | 1.6 |
| 3070 Ti laptop (S1), T1 | `t1.gfp_one_bank`, `t1.gfp_see_everything`, `t1.intake`, `t1.multi_gnn`, `t1.edge_scorer`, `t1.explainers`, `t1.evidence_views`, then `t1.mega_gnn`, `t1.federated`, `t1.layerwise` (T3's laptop list, PR #132); `t1.gfp_see_everything` there is HI-Small only | about 19.6 |
| CPU (T2) | `raw_share.*` (H2), the non-LLM planners the hypotheses need (`nonllm.random`, `fixed_order`, `eig`, `learned_policy`, and `enumerator` if its table exists) on both sets | about 10 |

LLM jobs on S3 add up to about 40 h against about 77 h available, so there is slack. Candidates to add before the freeze, in the order of what they buy: `ours.eps_1` and `ours.eps_4` (5.25 h each, the privacy curve of E2), `sharma.ported.subset` (1.7 h), `react.ported.subset` (2.8 h). Not decided here.

External-agent selection (G4, `configs/final/top_external.json`, frozen with the other files): **one agent, `bed_llm`**, with `reduced_run: true` and its reason in the file. `plan.md` asks for the top 2 to 3. The rule was fixed on 1 Nov before the reruns: ranked only if the verdict rate is at least 0.95, then by val Brier. That rule favours agents that stay close to p0, and none of the agents beats p0 on validation; the report must say bed_llm "won" under that rule, not that it is a strong baseline.

## What does not run on test (every one listed as **not run**)

- `ours` ablations: `eps_0.5`, `eps_inf`, `library_account`, `batched_2_1`, `student_s0..s2` (E16), `prompted_qwen3_8b`, and the full-test versions of `no_backtest`, `act_only`, `fixed_list`, `llm_decides` (their subset versions run).
- External agents on the subset: `sharma` (native, ported), `react`, `reflexion`, `rewoo` (ported), `lats`, `llmcompiler`, `dspy_react` (ported), `torres`, `finfre_rag`, `co_investigator` (native), `uot`, `debate`, `mediq`, and the adapted arms other than `rewoo_adapted` and `dspy_react_adapted`.
- E15 injection tests, E10 (fast model, `fast_model.*`), E13 batched run, the RAG matrix, System 1: validation only, if at all.
- Group A/B models that are neither on the laptop list nor required: `fraudgt` (see below), `hi_medium_see_everything` (optional row 12).
- Non-LLM ablation arms beyond those listed above.

## What each hypothesis can still say

| Hypothesis | Test status under the reduced run |
|---|---|
| H1 gap closed >= 30% | tested (`ours.compose.full_test`, one bank, GFP see-everything as the upper limit; time-respecting scores only, issue #127) |
| H2 raw-sharing ratio | tested if `raw_share` runs; the ratio is not interpreted when its denominator CI contains 0 |
| H3 compose vs best simple planner, enumerator, fixed list | the fixed-list comparison on the subset; the enumerator comparison needs T2's enumerated table: **not run** if it is missing |
| H4 back-testing helps | tested on the subset (`no_backtest`), plus the FDR audit on compose |
| H9 vs best ported external | tested against `bed_llm` only (subset and full test); the margin follows the rule in `prereg/hypotheses.yaml` |
| H7 formula vs LLM decides | tested on the subset only if `ours.llm_decides.subset` is added (it is in the table above) |
| H10 vs DP-FedAvg | tested with `t1.federated` |
| H6 retrieval | the pattern-edge share half (T1 evidence views); the Brier half is validation only |
| H5, H8 | **not run** on test; validation only |

On validation no arm clearly beats p0 on Brier (p0 0.1896, `ours_compose` 0.1907, per T3, 7 Nov). H1, H2 and H7 are expected not to be met (`validation_note` in the prereg); they are reported as pre-registered negatives.

## Other deviations, all labelled

1. **No TA.** The hash goes to nobody. `freeze_prereg.py --record-no-ta <hash>` records that in `FREEZE.json`; the pre-registration is fixed by the git commit of that file only, with no external timestamp. The report says so.
2. **Equivalence margin** fixed at 0.01 Brier instead of the derived 2.4e-05 (decided 6 Nov, before any freeze).
3. **H9 `margin_gap`** is a rule computed on validation: `0.01 / (Brier(p0 v2) - Brier(GFP see-everything))` on all 806 val eval alerts = 0.01 / 0.0602 = **0.166**, recorded in the prereg as `value_check` and recomputed into `fixed_on_val.json` by `final_analysis.py --split val`. T1's variants (p0 v1 on 806 alerts 0.1425; 160 askable alerts 0.1291 with v1 and 0.1460 with v2) answer other questions: the test subset holds askable and non-askable alerts, and the final run's p0 is v2.
4. **Upper limit.** Only time-respecting see-everything scores may be an upper limit (issue #127). The stored HI-Medium Multi-GNN and MEGA-GNN see-everything scores use future edges of the validation window and are baseline rows only.
5. **FraudGT test scoring ran once before the freeze**, outside `run_final.py` (7 Nov, `docs/t1_final_run.md`, PR #130). Nothing was trained or retuned; its val scores reproduced exactly. T1 confirms in that note that no test label was read for evaluation. **Proposal, needs T1's OK:** `t1.fraudgt` is not re-run through the orchestrator, so the test split is not scored twice. It is a required step, so `MANIFEST.json` stays `MANIFEST.partial.json` with `t1.fraudgt` listed as missing; the report says it was scored outside the orchestrator and cites that file. FraudGT is HI-Small only while the headline data is HI-Medium, so it is a baseline row, not an upper limit.
6. **E18 refresh** (T3, R3) is an addendum; the 26 Oct decision (Gemma 4 12B, thinking off) stays the pre-registered one. **R2**: the ten `*_adapted` arms are exploratory, subset only, outside Table 5a and H9.
7. **One G4 agent** instead of two to three (above).
8. **p0 v2**: the selection tolerance of the recalibration rule was set after eval numbers existed (disclosed in `docs/t1_baselines_val.md`); the map is frozen with a hash.

## Commands

```
python scripts/run_final.py --plan --only <the ids above>          # check the selection and the hours; sets nothing
python scripts/run_final.py --run-id g5 --confirm-final <hash> --machine s3 --only t1.edge_scorer_home t1.gfp_see_everything_medium      # CPU steps first
python scripts/run_final.py --run-id g5 --confirm-final <hash> --machine s3 --only <S3 LLM ids> --endpoint http://localhost:8000/v1
python scripts/run_final.py --run-id g5 --confirm-final <hash> --machine s1 --only t1.gfp_one_bank ...      # T3's laptop list
python scripts/run_final.py --run-id g5 --confirm-final <hash> --machine cpu --only raw_share.* 'nonllm.eig.*' ...
python scripts/run_final.py --run-id g5 --confirm-final <hash> --manifest-only                               # writes MANIFEST.partial.json while arms are missing
```

A rehearsal of the selection on the synthetic world (mock LLM, `--dry-run`) is accepted and writes `MANIFEST.partial.json`; on the T4 laptop its jobs fail only because `outputs/t3_library/frozen/shape_gemma.json` is not there. That file and `outputs/t3_dspy/react_miprov2.json` are hashed by the freeze (`freeze.EXTRA_UNTRACKED`) and so must be present on the machine that runs `freeze_prereg.py`.
