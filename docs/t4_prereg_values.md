# T4: the `fix_on_val` values of the pre-registration (for the 30 Oct freeze)

Owner: T4 (Shreyas). Written 28 Oct 2026. Source: the validation dry run of `scripts/final_analysis.py` on real HI-Medium val (806 `eval` alerts, 160 of them with a modelled partner bank), T2's non-LLM arms through `scripts/final_non_llm.py --split val`, and `docs/schedule_nov10.md`. These are candidates from validation, not results; nothing here touches test.

Three values in `prereg/hypotheses.yaml` are still `fix_on_val`, and the freeze script (`docs/t4_final_run.md`, 30 Oct checklist) refuses a file that has any.

## 1. `statistics.clusters_fallback.hub_cap`

Rule in the file: use the hub cap only if the largest cluster share exceeds 0.2 on HI-Medium val alerts. Measured: the largest share of the `attempt+accounts` clusters on the 806 `eval` alerts is **0.0149**, far under 0.2. T1 reported that HI-Medium *rows* cluster (largest share 58%), but the unit of analysis is the alert.

Proposal: set `hub_cap: none  # rule not triggered, largest share 0.0149 on val eval alerts (28 Oct)`. The test run re-checks the share and reports it; the rule itself stays in the file.

## 2. `statistics.equivalence_margin.value`

Definition in the file: 0.5 x |Brier(random planner) - Brier(best simple planner)| on the same alerts. On the dry run (806 `eval` alerts, naive fusion, no LLM): random 0.1697, fixed-order 0.1714, EIG 0.1703, learned policy 0.1698. After PR #83 (random is the reference, not a candidate) the best simple planner is the learned policy and the margin is **2.4e-05**.

Problems, found before the freeze:
- The margin is orders of magnitude below what the data resolve: the detectable Brier difference is about 0.0245 on 806 alerts and about 0.05 on the 160 askable ones (`power.txt` of the dry run). At 2.4e-05 every equivalence-type claim (H3 non-inferiority, H10) is a superiority test in practice. T3 reported the same for E18 (0.0018 on 460 alerts).
- The three simple planners are within 0.0006 of each other, so the definition is dominated by noise.

Options (decision for the team before the freeze):
1. Keep the definition and the value as measured. Honest and mechanical; claims that need an equivalence margin will mostly come out "not shown".
2. Replace the definition by a fixed margin chosen on validation, for example half the detectable difference of the evaluation set. Needs the TA's OK, because it changes a pre-registered definition.
3. Keep the definition but fix its value only after the LLM planner and the simple planners have run on val (G4, 30 Oct), on the askable alerts. Same issue, larger value (about 0.0015), still below resolution.

Recommendation: option 1, stated plainly in the report, because the margin is a property of the problem here rather than a choice: with the questions barely moving the Brier, no planner can be shown equivalent to or better than another at these sample sizes.

## 3. `H9.margin_gap`

H9 compares the gap closed by our system with the best ported external agent. `margin_gap` is the equivalence margin expressed in gap-closed units, so it needs the equivalence margin (section 2) and the one-bank-to-upper-limit denominator. The denominator needs the HI-Medium see-everything val files (GFP is done on the 5070 Ti, Multi-GNN and MEGA are queued, `docs/t1_final_run.md`), and the best ported external agent needs Akshat's val runs. **Not computable yet.** It follows mechanically once those exist: `margin_gap = equivalence_margin / (Brier(one bank) - Brier(upper))`, as `final_analysis.py --split val` writes it to `fixed_on_val.json`.

## 4. Other items for the freeze

- `status` of the E18 decision is already `committed` (PR #66). `rule_commit` of the subset manifest is filled (`84c15b7`, PR #68).
- Fusion setting (fixed before G3, the file's own comment): `per_family`, `prior="zero"`, `fit_cv` is the proposal; naive is the pre-registered negative result. Pending the team call (issue and message of 28 Oct).
- Acceptance rule: if T2 adopts time-aware acceptance (issue #84), it has to be written into `prereg/` before the freeze; after that only fixes.
- Additional secondary analyses (review capacity, p0-adjusted estimator, issues #87 and #88) must be listed in `prereg/hypotheses.yaml` before the freeze to count as pre-registered.
