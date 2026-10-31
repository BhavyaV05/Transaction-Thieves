# Pre-registration

Owner: T4 (Shreyas). Thresholds for H1-H4, H9 (and H10) are fixed on **validation** and committed here before the test set is touched. The commit hash is sent to the TA before G5 (week 13 freeze). Any change to a primary hypothesis after G3 goes back to the TA.

Holm order for the primary tests: H1 → H2 → H3 → H4 → H9. Secondary hypotheses are corrected separately; exploratory ones (H5, H8) get no significance claims.

Files (drafts until the G3 commit; frozen at week 13):

| File | What | Status |
|---|---|---|
| `hypotheses.yaml` | Every hypothesis: tier, statistic, decision rule, threshold; bootstrap, clusters, Holm families, equivalence margin, gap-closed rules | draft; values marked `fix_on_val` are set from validation before G3 |
| `e18_decision.yaml` | Backbone-size decision rule (stages, pick rule, thinking arm) | draft; **must be committed before E18 stage 1 runs** (week 3) |
| `external_list.yaml` | External systems, modes, tiers, owners, implementation; backbone and caps; settings fixed before each reimplementation runs | draft; locked at G3 |
| `subset_manifest.txt` | Rule for the frozen 1,000-alert subset now, the sha256 of its IDs at G5 | draft |
| `../configs/final/non_llm.yaml`, `../configs/experiments/ablations_nonllm.yaml` | T2's non-LLM arm settings and the name of their question table; not in this folder, but covered by the freeze hash (`ftm.final.freeze.EXTRA_FROZEN`) | draft; table path named at G4 |

`tests/test_prereg.py` checks that the files parse and agree with the plan (primary family, tiers, experiments).
