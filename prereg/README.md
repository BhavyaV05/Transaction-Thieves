# Pre-registration

Owner: T4 (Shreyas). Thresholds for H1-H4, H9 (and H10) are fixed on **validation** and committed here before the test set is touched. The commit hash is sent to the TA before G5 (week 13 freeze). Any change to a primary hypothesis after G3 goes back to the TA.

Holm order for the primary tests: H1 → H2 → H3 → H4 → H9. Secondary hypotheses are corrected separately; exploratory ones (H5, H8) get no significance claims.

Files to add here (week 4 draft, week 13 freeze):

- `hypotheses.yaml`: thresholds and equivalence margin per hypothesis
- `external_list.yaml`: locked list of external systems and modes (native/ported), backbone and caps
- `subset_manifest.txt`: hash of the frozen 1,000-alert external subset
