# ftm/orchestrator: running systems over alerts (C15), the small-decision set, the raw-sharing arm

Owner: T2 (Bhavya). Spec: `tracks.md` 4.8 and 4.9.

| File | What |
|---|---|
| `runner.py` | Sequential orchestrator for the non-LLM arms: build a planner, run it over alerts in fresh harness sessions, score, write run-log rows |
| `decision_set.py`, `scripts/make_decision_set.py` | **Small-decision dataset v1 (hand-off #13, T2 to T3)** |
| `raw_share.py` | **The raw-sharing baseline arm (H2)**: the one place where rows leave a bank |

## Small-decision dataset (hand-off #13)

```
python scripts/make_decision_set.py --data hi_medium --table data/backtest/hi_medium/<ns>/lr_table.jsonl --threshold 0.3 --max-total 50000
python scripts/make_decision_set.py --synthetic 300 --out <dir>          # smoke test, never a result
```

Writes `data/decisions/<ds>/decisions_{train,val}.jsonl` and `manifest.json` (counts, label balance, sha256). Rows are `{decision, context, label, alert_id, split}`, the format of T3's `ftm/fastmodel/decisions.py`; the label sets are in `DECISION_LABELS` and a test pins them. 30 decisions per alert by default (`DEFAULT_MIX`), about 60,000 on HI-Medium train and val, capped by `--max-total`. Not committed (README rule 7): share the files and the manifest through the cloud folder.

| Decision | Label comes from | Contexts |
|---|---|---|
| `worth_asking` | the alert label (the alerted transfer is in a real laundering attempt) | alert summary and what the home bank saw about the counterparty in the last 24 / 48 / 72 h, with and without p0 |
| `typology` | the pattern file's typology of the attempt, `none` for a clean alert; laundering alerts with no typology are skipped | same evidence |
| `rule_applies` | T4's gate rules: yes iff the cited rule permits `ask_bank` in the home jurisdiction for the answer kind | a candidate expert question and a permitting or non-permitting citation (US and SG) |
| `stop` | yes iff no remaining back-tested question has an answer that could move the posterior across the decision threshold (or there is no eps for one) | the answers so far, produced by the simulated banks, with posterior, threshold, eps left |
| `injection_present` | T4's injection generator (`ftm/safety/injection.py`): 4 attack types and benign notes | a partner-bank reply with the note |

Contexts are aggregate text: counts, amounts, public bank ids. No account id, alias, name or date (tested against the gate's own literal detectors); evidence is cut at the alert time (tested); worth_asking and typology contexts are identical when the label is flipped (tested), so a context cannot encode its label. Test-split alerts raise an error.

## Raw-sharing arm (H2)

`raw_share.py`: three banks hand over raw records (the bank holding the alert's partner-side account, then the two banks it dealt with most, for the first account's neighbours), the home bank computes every expert question **exactly** on them plus second-hop features, and a ridge logistic model on those and logit(p0), fitted on training alerts, gives the probability. Unlimited eps, no noise, any feature: what full sharing buys. `gain_ratio(brier_p0, brier_questions, brier_raw)` is H2's ratio.

It is gated twice. Every hand-over asks the compliance gate for `request_records`, which the production gate blocks (`allow_raw_share: false`); only a gate built with `allow_raw_share=True` allows it, and the harness toolbox has no such tool. And only this module defines or calls `share_records` (a test greps the package), so `ftm/banks` still has no code path that returns rows.

## Per-account cap

`ftm/privacy/cap_sweep.py` and `scripts/cap_sweep.py`: how often one partner account is the subject of several alerts, and the refused share of questions per cap. Run it on the real alert sets to settle `per_account_cap` (currently 8.0): `python scripts/cap_sweep.py --data hi_medium --split val`.

## eps sweep (E2)

`eps_sweep.py`, `scripts/eps_sweep.py`: one system at per-alert budgets 0.5, 1, 2, 4 and no noise on the same validation alerts through the same harness. The no-noise point (eps = inf) answers exactly from a pool back-tested the same way; it is an explicit opt-in of the harness (`Harness.allow_exact`): the gate still checks everything but the eps level and nothing is priced. Every row carries the theoretical ceiling (basic composition, unit sensitivity: `B` eps can move the log-odds by at most `B`, the LR by at most `e^B`), the line to draw on the privacy plot.

```
python scripts/eps_sweep.py --system eig --synthetic 360                       # smoke test, never a result
python scripts/eps_sweep.py --system eig --data hi_medium --split val --table-noisy <lr_table.jsonl> --table-exact <lr_table_exact.jsonl>
```
