# ftm/banks and ftm/privacy: partner banks (C3), DP mechanisms and the accountant (C5)

Owner: T2 (Bhavya). Spec: `plan.md` 6.4 and 6.6, `tracks.md` 4.4 and 4.5. This is **hand-off #4 (executor + `PartnerBank` + accountant)**.

```python
from ftm.banks.binding import bind_alert
from ftm.banks.partner import PartnerBank, Question
from ftm.privacy.accountant import PrivacyAccountant, charge_keys

binding = bind_alert(alert_id, tx_row, home_bank, home_view)      # real node ids; simulator-internal, never in a prompt
bank = PartnerBank(bank_id, ledger=bank_view_frame, seed=0)
reply = bank.handle(Question(program, eps, binding), as_of)       # Reply | Decline | Timeout
```

## PartnerBank

* **Exact answer** by the DSL executor on the bank's own ledger, cut at the answer time `as_of + latency`.
* **Noise** by the mechanism for the answer kind (`ftm/privacy/mechanisms.py`) with the question's eps and the program's sensitivity.
* **Latency** (days, config `configs/banks/default.yaml`): `p_tail` (0.15) of replies take 1 to 4 days, the rest 0.02 to 1 day. `p_decline` (0.05) declines; `p_timeout` (0.02) never replies (`Timeout` after 5 days). `respond()` also returns the wait; `ask(env, question, start)` is the SimPy process.
* **Note**: with probability `p_note` a neutral text. It is drawn independently of the true answer (tested: two banks with different ledgers and the same seed give identical notes), so it carries no information. `note_fn=` swaps in T4's injection text.
* **Reproducible**: every draw is seeded from (seed, bank, alert, canonical question, eps, purpose). A reply does not depend on what else was asked, shards can run on any machine, and asking the same question again returns the same reply (no averaging away the noise by re-asking; the accountant still charges every ask).
* **Invalid or unaskable** programs are declined, never run.
* **No rows leave**: `handle` returns only `Reply | Decline | Timeout` with a `bool` or `int` value. `tests/test_banks.py` checks the types, the public surface, and scans the package for row-returning calls.

The ledger given to a bank must already be guarded (`ftm.data.io.scan_view(..., split=..., as_of=...)`); nothing in `ftm/banks` reads Parquet.

## Mechanisms (`ftm/privacy/mechanisms.py`)

| Kind | Mechanism | Odds-shift ceiling |
|---|---|---|
| bool | randomised response, keep with `e^eps/(1+e^eps)` | `e^eps` |
| k-ary | k-ary randomised response | `e^eps` |
| count | two-sided geometric, `alpha = e^(-eps/sensitivity)` | `e^(eps*diff/sensitivity)` for true answers `diff` apart |
| set count | same noise as count (simulated DP-PSI cardinality, no cryptography) | as count |

`eps = inf` is the "no noise" arm. `likelihood` / `log_lr` give the exact probabilities, so back-testing (C7) can use the true mechanism. `lr_ceiling` is the line for every privacy plot. Tests: 10k-draw chi-square and Kolmogorov-Smirnov (DKW bound) against the exact pmf, neighbouring answers never exceed `e^eps`, the empirical likelihood ratio stays under the ceiling.

## Accountant (`ftm/privacy/accountant.py`)

Basic composition (summed eps), three limits:

* **per alert**: `eps_per_alert` (2.0), across every bank asked for that alert;
* **per partner account**: `per_account_cap` across all alerts and all asking banks for the whole test period. A question is charged to every partner account whose transactions it can see (`charge_keys`): the subject for an account question, **every member** of the set for `set_contacts`. Shipped value **8.0** (four full-budget alerts per account) is a placeholder until T1's test-alert projection shows how often accounts repeat; E14 sweeps it;
* **calibration**: back-testing spends from a separate ledger (`spend_cal`, limit `eps_cal`, default tracked but uncapped) and is reported on its own.

`report()` gives the totals; `to_dict`/`from_dict` carry state between runs. `NullAccountant` is for the no-noise arm.

**For T4 (gate):** `PrivacyAccountant` satisfies the gate's `Accountant` protocol (`can_spend(alert_id, bank, eps)`, `spend(...)`), and `ftm.dsl.parser.gate_validator` is the gate's `Validator`. The gate does not pass the program, so through the gate alone only the per-alert limit is checked. The harness (week 3, mine) calls `can_spend(..., accounts=charge_keys(...))` after the gate allows, which is where the per-account cap bites. No change to `gate.py` needed.
