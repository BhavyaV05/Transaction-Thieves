# ftm/dsl: the question language (C4), grammar v0

Owner: T2 (Bhavya). Spec: `plan.md` 6.5, `tracks.md` 4.3. This is **hand-off #3 (grammar v0, T2 to T3)**.

A question is a function call with a subject and keyword parameters. The partner bank runs it exactly on its own
ledger and returns one value (then C5 adds noise).

```
n_counterparties(ALERT.receiver, dir=in, window=before(72))
at_least(n_tx(ALERT.receiver, dir=in, window=before(72), max_amount=3000), k=5)
overlap(ALERT.receiver, dir=out, window=after(72), among=ALERT.payers)
```

| File | What |
|---|---|
| `grammar.lark` | The shape only: `name(arg, key=value, ...)`, nested calls, `ALERT.slot`, numbers, lists |
| `spec.py` | **The registry.** Every primitive: answer kind, subject type, parameter types and ranges, sensitivity, family keys, doc, example. The type checker and the LLM reference text are generated from it |
| `parser.py` | `parse`, `check`, `canonical`, `size`, `sensitivity`, `family`, `arity`, `reference_text` |
| `executor.py` | `execute(node, ledger, ctx)`: exact answer on one bank's ledger (Polars) |
| `expert.py` | The expert list: one program per variant of `configs/questions/expert_specs.yaml` |

## For the planner (T3)

```python
from ftm.dsl.parser import check, reference_text

c = check(text)        # never raises
c.ok, c.canonical, c.kind, c.family, c.sensitivity, c.size, c.error
reference_text()       # grammar description for the prompt, generated from the registry
```

`Check` has the same fields as the stand-in's `DslCheck` (`ok, canonical, kind, family, error`) plus `sensitivity` and
`size`, so the adapter that replaces `ProvisionalDsl` is a few lines. `error` is written so the model can fix its
next try (it names the unknown function, the missing or unknown parameters, or the allowed values).
`canonical` is the dedup and back-test cache key (with the ε level). The JSON schema's question enum should list
canonical forms.

## Language

* **Subject.** The first argument. An account slot (`ALERT.sender`, `ALERT.receiver`) for most primitives, a set slot
  (`ALERT.payers`, `ALERT.payees`) for `set_contacts`, a wrapped count question for `at_least`. No other account
  reference exists, so no account id, alias, date or amount of a particular transfer can be written.
* **Windows** (`window=`): `before(h)` = the h hours up to and including the alert time; `after(h)` = the h hours after
  it; `around(h)` = h/2 hours each side. h is 1..336. An `after` window only covers what the partner knows when it
  answers (the executor cuts the ledger at the answer time), so a reply after 12 h sees 12 h of a 24 h window.
* **Answer kinds.** `bool`, `count`, `kary` (an index; `arity()` says how many values), `set_count`.
* **Amounts** are the amount on the subject's side of a transfer (`amount_paid` for money it sent, `amount_received`
  for money it got). `n_tx` takes `min_amount <= amount < max_amount`.
* **Bank holds subject.** A bank only sees an account's transfers on the side whose bank is itself. A bank that does
  not hold the subject answers 0 / false.
* **The alerted transfer** is ordinary data: it falls in `before` and `around` windows when the subject is a party.
* **Canonical form.** Parameters in registry order, defaults omitted, `%g` numbers, bank lists sorted and
  de-duplicated. `parse(canonical(p)) == parse(p)`.
* **Size** (for the enumerator): one per call plus one per parameter that differs from its default.
* **Family**: primitive name plus its family keys (`dir` for the count and set primitives), e.g.
  `n_counterparties:in`; `at_least:<inner family>`.

Primitives (kind, sensitivity): see `PRIMS` in `spec.py` or `reference_text()`. 24 askable ones cover the 28 askable
families of the expert specs (`set_contacts`, the bank-level set question of plan 6.5, is not used by an expert family yet).

## Sensitivity

The change of the exact answer when one transaction is added or removed (the protection unit). Bool, k-ary and the
distinct counts are 1; `n_repeated` is 2 (one transfer can complete a pair); `chain_length` equals its `max_hops`
cap (one added transfer can join two chains), default 3. This is a function of the program, so C5 reads it from
`sensitivity(node)`.

Two places where this differs from T4's draft specs, which say T2's function is authoritative:

* `internal_chain` (`chain_length`): spec says 1, the DSL says `max_hops` (3 by default). Noise on that family is
  correspondingly larger; lower `max_hops` is available if it back-tests badly.
* `entity_accounts` is registered but **not askable** (`restricted`): it reads the customer record, which
  transaction-level DP does not cover. T2 and T4 decide in week 2 whether it may be asked.

## Executing

```python
from ftm.dsl.executor import ExecContext, execute
ctx = ExecContext(bank=..., t0=..., cutoff=..., alert_amount=..., alert_currency=..., home_bank=...,
                  sender=..., receiver=..., payers=frozenset(...), payees=frozenset(...))
value = execute(parse(text), bank_ledger, ctx)       # bool or int, never rows
```

`cutoff` is the partner's answer time; the ledger is cut there before anything runs (time guard). Ledger columns are
those of `transactions.parquet`; the bank's view is `ftm.data.views`. Only `n_same_entity` needs `entity_of`.

## Tests

`tests/test_dsl.py` (parse, errors, canonical, expert list vs specs), `tests/test_dsl_executor.py` (hand-computed
cases; every expert program plus 10 more compared with a naive pure-Python implementation on 50 random toy graphs).

## Not done yet

* Partner-bank class (C3), noise (C5) and the accountant: week 2.
* The expert list has 55 programs (the specs have 55 askable variants), not ~60 yet; Shreyas and I add variants as
  back-testing shows which families carry signal.
* Enumerator over the registry (week 4): `size()` and the parameter ranges in `spec.py` are what it needs.
