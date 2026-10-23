# ftm/backtest: back-testing (C7)

Owner: T2 (Bhavya); statistics reviewed by T4 (Shreyas). Spec: `plan.md` 6.8, `tracks.md` 4.6. This is **hand-off #9 (back-test LR table, T2 to T3/T4)**.

A question may influence a decision only after it has proved itself on past alerts with known labels.

```
python scripts/backtest.py --data hi_small --eps 0.5 1 2 --shard 0/4        # on any machine, shard i of n
python scripts/backtest.py --data hi_small --eps 0.5 1 2 --assemble         # e-BH over the cache; writes lr_table.jsonl
```

Questions: the expert list (both subject slots) and/or a JSONL file of DSL programs (`--questions-file`, T3's hand-off #8). `--exact` runs the no-noise variant on the same alerts. The test split cannot be loaded.

## What one hypothesis (a question at one eps) goes through

1. **Ask.** Every past alert whose partner bank can be asked (`asked_bank`: the bank holding the subject; set questions go to the partner on the other side of the alerted transfer) is put to that bank with `PartnerBank.handle(purpose="cal")`: exact answer, the same noise as a live answer, the same declines and latencies. Cost goes to the accountant's `eps_cal` ledger, per answer.
2. **Cells.** Bool: yes/no. K-ary: the index. Counts: quantile bins of the pooled noisy answers, merged until each bin holds `min_per_bin` answers. Chosen from the answers only, before the labels are looked at.
3. **LR per cell** `(P(cell | launder) + alpha) / (P(cell | clean) + alpha)`, `alpha = 0.01`, clipped to the DP ceiling `lr_ceiling` (`e^eps` for bool and k-ary; `e^(eps*gap/sensitivity)` for counts, `gap` the difference of the launder and clean medians, at least the sensitivity). The answers are already noisy, so this is the LR of the noisy answer the system will actually see.
4. **E-value** (`evalue.py`): label-permutation e-process over the alerts in time order. Under "the answer says nothing about the label", the labels are exchangeable given the answers, so the chance that alert t is a launderer given the earlier labels is exactly `remaining launderers / remaining alerts`. We bet against that with a running per-cell launder rate shrunk toward the null; the product is a martingale with expectation 1 under the null, so it is valid at any stopping time (Ville). The bet can be anything computed from the past: validity does not depend on it being good.
5. **Accept** a batch with **e-BH** at FDR 10% (arbitrary dependence, which is what the same question at several eps needs). **At test time** a brand-new question is back-tested the same way and decided by **e-LOND** (`OnlineAcceptor`), so the 10% bound holds as questions keep arriving. A question with fewer than `min_answers` answers or `min_per_class` of either class is not eligible and never accepted.

Checks (`tests/test_backtest.py`): the e-value has mean at most 1 and `P(E >= 10) <= 0.10` under the null, also at every stopping time; e-BH on 20 random-predicate questions accepts anything in at most 10% of 1,200 repeats (the plan's "false-acceptance <= 10%" check); real signal among nulls is accepted; the engine accepts an informative question and no question in a world with no signal; the real answers of real questions with permuted labels give at most 10% false acceptance.

## The LR table (hand-off #9)

`lr_table.jsonl`, one row per eligible hypothesis, the format of T3's `LRTable.from_rows` (`ftm/llm/investigation.py`):

```json
{"program_canonical": "n_counterparties(ALERT.receiver, dir=in, window=before(72))", "eps": 2.0, "accepted": true,
 "family": "n_counterparties:in", "bins": [1.0, 2.0, 3.0], "lr": {"b0": 0.6, "b1": 0.9, "b2": 1.4, "b3": 6.2}}
```

Answer cells: bool `true`/`false`; counts `b<i>`, `i` = number of `bins` edges at or below the value; k-ary the index as text. `ftm.backtest.lr.answer_key` is a copy of T3's `answer_key` and a test pins the cases. The cache entries (`data/backtest/<ds>/<namespace>/*.json`) also hold `e_value`, `log_e`, `n_answers`, class counts and cell counts. For T4's fusion: `(alert_id, family, log LR)` is `log(lr[answer_key(reply.value, bins)])` of an accepted question.

## Caching and sharding

Keyed by `(namespace, canonical form, eps)`; the namespace hashes the alert set, seed and configuration, so a stale entry is never read. One JSON file per key, written atomically; shards (`--shard i/n`, split by canonical form) write the same directory and `--assemble` runs e-BH over everything in it. Cost on a laptop: about 2 ms per answer on the toy ledgers; real ledgers are larger, and the plan's 90 to 320 CPU-hours is the budget to shard across the four machines.

## Real data and the synthetic world

`world.py` loads the real thing (T1's alert sets and bank views, only through `ftm.data.io`, train or val only; the ledger of a partner is its view of that split, so an answer never sees a later split). `synth.py` builds a small world where laundering alerts behave differently at the partner bank (`signal` scales it, 0 gives all-null questions); it is for tests and smoke runs and never a result. `tests/test_backtest_world.py` runs the whole path, shards and `--assemble` included, on a fabricated dataset in the real file layout.

## Exact mode and the realistic-vs-exact comparison (week 6, G2)

`scripts/backtest.py --exact-labels` is exact mode proper: a label-stratified sample of cross-bank transactions from every bank's view (`--n-pos` laundering ones, `--neg-ratio` clean per laundering one), each treated as an alert at one of its two banks, answered with **no noise**. Because the sample is stratified by label the LRs do not depend on the sampling rates. `--exact` alone is the no-noise run on the same alerts. `--compare REALISTIC_DIR EXACT_DIR` prints, over the questions eligible in both: how many are accepted in each, the agreement, the rank correlation of their strength (max |log LR|), and the median ratio of realistic to exact strength, i.e. what the noise and the harder alert set cost (`ftm/backtest/exact.py::compare_modes`). Entry files are keyed by 20 hex digits, so the report and the LR table written next to them are never read as entries (a bug the comparison found).

## Held-out cycles (week 8, E12)

`scripts/backtest.py --heldout-cycles` back-tests on a train split without any cycle attempt: the cycle rows are dropped by `ftm.data.io(heldout_cycles=True)` (T1) and the cycle alerts by the loader; validation is untouched so the cycle alerts can be evaluated. `ftm/backtest/heldout.py::transfer_report` then scores the accepted questions on validation subsets (e.g. cycle vs the rest) by their **separation**: the mean log-LR the table gives a laundering alert's answer minus a clean one's. `--compare DIR_FULL DIR_HELDOUT` shows whether the accepted set and the strengths change when cycles are not in training.

## Not done

* **G2 on real data** (realistic vs exact LRs for the expert list; at least 20 expert questions accepted at G3): needs T1's `data/` and the alert sets on this machine.
* `n_same_entity` needs `accounts.parquet` (loaded when present).
