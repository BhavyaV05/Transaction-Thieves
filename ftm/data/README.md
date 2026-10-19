# ftm/data: data hand-off #1 (T1 to everyone)

Owner: T1 (Abhishek). Spec: `plan.md` section 5, `tracks.md` section 3.3 and hand-off #1.
Everything below is produced by one command and is deterministic (no randomness except the alias seed):

```
python scripts/prepare_data.py --data hi_small        # ~5 s on HI-Small, idempotent; --force rebuilds
python scripts/prepare_data.py --data hi_medium       # same code, streaming; see the HI-Medium status below
```

Raw files go in `data/raw/<ds>/` first (Kaggle, CDLA-Sharing):
`python -m kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_Trans.csv -p data/raw/hi_small` (also `HI-Small_accounts.csv`, `HI-Small_Patterns.txt`; needs kaggle >= 1.8 and `~/.kaggle/access_token`).

## Keys and rules

- **tx_id** (int64): the 0-based row index of the transaction in the raw `*_Trans.csv` (header excluded). The CSV is a fixed file, so ids are stable; its sha256 is in the manifest. Do not renumber after filtering.
- **Account key** = `(bank, account)`. Account numbers are only unique within a bank (16 HI-Small account numbers occur in two banks). `node_id` (int64) is the dense id of that key: rank in `(bank, account)` order. Do not read meaning into node_id order; use aliases for anything an LLM sees.
- **Bank ids are integers.** The raw files write the same bank as `010` and `0010`; both parse to 10.
- **Rows are not sorted by timestamp** (they are roughly ordered, with jitter inside a day). Sort when you need order; `tx_id` is the stable tiebreak.
- **day** (int32) = calendar days since the first day in the data, 1-indexed. Timestamps have minute resolution, no time zone.

## Splits

Time-based, set by `split_days` in `configs/data/<ds>.yaml` (1-indexed, inclusive) and stored as `split` in every row.

| | train | val | test |
|---|---|---|---|
| HI-Medium (16 days) | 1-9 | 10-12 | 13-16 |
| HI-Small (18 calendar days) | 1-6 | 7-8 | 9-18 |

**HI-Small rule.** The dataset paper says HI-Small spans 10 days, and 99.98% of the rows are in days 1-10; days 11-18 are a sparse tail (1,108 rows, 560 of them laundering: late legs of long attempts). We split the 10 bulk days in the HI-Medium proportions 9:3:4 (5.6 / 1.9 / 2.5 days, rounded to 6 / 2 / 2) and put the tail in test so time order is kept and nothing is dropped. Note this is not the 60/20/20 split by transaction index that the AMLworld paper and Multi-GNN use (ours is 64/19/17 of rows), so SOTA reproduction on "their" split needs their split, not ours.

**Alerts and attempts.** `attempts.parquet` marks `spans_split`; drop an alert whose laundering attempt spans a split boundary (plan 5.2 step 3). In HI-Small this is 147 of 370 attempts, because attempts last days. The 72-hour evidence-window rule is applied by whoever builds alerts (T1, week 3-4), using `timestamp`.

**Test lock.** The test rows and labels are on disk, but every reader in `ftm.data.io` that can return them calls `ftm.guards.require_final_run()` (needs `FTM_FINAL_RUN=1`, set at G5 only).

## Tables

All under `data/parquet/<ds>/` unless noted. Gitignored; rebuilt by the command above.

**transactions.parquet** (one row per raw transaction)

| column | type | note |
|---|---|---|
| tx_id | int64 | see above |
| timestamp | datetime[us] | |
| day | int32 | 1-indexed |
| from_bank, to_bank | int64 | |
| from_account, to_account | string | raw account number (never show to an LLM) |
| src_id, dst_id | int64 | node_id of sender / receiver |
| amount_received, amount_paid | float64 | |
| receiving_currency, payment_currency | string | |
| payment_format | string | ACH, Bitcoin, Cash, Cheque, Credit Card, Reinvestment, Wire |
| is_laundering | bool | the label |
| split | string | train / val / test |

**accounts.parquet**: `node_id` int64, `bank` int64, `account` string, `entity_id` string, `entity_name` string. Includes accounts that occur in transactions but not in `accounts.csv` (entity fields null; HI-Small has 0 of them).

**patterns.parquet** (laundering transactions that belong to an attempt): `tx_id` int64, `attempt_id` int64, `typology` string, `split` string. Typologies: `fan_in, fan_out, cycle, scatter_gather, gather_scatter, bipartite, stack, random` (the header detail such as "Max 16-degree" is dropped). `attempt_id` is the 0-based ordinal of the block in `*_Patterns.txt`.
Matching rule: a pattern line is matched to a transaction on every column except the label. If several identical transactions exist, lines are assigned in file order to candidates ordered (laundering first, then lowest tx_id), one transaction per line; counts are in the manifest. A transaction is assigned to at most one attempt.
**Not every laundering transaction is in an attempt**: 1,968 of the 5,177 in HI-Small are labelled `is_laundering` but absent from the pattern file, so `patterns.parquet` has no row for them. Do not treat "not in patterns" as "not laundering".

**attempts.parquet**: `attempt_id`, `typology`, `n_lines` (lines in the pattern file), `n_tx` (matched), `first_timestamp`, `last_timestamp`, `splits` (sorted list of splits touched), `spans_split` (bool).

**data/views/<ds>/bank=<id>.parquet**: View(b) = rows of transactions with `from_bank == b` or `to_bank == b`, same columns and order as transactions, all splits. The 30 largest banks by that count. **data/views/<ds>/banks.parquet**: `rank, bank, n_tx, n_laundering` (whole dataset, for planning only).

**aliases/seed=<s>.parquet** (`data/parquet/<ds>/aliases/`): `node_id`, `alias` (like `A00f3a1c`). Seeds in `alias_seeds` of the config (default `[0]`). **Server-side only**: never send this file, or anything that can be joined with it, to an LLM, a partner-bank simulator or a log an LLM reads.

**manifest.json**: raw-file sha256, row counts, date range, split rule and per-split / per-day counts, pattern statistics (unmatched, multi-match, not-in-attempt), typology counts, bank coverage, build key.

## Loading data (use only `ftm.data.io`)

```python
from ftm.data import io, alias

cfg = io.load_config("hi_small")
train = io.scan_transactions(cfg["parquet_dir"], split="train").collect()          # LazyFrame -> DataFrame
seen = io.scan_view(cfg["views_dir"], bank=70, as_of="2022-09-03T12:00:00").collect()
labels = io.scan_patterns(cfg["parquet_dir"], split="val").collect()
# E12: train without cycle attempts
train_hc = io.scan_transactions(cfg["parquet_dir"], split="train", heldout_cycles=cfg["heldout_cycles"])
# for anything an LLM reads
view = alias.apply_aliases(seen, alias.read_aliases(cfg["parquet_dir"], seed=0))   # src_alias, dst_alias; raw accounts dropped
```

## The time guard and the test lock

- Every reader needs `split=` and/or `as_of=`; there is no read-everything call. `as_of` applies `ftm.guards.enforce_as_of` (`timestamp <= as_of`).
- `split="test"` (or a list containing it) raises `TestSplitLocked` unless `FTM_FINAL_RUN=1`. With `as_of` only (no split), test rows are removed unless `FTM_FINAL_RUN=1`, so a late `as_of` cannot reach the test days.
- `tests/test_static_checks.py::test_every_tool_applies_time_guard` fails if any file under `ftm/` outside `ftm/data/` scans or reads Parquet or imports duckdb/pyarrow. **Do not read `data/parquet` or `data/views` directly in tools**; call `ftm.data.io`. Ad-hoc analysis in notebooks may use DuckDB, but never for results.
- Tools must still pass `as_of` per question; the guard is only as good as its callers. Labels in `patterns.parquet` have no `as_of` (they are labels for training and back-testing, not evidence).

## Held-out cycles (E12)

`heldout_cycles: false` in the data config. `ftm.data.splits.drop_cycle_attempts(frame, patterns)` removes the transactions of cycle attempts from the **train** rows only (val and test rows stay, so cycle alerts can be evaluated); the readers apply it when called with `heldout_cycles=True`. Note: only transactions that are in `patterns.parquet` are dropped. Cycle laundering that is not in the pattern file stays in train.

## Memory and speed

Polars lazy scans, `sink_parquet(engine="streaming")` for the big tables and one streaming pass per bank view. Peak RSS and run time for each dataset are in the status lines below. Pattern matching collects only the rows whose key occurs in the pattern file. On a 16 GB machine use `scan_*` and filter before `.collect()`.

## Statistics (from `manifest.json`)

### HI-Small

| | HI-Small |
|---|---|
| Transactions | 5,078,345 |
| Laundering transactions | 5,177 (0.102%) |
| Accounts | 518,581 (0 not in accounts csv) |
| Time range | 2022-09-01 00:00:00 to 2022-09-18 16:18:00 (18 calendar days) |
| Split days | train 1-6, val 7-8, test 9-18 |
| train split | 3,248,921 tx, 2,530 laundering (0.078%) |
| val split | 965,524 tx, 1,036 laundering (0.107%) |
| test split | 863,900 tx, 1,611 laundering (0.186%) |
| Pattern attempts / lines | 370 / 3,209 |
| Unmatched pattern lines | 0 |
| Pattern lines matching more than one tx | 0 |
| Laundering tx not in any attempt | 1,968 |
| Attempts spanning a split boundary | 147 of 370 |
| Top-30 bank views cover | 32.9% of tx, 38.3% of laundering tx |
| Largest bank | bank 70: 452,751 tx |
| Smallest view | bank 1588: 29,269 tx |
| Raw Trans.csv sha256 | b19d39f515523373... |

Typologies:

| typology | attempts | transactions | attempts spanning splits |
|---|---|---|---|
| bipartite | 49 | 263 | 7 |
| cycle | 54 | 287 | 23 |
| fan_in | 40 | 318 | 20 |
| fan_out | 48 | 342 | 17 |
| gather_scatter | 51 | 716 | 24 |
| random | 41 | 191 | 11 |
| scatter_gather | 44 | 626 | 21 |
| stack | 43 | 466 | 24 |

Cross-check against the AMLworld paper (arXiv:2306.16424, Table 4): HI-Small has 5M transactions, 515K accounts (we see 515,088 used, 518,581 listed), 10 days, and a laundering rate of 1 per 981, which is exactly 5,078,345 / 5,177. The paper's "3.6K laundering transactions" for HI-Small does not match its own rate; it is close to our 3,209 pattern lines, so it probably counts pattern-file transactions.

### HI-Medium

(status below)
