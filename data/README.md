# data/

Gitignored except this file. Everything here is rebuilt from scripts with fixed seeds:

```
raw/       AMLworld CSVs and *_Patterns.txt from Kaggle (CDLA-Sharing)
parquet/   typed Parquet with stable tx_id
views/     one Parquet per bank (30 largest banks)
alerts/    train / val / test alert sets, plus the frozen 1,000-alert external subset
runs/      run logs (schema: ftm/eval/runlog.py)
```

Schemas, split rule, keys and loading code: **`ftm/data/README.md`** (hand-off #1). Rebuild with `python scripts/prepare_data.py --data hi_small`.
