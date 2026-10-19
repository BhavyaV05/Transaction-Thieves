"""CSV -> Parquet, pattern map, splits, bank views, aliases, manifest (hand-off #1).

Owner: T1 (Abhishek).  Run from the repo root:  python scripts/prepare_data.py --data hi_small
Idempotent: a second run with unchanged inputs is a no-op (--force rebuilds).
Raw CSVs must already be in `raw_dir` of the config (see data/README.md for the Kaggle command).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.data.build import build_dataset  # noqa: E402
from ftm.data.io import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="hi_small", help="config name in configs/data/ or a yaml path")
    ap.add_argument("--force", action="store_true", help="rebuild even if up to date")
    args = ap.parse_args()
    build_dataset(load_config(args.data), root=Path(__file__).resolve().parents[1], force=args.force)


if __name__ == "__main__":
    main()
