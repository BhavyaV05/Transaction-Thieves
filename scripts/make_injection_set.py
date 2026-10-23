"""Build the injection test set (hand-off #14). Owner: T4 (Shreyas).

    python scripts/make_injection_set.py                     # -> data/safety/injection_set.jsonl (seed 0)
    python scripts/make_injection_set.py --seed 1 --out data/safety/injection_set_seed1.jsonl

400 attacks (4 types x 100) + 400 benign notes, shuffled, one JSON object per line (ftm.safety.injection).
Prints counts per type and the file's sha256, so everyone can check they hold the same set.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.safety import injection as inj  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-type", type=int, default=100)
    ap.add_argument("--n-benign", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/safety/injection_set.jsonl")
    a = ap.parse_args()
    items = inj.generate(a.n_per_type, a.n_benign, a.seed)
    path = inj.write_jsonl(items, a.out)
    print(dict(Counter(it.kind for it in items)))
    print(f"{path}  sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
