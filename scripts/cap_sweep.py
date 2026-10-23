"""How often accounts repeat across alerts, and how often a per-account cap would refuse questions.

Owner: T2 (Bhavya).  Answers the open question behind `per_account_cap: 8.0` (configs/privacy/default.yaml).

    python scripts/cap_sweep.py --data hi_small --split val         # real alert sets (needs data/)
    python scripts/cap_sweep.py --synthetic 400                      # smoke test

Prints the repeat statistics and, per cap, the refused share of questions when every alert asks `--questions` questions at
`--eps`. Reads train or val alerts only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.privacy.cap_sweep import repeat_stats, simulate_cap, subject_stream  # noqa: E402


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--synthetic", type=int, default=None)
    ap.add_argument("--caps", type=float, nargs="+", default=[2.0, 4.0, 8.0, 16.0, 32.0])
    ap.add_argument("--questions", type=int, default=2)
    ap.add_argument("--eps", type=float, default=1.0)
    a = ap.parse_args(argv)
    if a.synthetic:
        from ftm.backtest.synth import make_world

        alerts = make_world(a.synthetic, 0).alerts
    else:
        from ftm.backtest.world import load_world

        alerts = load_world(a.data, a.split).alerts
    stream = subject_stream(alerts)
    out = {"repeats": repeat_stats(stream), "caps": simulate_cap(stream, [*a.caps, None], a.questions, a.eps)}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
