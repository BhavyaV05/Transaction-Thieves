"""Why does naive fusion hurt?  Double counting, shift and calibration diagnostics on one dataset's validation alerts.

Owner: T4 (Shreyas).  Spec: plan.md 6.9, 13 (G2).  Run from the repo root.  The method and how to read the tables are in
`ftm/fusion/diagnostics.py` and `docs/t4_fusion_findings.md`.

    # smoke test on the synthetic world (never a result)
    python scripts/fusion_diagnostics.py --world synth
    # the real thing: the val alerts of the eval part, the expert questions back-tested on train
    python scripts/fusion_diagnostics.py --world real --data hi_medium --val-part eval
    # with the back-test cache scripts/backtest.py already wrote (skips loading the train split)
    python scripts/fusion_diagnostics.py --world real --data hi_medium --val-part eval --backtest-dir data/backtest/hi_medium/<namespace>

It runs the same planner as `scripts/g2_check.py` (fixed-order expert questions, eps 0.5 each, budget 2, through the
harness), then writes data/runs/fusion_diagnostics_<world>.json and a .md table next to it, and prints one summary line.
<world> is `synth` or the dataset name (e.g. `hi_medium`), so two datasets do not overwrite each other.
Exit code 0 = done, 2 = not run (data, alert set or cache missing).  The test split is never read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from g2_check import real_inputs  # noqa: E402  (the real world is built exactly as the G2 check builds it)

from ftm.eval import bootstrap as bs  # noqa: E402
from ftm.eval import g2  # noqa: E402
from ftm.eval import replication as rep  # noqa: E402
from ftm.fusion import diagnostics as dg  # noqa: E402

DONE, NOT_RUN = 0, 2


def _clean(x):
    """JSON without NaN: NaN -> null, numpy scalars -> python."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        x = x.item()
    if isinstance(x, float) and x != x:
        return None
    return x


def _train(a):
    """(banks, alerts) of the train world, for --replication-blocks only."""
    from ftm.backtest.world import load_world

    w = load_world(a.data, "train", a.seed)
    return w.banks, w.alerts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", choices=["synth", "real"], default="synth")
    ap.add_argument("--data", default="hi_medium", help="dataset name (real world)")
    ap.add_argument("--val-part", default="eval", choices=["fit", "cal", "eval", "all"], help="val alerts scored (real world)")
    ap.add_argument("--backtest-dir", default=None, help="assembled back-test cache directory (real world)")
    ap.add_argument("--n", type=int, default=360, help="alerts of the synthetic world")
    ap.add_argument("--strata", type=int, default=3, help="strata of p0 for the conditional LR (3 = terciles)")
    ap.add_argument("--budget", type=float, default=g2.BUDGET, help="eps per alert")
    ap.add_argument("--ask-eps", type=float, default=g2.ASK_EPS, help="eps level of each question")
    ap.add_argument("--eps-levels", type=float, nargs="+", default=list(g2.EPS_LEVELS))
    ap.add_argument("--fdr", type=float, default=0.10)
    ap.add_argument("--hub-cap", type=int, default=None, help="accounts in more alerts than this do not link clusters")
    ap.add_argument("--n-boot", type=int, default=bs.N_BOOT, help="bootstrap resamples of the Brier differences")
    ap.add_argument("--n-boot-share", type=int, default=200, help="bootstrap resamples of the explained-by-p0 share")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default data/runs/fusion_diagnostics_<world>.json")
    rep.add_filter_args(ap)
    a = ap.parse_args(argv)

    name = "synth" if a.world == "synth" else a.data
    try:
        if a.world == "synth":
            banks, entries, val = g2.synthetic_inputs(a.n, a.seed, eps_levels=a.eps_levels)
            attempt_of = None
        else:
            banks, entries, val, attempt_of = real_inputs(a)
    except (FileNotFoundError, KeyError) as e:  # the data, the alert set or the cache is not on this machine
        print(f"fusion diagnostics NOT RUN: {type(e).__name__}: {e}", file=sys.stderr)
        return NOT_RUN
    if not val or not entries:
        print(f"fusion diagnostics NOT RUN: {len(val)} validation alerts, {len(entries)} back-test entries", file=sys.stderr)
        return NOT_RUN
    filter_info = None
    if a.replication_filter or a.replication_blocks:
        if a.world != "real" and a.replication_blocks:
            print("fusion diagnostics NOT RUN: --replication-blocks needs the real world (use --replication-filter)",
                  file=sys.stderr)
            return NOT_RUN
        try:
            entries, filter_info = rep.filter_from_args(a, entries, lambda: _train(a), a.fdr)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"fusion diagnostics NOT RUN: {type(e).__name__}: {e}", file=sys.stderr)
            return NOT_RUN
    r = dg.run_diagnostics(banks, entries, val, name, a.budget, a.ask_eps, a.eps_levels, attempt_of, a.hub_cap, a.seed,
                           a.fdr, strata=a.strata, n_boot=a.n_boot, n_boot_share=a.n_boot_share)  # fmt: skip
    if a.world == "real":
        r["data"], r["val_part"] = a.data, a.val_part
    if filter_info is not None:
        r["replication_filter"] = filter_info
    out = Path(a.out or ROOT / "data" / "runs" / f"fusion_diagnostics_{name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_clean(r), indent=1, default=str))
    out.with_suffix(".md").write_text(dg.to_markdown(r))
    print(r["summary"])
    if filter_info is not None:
        print(f"replication filter: {filter_info['n_dropped']} accepted questions dropped "
              f"({filter_info['accepted_before']} -> {filter_info['accepted_after']} accepted, m = {filter_info['m']})")
    return DONE


if __name__ == "__main__":
    sys.exit(main())
