"""Reports of the rerun tree (outputs/t3_val_v3/{seed,fit,cal,eval}/<spec>/c###) with the existing analysis code (T3, docs/t3_rerun.md).

    t3_rerun_report.py analysis [--root outputs/t3_val_v3] [--out-dir outputs/t3_val_v3/analysis]   # scripts/t3_val_analysis.py on the new tree
    t3_rerun_report.py e16 [--root ...] [--out ...]                          # scripts/t3_e16_report.py (eval, 160 alerts, once)
    t3_rerun_report.py e16-choose --part fit|cal [--root ...] [--out ...]    # Brier / dBrier vs p0 per E16 arm on fit or cal (choose; no eval)

scripts/t3_val_analysis.py and t3_e16_report.py hard-code outputs/t3_val and the week-2 table; this wrapper points their module paths at the
rerun tree and at configs/fusion/headroom_v1_table.jsonl (the table the rerun used) and calls their own main(). CPU only, validation only.

Owner: T3 (Akshat).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TABLE = ROOT / "configs" / "fusion" / "headroom_v1_table.jsonl"
E16_ARMS = ["e16_prompted", "e16_student_s0", "e16_student_s1", "e16_student_s2"]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def point_at(va, root: Path, part: str = "eval") -> None:
    """Re-point the analysis module's tree: EVAL = <root>/<part>, the tuning runs = the seeding runs <root>/seed/{fit,cal}, the new table."""
    va.EVAL = root / part
    va.FIT = [root / "seed" / "fit", root / "seed" / "cal"]
    va.TABLE = TABLE


def choose(va, root: Path, part: str, n_boot: int) -> dict:
    """Per E16 arm (and the Gemma reference ours_compose) on fit or cal: Brier and paired dBrier vs p0 with the cluster bootstrap CI."""
    from ftm.planners import e18  # noqa: PLC0415

    point_at(va, root, part)
    table, mapping = va.load_table(), e18.load_clusters(va.CLUSTERS)
    res = {}
    for name in [*E16_ARMS, "ours_compose"]:
        d = va.EVAL / name
        if not any(d.glob("c[0-9][0-9][0-9]/summary.json")):
            continue
        al = va.load_run([d], table)
        r = va.diff(al, va.arr(al, "prob"), mapping, n_boot)
        res[name] = {"n": len(al), "brier": r["brier"], "diff_vs_p0": r["diff"], "ci": [r["lo"], r["hi"]]}
    if res:
        base = va.load_run([va.EVAL / next(iter(res))], table)
        res["p0"] = {"n": len(base), "brier": float(np.mean((va.arr(base, "p0") - va.arr(base, "label")) ** 2))}
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["analysis", "e16", "e16-choose"])
    ap.add_argument("--root", type=Path, default=ROOT / "outputs" / "t3_val_v3")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--part", choices=["fit", "cal"], default="fit")
    ap.add_argument("--n-boot", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "analysis":
        va = _load("t3_val_analysis")
        point_at(va, a.root)
        return va.main(["--out-dir", str(a.out_dir or a.root / "analysis"), *(["--n-boot", str(a.n_boot)] if a.n_boot else [])])
    if a.cmd == "e16":
        mod = _load("t3_e16_report")
        point_at(mod.va, a.root)
        return mod.main(["--out", str(a.out or a.root / "analysis" / "e16.json"), *(["--n-boot", str(a.n_boot)] if a.n_boot else [])])
    va = _load("t3_val_analysis")
    res = choose(va, a.root, a.part, a.n_boot or va.N_BOOT)
    out = a.out or a.root / "analysis" / f"e16_{a.part}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
