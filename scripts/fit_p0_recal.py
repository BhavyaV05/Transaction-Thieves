"""Fit the recalibrated one-bank baseline (issue 108): logit p' = a + b logit p on labelled validation alerts.

Owner: T4 (Shreyas). CPU only, reads data/alerts/<ds>/val.parquet (labels of validation alerts only; never test).
The maps are fitted on parts of validation that are scored by nothing: a validation analysis uses parts before `eval`, the test analysis uses all of validation.

    python scripts/fit_p0_recal.py --data hi_medium --map cal=cal --map fit_cal=fit,cal --map shift_cal=cal:intercept --primary cal --out outputs/analysis/inputs/p0_recal_val.json
    python scripts/fit_p0_recal.py --data hi_medium --map all_val=fit,cal,eval --primary all_val --out outputs/analysis/inputs/p0_recal_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftm.eval.final_additions import fit_logit_recal  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="hi_medium")
    ap.add_argument("--alerts", help="alert parquet (default data/alerts/<data>/val.parquet)")
    ap.add_argument("--map", action="append", required=True, help="name=part[,part...][:platt|:intercept] with parts among fit, cal, eval")
    ap.add_argument("--mode", choices=["platt", "intercept"], default="platt", help="default mode of a map without a :mode suffix")
    ap.add_argument("--primary", help="the map the analysis compares systems against (default: the first)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    path = Path(a.alerts) if a.alerts else ROOT / "data" / "alerts" / a.data / "val.parquet"
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 2
    alerts = pl.read_parquet(path).filter(pl.col("p0").is_not_null())
    print("label rate by val part:", {r["val_part"]: round(r["rate"], 4) for r in alerts.group_by("val_part").agg(pl.col("label").cast(pl.Float64).mean().alias("rate")).iter_rows(named=True)})
    maps = {}
    for spec in a.map:
        name, _, parts = spec.partition("=")
        parts, _, mode = parts.partition(":")
        mode = mode or a.mode
        sel = [p for p in parts.split(",") if p]
        d = alerts.filter(pl.col("val_part").is_in(sel))
        if d.is_empty():
            print(f"error: no alerts in parts {sel}", file=sys.stderr)
            return 2
        p, y = d["p0"].to_numpy(), d["label"].cast(pl.Float64).to_numpy()
        try:
            ai, bi = fit_logit_recal(p, y, mode)
        except ValueError as e:
            print(f"error: map {name}: {e}", file=sys.stderr)
            return 2
        maps[name] = {"a": ai, "b": bi, "mode": mode, "parts": ",".join(sel), "n_fit": int(len(y)), "base_rate_fit": float(y.mean()), "mean_p0_fit": float(p.mean())}
        print(f"map {name}: parts {sel}, n {len(y)}, label rate {y.mean():.3f}, mean p0 {p.mean():.3f}, a {ai:+.3f}, b {bi:.3f} ({mode})")
    out = {"data": a.data, "alerts": str(path), "primary": a.primary or next(iter(maps)), "maps": maps}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
