"""E10 table (fast model + conformal cascade) from `score_fast_baseline.py report` outputs, and the frozen cascade (T3, docs/t3_e10.md).

    python scripts/e10_report.py --arm tuned_s0=outputs/e10/tuned_s0/report.json --arm tuned_s1=... --arm untuned_4b=... \
        --latency outputs/e10/latency.json --frozen tuned_s0 --md docs/t3_e10.md --json outputs/e10/fast.json

Each report.json is what `scripts/score_fast_baseline.py report` writes (temperature on val `fit`, conformal sets of both tiers on `cal`,
every number on `eval`). `latency.json`: {"<arm>": {"ms_per_decision": x}, "large": {"ms_per_decision": y}} measured one request at a time
on the reference machine. Per decision type: fast-tier argmax accuracy, ECE (temperature-scaled), share kept by the fast tier, sent to the
large tier, sent to the safe default, error_marginal = P(accepted and wrong) against 2 alpha, latency and speed-up (large latency over the
cascade's mean latency fast + (1 - kept) * large). The frozen model is named up front (seed 0 of the tuned runs: no choice on `eval`).
`--json` writes the file T4's final analysis reads (`configs/analysis/inputs.yaml` fast_model.path) and the frozen cascade thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def rows_for(name: str, rep: dict, lat: dict) -> list[dict]:
    out = []
    large_ms = (lat.get("large") or {}).get("ms_per_decision")
    fast_ms = (lat.get(name) or {}).get("ms_per_decision")
    acc = rep.get("eval_argmax_accuracy", {})
    for d, r in rep["decisions"].items():
        if not r.get("n_eval"):
            continue
        eff = None if fast_ms is None or large_ms is None else fast_ms + (1 - r["kept_fast"]) * large_ms
        out.append({"arm": name, "decision": d, "n_eval": r["n_eval"], "accuracy_fast": (acc.get(d) or {}).get("fast"),
                    "accuracy_large": (acc.get(d) or {}).get("large"), "ece_eval": r.get("top_label_ece_eval"),
                    "temperature": r["temperature"], "kept_fast": r["kept_fast"], "to_large": r["to_large"], "to_default": r["to_default"],
                    "error_marginal": r["error_marginal"], "bound_2alpha": r["bound_2alpha"], "within_bound": r["within_bound"],
                    "fast_ms": fast_ms, "large_ms": large_ms, "cascade_ms": eff,
                    "speedup_fast_tier": None if not fast_ms or not large_ms else large_ms / fast_ms,
                    "speedup_cascade": None if not eff or not large_ms else large_ms / eff,
                    "fast_full_set": r["fast_full_set"], "always_defer": r["always_defer"]})  # fmt: skip
    return out


def f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="name=report.json")
    ap.add_argument("--latency", type=Path, default=None)
    ap.add_argument("--frozen", required=True, help="the arm whose model and cascade are frozen for the final run")
    ap.add_argument("--md", type=Path, default=None, help="write the generated table here (kept after the <!-- notes --> line)")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    lat = json.loads(a.latency.read_text()) if a.latency and a.latency.exists() else {}
    reps = {n: json.loads(Path(p).read_text()) for n, p in (x.split("=", 1) for x in a.arm)}
    rows = [r for n, rep in reps.items() for r in rows_for(n, rep, lat)]
    head = ("| arm | decision | n eval | acc fast | acc large | ECE | kept fast | to large | to default | error_marginal (<= 2 alpha) | "
            "fast ms | large ms | cascade ms | speed-up fast / cascade |")
    L = [head, "|" + "---|" * 14]
    for r in rows:
        L.append(f"| {r['arm']} | {r['decision']} | {r['n_eval']} | {f(r['accuracy_fast'])} | {f(r['accuracy_large'])} | {f(r['ece_eval'])} | "
                 f"{f(r['kept_fast'])} | {f(r['to_large'])} | {f(r['to_default'])} | {f(r['error_marginal'], 4)} ({f(r['bound_2alpha'], 2)}) "
                 f"{'ok' if r['within_bound'] else 'OVER'} | {f(r['fast_ms'], 1)} | {f(r['large_ms'], 1)} | {f(r['cascade_ms'], 1)} | "
                 f"{f(r['speedup_fast_tier'], 1)} / {f(r['speedup_cascade'], 1)} |")
    table = "\n".join(L)
    fr = reps[a.frozen]
    frozen_rows = [r for r in rows if r["arm"] == a.frozen]
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({
            "frozen_arm": a.frozen, "alpha": fr.get("alpha"), "temperatures": fr.get("temperatures"), "conformal_q": fr.get("conformal_q"),
            "decision_types": [{"name": r["decision"], "alpha": fr.get("alpha"), "error_marginal": r["error_marginal"], "kept": r["kept_fast"],
                                "speedup": r["speedup_cascade"], "speedup_fast_tier": r["speedup_fast_tier"], "brier_delta": None}
                               for r in frozen_rows]}, indent=2) + "\n")
    if a.md:
        mark = "<!-- notes -->"
        old = a.md.read_text() if a.md.exists() else ""
        notes = old.split(mark, 1)[1] if mark in old else ""
        a.md.write_text(f"# T3 E10: fast model and conformal cascade (generated table)\n\nGenerated by `scripts/e10_report.py`; frozen arm: `{a.frozen}`.\n\n"
                        f"{table}\n\n{mark}{notes}")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
