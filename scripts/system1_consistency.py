"""Consistency of the fast model under meaning-preserving perturbations (Jev's "similar answers for similar inputs"), per decision.

Owner: T3 (Akshat).  Spec: docs/t3_system1.md.  Two steps, the GPU step in between is the ordinary scorer:

    python scripts/system1_consistency.py prepare --decisions data/decisions/hi_medium_s1v1/decisions_val.jsonl --out outputs/s1v1/perturbed.jsonl
    python scripts/score_fast_baseline.py --decisions outputs/s1v1/perturbed.jsonl score --tier fast --model fast3-s0 --out outputs/s1v1/tuned_s0/perturbed_fast.jsonl
    python scripts/system1_consistency.py report --perturbed outputs/s1v1/perturbed.jsonl --orig-scores outputs/s1v1/tuned_s0/fast.jsonl \\
        --pert-scores outputs/s1v1/tuned_s0/perturbed_fast.jsonl --temperatures outputs/s1v1/system1_frozen.json --out outputs/s1v1/consistency.json

`prepare` keeps val `eval` rows of every decision and writes up to `--per-row` perturbed twins of each (`orig_i` = the row's index in the
val file, so the original's score is looked up in the ordinary fast score file). Perturbations: ftm/fastmodel/consistency.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftm.fastmodel.calibration import softmax_t  # noqa: E402
from ftm.fastmodel.consistency import consistency_stats, perturb  # noqa: E402
from ftm.fastmodel.decisions import DECISIONS  # noqa: E402


def cmd_prepare(a) -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    import score_fast_baseline as sfb

    rows = [r for r in sfb.load_rows(a.decisions, a.alerts, None, a.seed) if r["val_part"] == "eval"]
    rng = random.Random(a.seed)
    out = []
    for r in rows:
        for text in perturb(r["decision"], r["context"], rng, a.per_row):
            out.append({**{k: r[k] for k in ("alert_id", "split", "decision", "label", "val_part")}, "context": text, "orig_i": r["i"]})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(x) + "\n" for x in out), encoding="utf-8")
    print(json.dumps({"eval_rows": len(rows), "perturbed": len(out)}))
    return 0


def cmd_report(a) -> int:
    pert = [json.loads(x) for x in a.perturbed.read_text().splitlines() if x.strip()]
    orig = {json.loads(x)["i"]: json.loads(x) for x in a.orig_scores.read_text().splitlines() if x.strip()}
    ps = {json.loads(x)["i"]: json.loads(x) for x in a.pert_scores.read_text().splitlines() if x.strip()}
    temps = json.loads(a.temperatures.read_text())["temperatures"] if a.temperatures else {}
    by: dict[str, tuple[list, list]] = {}
    for i, r in enumerate(pert):
        if r["orig_i"] not in orig or i not in ps:
            continue
        d = r["decision"]
        t = temps.get(d, 1.0)
        a_, b_ = by.setdefault(d, ([], []))
        a_.append(softmax_t(np.array(orig[r["orig_i"]]["logps"])[None, :], t)[0])
        b_.append(softmax_t(np.array(ps[i]["logps"])[None, :], t)[0])
    out = {d: {**consistency_stats(np.stack(x), np.stack(y)), "labels": list(DECISIONS[d].labels)} for d, (x, y) in sorted(by.items())}
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--decisions", type=Path, default=ROOT / "data/decisions/hi_medium_s1v1/decisions_val.jsonl")
    p.add_argument("--alerts", type=Path, default=ROOT / "data/alerts/hi_medium_v2/val.parquet")
    p.add_argument("--per-row", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("report")
    r.add_argument("--perturbed", type=Path, required=True)
    r.add_argument("--orig-scores", type=Path, required=True)
    r.add_argument("--pert-scores", type=Path, required=True)
    r.add_argument("--temperatures", type=Path, default=None, help="the frozen System 1 file (temperatures); default T=1")
    r.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    return {"prepare": cmd_prepare, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
