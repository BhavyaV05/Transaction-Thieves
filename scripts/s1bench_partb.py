"""S1BENCH Part B CLI: open System One models on OUR decisions, selection on fit+cal, one eval report.  Owner: T3 (Akshat).

    # score our val rows with an open model through ITS typed interface (CPU encoders now; decoders need the GPU)
    python scripts/s1bench_partb.py open --model bekko-400m --cap 600 --out outputs/s1bench/partb/bekko-400m/scores.jsonl
    # choose the System 1 candidate from FIT+CAL only (writes selection.json; eval is not read)
    python scripts/s1bench_partb.py select --cand system1 outputs/s1v1/tuned_s0/fast.jsonl --cand qwen3-4b outputs/s1v1/untuned_4b/fast.jsonl \
        --cand bekko-400m outputs/s1bench/partb/bekko-400m/scores.jsonl --out outputs/s1bench/partb/selection.json
    # the eval table, once, with the frozen selection (also fit and cal tables for reference)
    python scripts/s1bench_partb.py eval --cand ... --selection outputs/s1bench/partb/selection.json --out outputs/s1bench/partb/report

System 1's score file is the one `scripts/score_fast_baseline.py score --tier fast` writes (same `i`); open models' files are written by `open`.
Only validation rows are read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftm.fastmodel.decisions import DECISIONS  # noqa: E402
from ftm.fastmodel.s1bench_adapters import TypedQuestion  # noqa: E402
from ftm.fastmodel.s1bench_partb import (  # noqa: E402
    DECISION_NAMES,
    eval_contrast,
    load_rows,
    metrics_table,
    read_scores,
    run_selection,
)

MODELS = yaml.safe_load((ROOT / "configs" / "s1bench" / "models.yaml").read_text())
DEFS = yaml.safe_load((ROOT / "configs" / "s1bench" / "decisions.yaml").read_text())
DEFAULT_DATA = ROOT / "data" / "decisions" / "hi_medium_s1v1" / "decisions_val.jsonl"


def question_for(row: dict) -> TypedQuestion:
    d = DEFS["decisions"][row["decision"]]
    ids = tuple(o["id"] for o in d["options"])
    assert ids == DECISIONS[row["decision"]].labels, (row["decision"], ids)  # the option order IS the label order of the decision
    return TypedQuestion({DEFS["state_key"]: row["context"]}, d["instruction"], tuple((o["id"], o["description"]) for o in d["options"]))


def make_adapter(name: str, adapter_dir: str | None = None):
    m = MODELS["open_models"][name]
    from ftm.fastmodel import s1bench_adapters as ad

    if name == "bekko-400m":
        return ad.BekkoAdapter(m["repo"], m["revision"], "cpu", name, m["params_m"])
    if name == "gliner25-decide":
        return ad.GlinerAdapter(m["repo"], m["revision"], "cpu", name, m["params_m"], adapter_dir)
    if name in ("laya", "laya-typed-decisions"):
        sub = "typed-decisions" if name == "laya-typed-decisions" else None
        return ad.LayaAdapter("convaiinnovations/laya", m["revision"] if name == "laya" else MODELS["open_models"]["laya"]["revision"], sub, name, m["params_m"])
    if name == "jevk5":
        return ad.JevK5Adapter(m["repo"], str(Path(os.environ.get("S1BENCH_CACHE", MODELS["cache_root"])) / "ext" / "jevk5"), "cuda", name, m["params_m"])
    if name == "tev1-4b":
        return ad.Tev1Adapter(m["repo"], m["revision"], "cuda", name, m["params_m"])
    raise SystemExit(f"{name}: no adapter ({MODELS['open_models'].get(name, {}).get('status', 'unknown model')})")


def cmd_open(a) -> None:
    rows = load_rows(a.decisions, a.cap, a.seed)
    rows = [r for r in rows if r["val_part"] in a.parts]
    if a.limit:
        rows = rows[: a.limit]
    done = set()
    if a.out.exists():
        done = {json.loads(x)["i"] for x in a.out.read_text().splitlines() if x.strip()}
    todo = [r for r in rows if r["i"] not in done]
    ad = make_adapter(a.model, str(a.adapter) if a.adapter else None)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with a.out.open("a") as f:
        for n, r in enumerate(todo, 1):
            p, secs = ad.timed(question_for(r))
            f.write(json.dumps({"i": r["i"], "probs": p, "seconds": secs}) + "\n")
            if n % 100 == 0:
                f.flush()
                print(f"{n}/{len(todo)} rows, {n / (time.perf_counter() - t0):.2f} rows/s", flush=True)
    print(json.dumps({"model": a.model, "rows": len(todo), "seconds": round(time.perf_counter() - t0, 1)}))


def load_cands(a):
    out = []
    for name, path in a.cand:
        path = Path(path)
        first = json.loads(next(x for x in path.read_text().splitlines() if x.strip()))
        kind = "probs" if "probs" in first else "logps"
        params = MODELS["ours"]["params_m"].get(name) or MODELS["open_models"].get(name, {}).get("params_m", 0.0)
        lat_file = path.parent / "latency.json"
        lat = json.loads(lat_file.read_text())["ms_per_decision"] / 1000 if kind == "logps" and lat_file.exists() else None
        out.append(read_scores(path, name, kind, float(params), lat, "RTX 5070 Ti, vLLM" if kind == "logps" else "CPU"))
    return out


def cmd_select(a) -> None:
    rows = load_rows(a.decisions, a.cap, a.seed)
    rep = run_selection(load_cands(a), rows, n_boot=a.n_boot)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("selected", "ineligible", "no_candidate_has_skill", "collapsed") if k in rep}, indent=1))


def cmd_eval(a) -> None:
    rows = load_rows(a.decisions, a.cap, a.seed)
    cands = load_cands(a)
    sel = json.loads(a.selection.read_text())
    chosen = next(c for c in cands if c.name == sel["selected"])
    rep = {"selection": sel, "own": metrics_table(cands, rows, ["fit", "cal", "eval"], "own"), "scaled": metrics_table(cands, rows, ["fit", "cal", "eval"], "scaled"),
           "contrast_eval": eval_contrast(chosen, [c for c in cands if c is not chosen], rows, "own", n_boot=a.n_boot)}  # fmt: skip
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.json").write_text(json.dumps(rep, indent=1))
    lines = ["| candidate | decision | part | n | acc | fit-majority acc | Brier | prior Brier | log loss | ECE |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, v in rep["own"].items():
        for d in DECISION_NAMES:
            for part, m in v["decisions"].get(d, {}).items():
                lines.append(f"| {name} | {d} | {part} | {m['n']} | {m['accuracy']:.3f} | {m['fit_majority_accuracy']:.3f} | {m['brier']:.4f} | {m['brier_prior']:.4f} | "
                             f"{m['log_loss']:.4f} | {m['ece']:.4f} |")  # fmt: skip
    (a.out / "report.md").write_text("\n".join(lines) + "\n")
    print("selected:", sel["selected"], "|", len(lines) - 2, "rows ->", a.out)
    _ = np  # numpy kept for downstream notebooks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["open", "select", "eval"])
    ap.add_argument("--decisions", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model")
    ap.add_argument("--adapter", type=Path, default=None, help="LoRA adapter directory of a fine-tuned open model")
    ap.add_argument("--parts", nargs="+", default=["fit", "cal", "eval"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cand", nargs=2, action="append", default=[], metavar=("NAME", "PATH"))
    ap.add_argument("--selection", type=Path)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"open": cmd_open, "select": cmd_select, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
