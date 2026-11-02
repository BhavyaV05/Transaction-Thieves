"""Small CPU helpers of scripts/t3_rerun.sh (T3 rerun orchestration, docs/t3_rerun.md). No server, no GPU, validation/train metadata only.

    t3_rerun_tools.py rebalance --in SFT.jsonl --out BALANCED.jsonl --report counts.json [--max-stop-share 0.25]
    t3_rerun_tools.py revision gemma4-12b                 # weights revision from configs/llm/serving.yaml
    t3_rerun_tools.py runlog-system RUN_DIR [RUN_DIR ...] # the `system` value of the run logs (what calibrate_external.py fit --system wants)
    t3_rerun_tools.py s1-eval-args selection.json         # "functions<TAB>tiers" of the System 1 selection (empty functions: no ON arm)
    t3_rerun_tools.py chunks TOTAL CHUNK                  # "offset index" per chunk, one per line
    t3_rerun_tools.py hours [--root outputs/t3_val/eval]  # GPU-hour table per stage from measured seconds (markdown)

Owner: T3 (Akshat).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Concurrency-4 speed-up of wall time over the serial per-alert latency (measured serial: sum of per_alert seconds == wall of the old
# concurrency-1 runs; docs/t3_system1.md section 8 quotes 23 s -> 8-10 s per alert at concurrency 4). An assumption, stated in docs/t3_rerun.md.
SPEEDUP = 2.5
VAL_N = {"fit": 191, "cal": 109, "eval": 160}
TRAIN_N = 1983


def chunks(total: int, chunk: int) -> list[tuple[int, int]]:
    """[(offset, index)] of the chunks run_chunks walks (c000, c001, ...), `chunk` alerts each, the last one shorter."""
    return [(off, i) for i, off in enumerate(range(0, total, chunk))]


def rebalance(rows: list[dict], max_stop_share: float = 0.25) -> tuple[list[dict], dict]:
    """Keep every question example; keep stop examples (kind == "stop") only up to `max_stop_share` of the output, chosen by a
    deterministic hash of (alert_id, step) so the draw is reproducible and spread over alerts. Never upsamples (no duplicate rows)."""
    if not 0 < max_stop_share < 1:
        raise ValueError("max_stop_share must be in (0, 1)")
    stops = [r for r in rows if r.get("kind") == "stop"]
    other = [r for r in rows if r.get("kind") != "stop"]
    cap = math.floor(max_stop_share / (1 - max_stop_share) * len(other))
    key = lambda r: hashlib.sha256(f"{r.get('alert_id')}:{r.get('step')}".encode()).hexdigest()  # noqa: E731
    kept = sorted(stops, key=key)[:cap]
    keep_ids = {id(r) for r in kept}
    out = [r for r in rows if r.get("kind") != "stop" or id(r) in keep_ids]
    rep = {"in": len(rows), "questions_or_other": len(other), "stops_in": len(stops), "stops_kept": len(kept), "out": len(out),
           "stop_share_in": len(stops) / len(rows) if rows else 0.0, "stop_share_out": len(kept) / len(out) if out else 0.0,
           "max_stop_share": max_stop_share, "stop_cap": cap}
    return out, rep


def revision(model: str, serving: Path = ROOT / "configs" / "llm" / "serving.yaml") -> str:
    """The `revision:` of a serving profile (the teacher revision recorded in the distilled set)."""
    import yaml  # noqa: PLC0415

    prof = yaml.safe_load(serving.read_text())
    prof = prof["models"]
    if model not in prof or "revision" not in prof[model]:
        raise KeyError(f"no revision for {model} in {serving}")
    return str(prof[model]["revision"])


def runlog_system(run_dirs: list[Path]) -> str:
    """The single `system` value across the run logs (calibration maps are fitted per run-log system name)."""
    import polars as pl  # noqa: PLC0415

    names: set[str] = set()
    for d in run_dirs:
        f = d / "runlog.parquet" if d.is_dir() else d
        if f.exists():
            names |= set(pl.read_parquet(f, columns=["system"])["system"].unique().to_list())  # time-guard: not transactions
    if len(names) != 1:
        raise ValueError(f"expected exactly one system in the run logs, found {sorted(names)}")
    return names.pop()


def s1_eval_args(selection: dict) -> tuple[str, str]:
    """(comma list of adopted functions, tier flag value) of a system1_report selection.json; tiers 'fast' -> '--system1-tiers fast'."""
    sel = selection["selection"] if "selection" in selection else selection
    return ",".join(sel.get("functions", [])), str(sel.get("tiers", "fast"))


def mean_seconds(spec_dir: Path) -> tuple[float, int]:
    """(mean per-alert seconds, alerts) over the chunks of one spec directory (per_alert.jsonl rows)."""
    s, n = 0.0, 0
    for f in sorted(spec_dir.glob("c[0-9][0-9][0-9]/per_alert.jsonl")):
        for ln in f.read_text().splitlines():
            if ln.strip():
                s += float(json.loads(ln)["seconds"])
                n += 1
    return (s / n if n else float("nan")), n


# spec -> existing run directory used to estimate its seconds (the adapted arms have no run yet: their ported counterpart stands in)
VAL_SPECS = ["ours_compose", "ours_llm_decides", "ours_act_only", "ours_fixed_list", "ours_batched_2_1", "ours_library_account", "ours_no_backtest",
             *[f"{k}_adapted" for k in ("react", "reflexion", "rewoo", "lats", "llmcompiler", "dspy_react", "debate", "uot", "bed_llm", "mediq")],
             "react", "reflexion", "rewoo", "lats", "llmcompiler", "dspy_react", "debate", "uot", "bed_llm",
             "sharma_native", "sharma_ported", "torres_native", "torres_ported", "co_investigator_native", "co_investigator_ported",
             "finfre_rag_native", "finfre_rag_ported"]


def stand_in(spec: str) -> str:
    return spec[: -len("_adapted")] if spec.endswith("_adapted") else spec


def hours(root: Path) -> dict:
    """Per-spec serial and concurrency-4 hours for the 460 val alerts (fit 191 + cal 109 + eval 160) from measured seconds under `root`."""
    n_val = sum(VAL_N.values())
    rows = {}
    for spec in VAL_SPECS:
        sec, n = mean_seconds(root / stand_in(spec))
        if math.isnan(sec):
            sec, n = mean_seconds(root / "ours_compose")
            src = "ours_compose (no run of this spec)"
        else:
            src = stand_in(spec) + (" (ported run stands in)" if spec.endswith("_adapted") else "")
        rows[spec] = {"s_per_alert": sec, "n_measured": n, "serial_h": sec * n_val / 3600, "c4_h": sec * n_val / 3600 / SPEEDUP, "basis": src}
    return rows


def hours_md(root: Path) -> str:
    r = hours(root)
    lines = ["| spec | s/alert (serial) | basis | 460 alerts serial h | at concurrency 4 (/%.1f) h |" % SPEEDUP, "|---|---|---|---|---|"]
    for k, v in r.items():
        lines.append(f"| {k} | {v['s_per_alert']:.1f} | {v['basis']} | {v['serial_h']:.2f} | {v['c4_h']:.2f} |")
    lines.append(f"| **total** | | | {sum(v['serial_h'] for v in r.values()):.1f} | {sum(v['c4_h'] for v in r.values()):.1f} |")
    return "\n".join(lines)


_WIN = re.compile(r"^(\S+) (START|END)\s+(\S+)(?: \((\S+)\))?(?: rc=(\d+))?")


def parse_windows(text: str) -> dict[str, list[dict]]:
    """windows.log -> {step name: [{start, end, hours, rc}]} for the closed windows (a START without an END is skipped)."""
    from datetime import datetime  # noqa: PLC0415

    out: dict[str, list[dict]] = {}
    open_: dict[str, str] = {}
    for ln in text.splitlines():
        m = _WIN.match(ln)
        if not m:
            continue
        ts, kind, name, _sha, rc = m.groups()
        if kind == "START":
            open_[name] = ts
        elif name in open_:
            a, b = datetime.fromisoformat(open_.pop(name)), datetime.fromisoformat(ts)
            out.setdefault(name, []).append({"start": a.isoformat(), "end": b.isoformat(), "hours": (b - a).total_seconds() / 3600, "rc": int(rc or 0)})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rebalance")
    r.add_argument("--in", dest="inp", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--report", type=Path, required=True)
    r.add_argument("--max-stop-share", type=float, default=0.25)
    sub.add_parser("revision").add_argument("model")
    sub.add_parser("runlog-system").add_argument("dirs", type=Path, nargs="+")
    sub.add_parser("s1-eval-args").add_argument("selection", type=Path)
    c = sub.add_parser("chunks")
    c.add_argument("total", type=int)
    c.add_argument("chunk", type=int)
    h = sub.add_parser("hours")
    h.add_argument("--root", type=Path, default=ROOT / "outputs" / "t3_val" / "eval")
    a = ap.parse_args(argv)
    if a.cmd == "rebalance":
        rows = [json.loads(ln) for ln in a.inp.read_text().splitlines() if ln.strip()]
        out, rep = rebalance(rows, a.max_stop_share)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text("".join(json.dumps(x) + "\n" for x in out))
        a.report.write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep))
    elif a.cmd == "revision":
        print(revision(a.model))
    elif a.cmd == "runlog-system":
        print(runlog_system(a.dirs))
    elif a.cmd == "s1-eval-args":
        f, t = s1_eval_args(json.loads(a.selection.read_text()))
        print(f"{f}\t{t}")
    elif a.cmd == "chunks":
        for off, i in chunks(a.total, a.chunk):
            print(off, i)
    else:
        print(hours_md(a.root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
