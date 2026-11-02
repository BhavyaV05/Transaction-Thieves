"""S1BENCH Part A runner: score the public benchmarks with a served model through the typed API.  Owner: T3 (Akshat).

The model is any OpenAI-compatible server with logprobs (vLLM): System 1 (adapter name), the untuned Qwen3-4B base, or Gemma 4 12B.  Every decision goes
through `ftm.fastmodel.typed.TypedDecider.decide` (one-token read of the label letters/words); no free generation.  Output (not committed):
    <out>/ext/<tag>/<task>.json     per-item probabilities for every arm, latency, failures
    <out>/s1mb/<tag>/<benchmark>.jsonl   per-judgment probabilities (stored option order), seconds, errors

    python scripts/s1bench_run.py ext  --tag system1 --model fast3-s0 --endpoint http://127.0.0.1:8000/v1 --tasks all --perm --workers 32
    python scripts/s1bench_run.py s1mb --tag system1 --model fast3-s0 --endpoint ... --workers 32
    python scripts/s1bench_run.py latency --tag system1 --model fast3-s0 --endpoint ...      # one request at a time, 200 items per task

`--variant prod|listed` selects the two-stage scorer for tasks above 26 options (doc section 2.4); `--resume` skips tasks/benchmarks already complete.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftm.fastmodel.model import LogprobFastModel  # noqa: E402
from ftm.fastmodel.s1bench import latency_summary  # noqa: E402
from ftm.fastmodel.s1bench_ext import build_task, items_for, load_items, score_item  # noqa: E402
from ftm.fastmodel.s1bench_s1mb import make_decider, run_line  # noqa: E402
from ftm.fastmodel.typed import TypedDecider  # noqa: E402
from ftm.llm.client import HttpChatClient  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs" / "s1bench" / "datasets.yaml").read_text())
PREP = Path(CFG["prepared_root"])
ALL_TASKS = list(CFG["tasks"])


def build_decider(a) -> TypedDecider:
    client = HttpChatClient.from_env(endpoint=a.endpoint, model=a.model, timeout_s=300.0)
    base = TypedDecider(LogprobFastModel(client, temperature=1.0))  # raw (T = 1): S1MB forbids tuning, external tasks fit T offline on calib
    return make_decider(base, a.variant)


def _perm_subset(n: int, cap: int, seed: int = 0) -> set[int]:
    return set(range(n)) if n <= cap else set(random.Random(seed).sample(range(n), cap))


def run_ext_task(decider: TypedDecider, task_name: str, a) -> dict:
    out_dir = a.out / "ext" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task_name}.json"
    if a.resume and path.exists():
        print("skip (done)", task_name)
        return json.loads(path.read_text())
    result: dict = {"task": task_name, "tag": a.tag, "model": a.model, "variant": a.variant, "parts": {}}
    for part in ("calib", "eval"):
        meta, rows = load_items(PREP, task_name, part)
        if a.limit:
            rows = rows[: a.limit]
        task = build_task(meta)
        k_max = max(len(s.labels) for s in task.specs)
        cap = CFG["perm_cap_large"] if k_max > 26 else CFG["perm_cap_unfair"] if task.kind == "multilabel_yesno" else len(rows)
        perm_rows = _perm_subset(len(rows), cap) if part == "eval" and a.perm else set()
        per_spec = items_for(task, rows)
        jobs = [(si, i, it) for si, items in enumerate(per_spec) for i, it in enumerate(items)]

        def one(job, task=task, perm_rows=perm_rows, part=part):
            si, i, it = job
            try:
                return si, score_item(decider, task.specs[si], it, f"{task.name}.{part}.{si}.{i}", with_perm=i in perm_rows)
            except Exception as e:  # server error / context overflow: the item is unscored and counted
                return si, {"id": it.id, "label": it.label, "error": f"{type(e).__name__}: {str(e)[:200]}"}

        t0 = time.perf_counter()
        scored: list[list[dict]] = [[] for _ in task.specs]
        with ThreadPoolExecutor(a.workers) as pool:
            for n, (si, rec) in enumerate(pool.map(one, jobs), 1):
                scored[si].append(rec)
                if n % 1000 == 0:
                    print(f"{task_name}/{part} {n}/{len(jobs)} {n / (time.perf_counter() - t0):.1f} items/s", flush=True)
        result["parts"][part] = {"meta": meta, "scored": scored, "seconds_total": time.perf_counter() - t0,
                                 "n_failed": sum("error" in r for rs in scored for r in rs)}  # fmt: skip
    path.write_text(json.dumps(result))
    print("wrote", path, {p: v["n_failed"] for p, v in result["parts"].items()})
    return result


def cmd_ext(a) -> None:
    decider = build_decider(a)
    for t in ALL_TASKS if a.tasks == ["all"] else a.tasks:
        run_ext_task(decider, t, a)


def cmd_s1mb(a) -> None:
    decider = build_decider(a)
    src = PREP / "s1mb"
    ids = sorted(p.stem for p in src.glob("*.jsonl")) if a.benchmarks == ["all"] else a.benchmarks
    out_dir = a.out / "s1mb" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for bid in ids:
        path = out_dir / f"{bid}.jsonl"
        lines = [json.loads(x) for x in (src / f"{bid}.jsonl").read_text().splitlines()]
        if a.limit:
            lines = lines[: a.limit]
        if a.resume and path.exists() and sum(1 for _ in path.open()) == len(lines):
            continue
        t0 = time.perf_counter()
        with ThreadPoolExecutor(a.workers) as pool:
            recs = list(pool.map(lambda iv, bid=bid: run_line(decider, iv[1], f"{bid}.{iv[0]}", permute=a.perm, max_chars=a.max_chars), enumerate(lines)))
        path.write_text("".join(json.dumps(r) + "\n" for r in recs))
        print(f"{bid}: {len(recs)} judgments, {sum('error' in r for r in recs)} unscored, {time.perf_counter() - t0:.0f}s", flush=True)


def cmd_latency(a) -> None:
    """One request at a time on a fixed sample: the latency of the production path per decision (all passes of a two-stage decision)."""
    decider = build_decider(a)
    res = {}
    for t in ALL_TASKS if a.tasks == ["all"] else a.tasks:
        meta, rows = load_items(PREP, t, "eval")
        task = build_task(meta)
        items = items_for(task, rows)[0][: a.latency_n]
        secs = []
        for i, it in enumerate(items):
            try:
                secs.append(score_item(decider, task.specs[0], it, f"lat.{t}.{i}", with_perm=False)["seconds"])
            except Exception:
                continue
        res[t] = latency_summary(secs)
        print(t, res[t], flush=True)
    d = a.out / "ext" / a.tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "latency.json").write_text(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ext", "s1mb", "latency"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True, help="served model / adapter name")
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "s1bench")
    ap.add_argument("--tasks", nargs="+", default=["all"])
    ap.add_argument("--benchmarks", nargs="+", default=["all"])
    ap.add_argument("--variant", choices=["prod", "listed"], default="prod")
    ap.add_argument("--perm", action="store_true", help="option-order arms")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="first N items per task/benchmark (smoke)")
    ap.add_argument("--latency-n", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=9000)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    {"ext": cmd_ext, "s1mb": cmd_s1mb, "latency": cmd_latency}[a.cmd](a)


if __name__ == "__main__":
    main()
