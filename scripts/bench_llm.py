"""Fixed LLM benchmark that converts laptop hours into reference-hours (tracks.md 1.2 rule 5, docs/hardware.md 4).

Owner: T3 (Akshat).  Every LLM-capable machine runs this against its own server with the SAME checkpoint and
vLLM version, then posts the printed row in docs/hardware.md section 4.

    python scripts/bench_llm.py --machine t3-5070ti                                   # the reference run
    python scripts/bench_llm.py --machine t1-3070ti --reference outputs/bench/t3-5070ti_qwen3-8b.json

What it measures (greedy decoding, `ignore_eos` so every run generates exactly --max-tokens tokens):
  prefill tok/s  = prompt tokens / latency of a 1-token request
  decode tok/s   = (completion tokens - 1) / (full latency - prefill latency)      <- the conversion metric
  batch tok/s    = completion tokens of --concurrency parallel requests / wall time
factor = decode tok/s here / decode tok/s on the reference machine; reference-hours = hours here x factor.
Every request has a different (but fixed) first line so prefix caching cannot hide the prefill.
Needs only the standard library and ftm.llm.client.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.llm.client import ChatClient, HttpChatClient  # noqa: E402

BENCH_VERSION = 1  # bump if the prompt or settings change; results across versions are not comparable
N_ROWS = 110  # ~3,000 prompt tokens with the Qwen3 tokenizer, well under the 6,000 cap of Limits.max_tokens
FORMATS = ["ACH", "Wire", "Cheque", "Credit Card", "Cash", "Bitcoin", "Reinvestment"]


def bench_prompt(run: int, part: str) -> list[dict]:
    """Deterministic planner-like prompt: an aliased transaction trail plus an instruction."""
    rng = random.Random(20261019)  # same rows on every machine and run
    rows = []
    for i in range(N_ROWS):
        rows.append(
            f"{i:03d} | 2022-09-{1 + i // 12:02d} {rng.randrange(24):02d}:{rng.randrange(60):02d} | "
            f"A{rng.randrange(16**7):07x} -> A{rng.randrange(16**7):07x} | "
            f"{rng.uniform(10, 250_000):,.2f} US Dollar | {rng.choice(FORMATS)} | "
            f"bank {rng.randrange(1, 3000)} -> bank {rng.randrange(1, 3000)}"
        )
    user = (
        f"Benchmark run {run}{part}.\n"
        "You are reviewing an alert for possible money laundering. Below is the home bank's view of the "
        "transactions around the alert (aliased accounts, time-ordered).\n\n"
        + "\n".join(rows)
        + "\n\nDescribe, step by step, which accounts and patterns deserve a question to a partner bank and why."
    )
    return [{"role": "system", "content": "You are a careful anti-money-laundering analyst."},
            {"role": "user", "content": user}]  # fmt: skip


def one_run(client: ChatClient, run: int, max_tokens: int) -> dict:
    extra = {"ignore_eos": True}
    pre = client.chat(bench_prompt(run, "a"), max_tokens=1, temperature=0.0, seed=0, extra=extra)
    full = client.chat(bench_prompt(run, "b"), max_tokens=max_tokens, temperature=0.0, seed=0, extra=extra)
    decode_s = max(full.latency_s - pre.latency_s, 1e-6)
    return {
        "prompt_tokens": full.tokens_in,
        "completion_tokens": full.tokens_out,
        "prefill_s": pre.latency_s,
        "full_s": full.latency_s,
        "prefill_tok_s": pre.tokens_in / max(pre.latency_s, 1e-6),
        "decode_tok_s": max(full.tokens_out - 1, 0) / decode_s,
    }


def batch_run(client: ChatClient, concurrency: int, max_tokens: int, offset: int = 1000) -> dict:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        outs = list(pool.map(
            lambda r: client.chat(bench_prompt(offset + r, "c"), max_tokens=max_tokens, temperature=0.0,
                                  seed=0, extra={"ignore_eos": True}),
            range(concurrency),
        ))  # fmt: skip
    wall = time.perf_counter() - t0
    return {"concurrency": concurrency, "wall_s": wall, "batch_tok_s": sum(o.tokens_out for o in outs) / wall}


def summarise(runs: list[dict], batch: dict | None, max_tokens: int) -> dict:
    med = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
    out = {f"median_{k}": v for k, v in med.items()}
    out["short_outputs"] = sum(r["completion_tokens"] < max_tokens for r in runs)  # >0: ignore_eos unsupported
    if batch:
        out.update(batch)
    return out


def factor(result: dict, reference: dict) -> float:
    return result["median_decode_tok_s"] / reference["median_decode_tok_s"]


def markdown_row(res: dict) -> str:
    s = res["summary"]
    f = f"{res['factor']:.3f}" if res.get("factor") is not None else "1.000 (reference)"
    return (f"| {res['machine']} | {res['model']} | {res.get('server_version') or '?'} | "
            f"{s['median_prompt_tokens']:.0f} | {s['median_prefill_tok_s']:.0f} | {s['median_decode_tok_s']:.1f} | "
            f"{s.get('batch_tok_s', float('nan')):.1f} | {f} |")  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine", required=True, help="t1-3070ti | t2-3050ti | t3-5070ti | ...")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=4, help="0 skips the batch test")
    ap.add_argument("--reference", type=Path, default=None, help="JSON written by the reference machine's run")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/bench"))
    a = ap.parse_args(argv)

    client = HttpChatClient.from_env()
    model = client.model
    one_run(client, run=999, max_tokens=16)  # warm-up, not counted
    runs = [one_run(client, r, a.max_tokens) for r in range(a.repeats)]
    batch = batch_run(client, a.concurrency, a.max_tokens) if a.concurrency > 0 else None
    res = {
        "bench_version": BENCH_VERSION,
        "machine": a.machine,
        "model": model,
        "server_version": client.server_version(),
        "settings": {"max_tokens": a.max_tokens, "temperature": 0.0, "seed": 0, "ignore_eos": True,
                     "repeats": a.repeats},  # fmt: skip
        "runs": runs,
        "summary": summarise(runs, batch, a.max_tokens),
        "factor": None,
    }
    if a.reference:
        ref = json.loads(a.reference.read_text())
        if ref.get("bench_version") != BENCH_VERSION or ref.get("model") != model:
            print("WARNING: reference used another bench version or model; the factor is not comparable")
        res["factor"] = factor(res["summary"], ref["summary"])
    if res["summary"]["short_outputs"]:
        print("WARNING: some outputs were shorter than --max-tokens (server ignores ignore_eos?)")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    path = a.out_dir / f"{a.machine}_{model}.json"
    path.write_text(json.dumps(res, indent=2))
    print(f"wrote {path}")
    print("| machine | model | server | prompt tok | prefill tok/s | decode tok/s | batch tok/s | factor |")
    print("|---|---|---|---|---|---|---|---|")
    print(markdown_row(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
