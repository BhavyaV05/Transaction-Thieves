"""Cheap sanity check of one finished run_llm_agent.py chunk: did the arm really call the model?  Owner: T3 (Akshat).

    python scripts/t3_chunk_check.py <chunk dir> [--min-call-share 0.3] [--max-prompt-too-long 0.2] [--max-no-verdict 0.2]

Reads <dir>/per_alert.jsonl, <dir>/summary.json and <dir>.log (the chunk's console log).  Exit 0 = plausible, 4 = stop and look (nothing should be recorded as a result):
  * fewer than --min-call-share of the alerts made at least one LLM call (a silent zero-call arm; the planner stops some alerts on p0 alone, 35 % in the recorded val-fit chunk),
  * more than --max-prompt-too-long PromptTooLong lines per alert in the log (since PR 118 an over-long prompt is a quiet per-alert failure, not a crash),
  * more than --max-no-verdict of the alerts ended without a probability.
Printed: the three numbers, always."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def check(chunk: Path, min_call_share: float = 0.3, max_ptl: float = 0.2, max_no_verdict: float = 0.2) -> tuple[bool, str]:
    rows = [json.loads(ln) for ln in (chunk / "per_alert.jsonl").read_text().splitlines() if ln.strip()]
    n = len(rows)
    if n == 0:
        return False, f"{chunk}: no alerts in per_alert.jsonl"
    calls = sum(1 for r in rows if r.get("llm_calls", 0) > 0) / n
    no_verdict = sum(1 for r in rows if r.get("probability") is None) / n
    log = chunk.with_name(chunk.name + ".log")
    ptl = sum(1 for ln in log.read_text(errors="ignore").splitlines() if ln.startswith("PromptTooLong")) / n if log.exists() else 0.0
    msg = f"{chunk}: alerts {n}, with an LLM call {calls:.2f}, PromptTooLong per alert {ptl:.2f}, without verdict {no_verdict:.2f}"
    ok = calls >= min_call_share and ptl <= max_ptl and no_verdict <= max_no_verdict
    return ok, msg + ("" if ok else f"  -> FAIL (minimum call share {min_call_share}, maximum PromptTooLong {max_ptl}, maximum no-verdict {max_no_verdict})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("chunk", type=Path)
    ap.add_argument("--min-call-share", type=float, default=0.3)
    ap.add_argument("--max-prompt-too-long", type=float, default=0.2)
    ap.add_argument("--max-no-verdict", type=float, default=0.2)
    a = ap.parse_args(argv)
    ok, msg = check(a.chunk, a.min_call_share, a.max_prompt_too_long, a.max_no_verdict)
    print(msg)
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
