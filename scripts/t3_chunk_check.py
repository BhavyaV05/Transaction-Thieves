"""Cheap sanity check of one finished run_llm_agent.py chunk: did the arm really call the model?  Owner: T3 (Akshat).

    python scripts/t3_chunk_check.py <chunk dir> [--min-call-share 0.3] [--max-prompt-too-long 0.2] [--max-no-verdict 0.2]

Reads <dir>/per_alert.jsonl, <dir>/summary.json (the `system` name and whether System 1 was on) and <dir>.log (the console log).  The arm class decides what stops the stage:
  * OURS (`ours_*`, `*_adapted`; our own planner and the agents running inside our system graph): exit 4 when fewer than --min-call-share of the alerts made an LLM call (a silent
    zero-call arm), when PromptTooLong lines exceed --max-prompt-too-long per alert (since PR 118 an over-long prompt is a quiet per-alert failure), or when more than
    --max-no-verdict of the alerts ended without a probability.
  * EXTERNAL agents (native and ported: react, reflexion, lats, dspy_react, sharma_*, ...): their real behaviour is reported, not policed (lats answers about 65 % of the alerts, co_investigator_native 46 %,
    react / reflexion 89-93 %, dspy_react has up to 0.2 PromptTooLong per alert).  Exit 4 only when NO alert made an LLM call or PromptTooLong exceeds 0.8 per alert (nearly every call);
    everything else prints a WARN line and does not stop.
  * System 1 on (summary `system1` set): the fast tier answers part of the questions, so the Gemma call-share rule is skipped (the other rules still apply).
Exit 0 = go on, 4 = stop and look; nothing should be recorded as a result.  Always printed: system, the three numbers, and for a failure which arm, which rule, which value."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXTERNAL_PTL_STOP = 0.8  # PromptTooLong per alert above which an external arm is dead too


def is_ours(system: str) -> bool:
    return system.startswith("ours_") or system.endswith("_adapted") or system == "planner"


def check(chunk: Path, min_call_share: float = 0.3, max_ptl: float = 0.2, max_no_verdict: float = 0.2) -> tuple[bool, str]:
    rows = [json.loads(ln) for ln in (chunk / "per_alert.jsonl").read_text().splitlines() if ln.strip()]
    n = len(rows)
    if n == 0:
        return False, f"{chunk}: FAIL rule=no alerts value=0 (per_alert.jsonl is empty)"
    summary = json.loads((chunk / "summary.json").read_text()) if (chunk / "summary.json").exists() else {}
    system = str(summary.get("system") or "unknown")
    system1 = bool(summary.get("system1"))
    calls = sum(1 for r in rows if (r.get("llm_calls") or 0) > 0) / n
    no_verdict = sum(1 for r in rows if r.get("probability") is None) / n
    log = chunk.with_name(chunk.name + ".log")
    ptl = sum(1 for ln in log.read_text(errors="ignore").splitlines() if ln.startswith("PromptTooLong") and not ln.startswith("PromptTooLong[planner]")) / n if log.exists() else 0.0
    head = f"{chunk} [{system}{', system1' if system1 else ''}]: alerts {n}, with an LLM call {calls:.2f}, PromptTooLong per alert {ptl:.2f}, without verdict {no_verdict:.2f}"
    fails, warns = [], []
    if is_ours(system):
        if not system1 and calls < min_call_share:
            fails.append(f"arm={system} rule=share of alerts with an LLM call value={calls:.2f} minimum={min_call_share}")
        if ptl > max_ptl:
            fails.append(f"arm={system} rule=PromptTooLong per alert value={ptl:.2f} maximum={max_ptl}")
        if no_verdict > max_no_verdict:
            fails.append(f"arm={system} rule=share of alerts without a verdict value={no_verdict:.2f} maximum={max_no_verdict}")
    else:
        if not system1 and calls == 0:
            fails.append(f"arm={system} rule=no alert made an LLM call value=0.00")
        if ptl > EXTERNAL_PTL_STOP:
            fails.append(f"arm={system} rule=PromptTooLong per alert (external arm, nearly every call) value={ptl:.2f} maximum={EXTERNAL_PTL_STOP}")
        if no_verdict > max_no_verdict:
            warns.append(f"WARN arm={system}: {no_verdict:.2f} of the alerts without a verdict (an external agent's own behaviour, reported)")
        if ptl > max_ptl:
            warns.append(f"WARN arm={system}: PromptTooLong {ptl:.2f} per alert (reported)")
        if not system1 and calls < min_call_share:
            warns.append(f"WARN arm={system}: only {calls:.2f} of the alerts made an LLM call")
    out = [head, *warns, *(f"FAIL {f}" for f in fails)]
    return not fails, "\n".join(out)


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
