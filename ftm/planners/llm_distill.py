"""Distillation filter: teacher planner traces -> SFT examples for the QLoRA student (C10 fine-tuning, step 2).

Owner: T3 (Akshat).  Spec: tracks.md 5.4 ("keep only teacher questions that (a) passed back-testing and (b) moved
the posterior toward the true label"), plan.md 6.11.

Input: traces.jsonl from teacher runs made with `run_llm_agent.py --keep-prompts` (each step keeps its exact
prompt messages, the raw JSON, the canonical question, eps, outcome and posterior before/after), the alert labels
(server side) and the back-test table. Output: chat-format examples {"messages": [... , assistant]} whose
assistant turn is the teacher's step re-serialised as compact JSON.
  question step kept  iff  it got a reply, its (canonical, eps) is accepted in back-testing, and
                           |posterior_after - y| < |posterior_before - y|
  stop step kept      iff  keep_stops and the same alert has at least one kept question (so the student learns
                           to stop after useful questions, not to stop at once)
Only train-split alerts by default; test alerts are always refused.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable

from ftm.llm.jsonout import JsonOutputError, extract_json


def _target(text: str) -> str | None:
    try:
        return json.dumps(extract_json(text), separators=(",", ":"), ensure_ascii=False)
    except JsonOutputError:
        return None


def distill_examples(
    traces: Iterable[dict],
    alerts: dict[str, dict],
    accepted: Callable[[str, float], bool],
    *,
    splits: tuple[str, ...] = ("train",),
    keep_stops: bool = True,
) -> tuple[list[dict], dict]:
    """Returns (examples, counts). `alerts`: alert_id -> {"split", "label"}."""
    if "test" in splits:
        raise ValueError("never distil from test alerts")
    examples, counts = [], {"steps": 0, "no_prompt": 0, "questions": 0, "kept_questions": 0, "stops": 0, "kept_stops": 0}
    for t in traces:
        a = alerts.get(t["alert_id"])
        if a is None or a.get("label") is None or a["split"] not in splits:
            continue
        if a["split"] == "test":
            raise ValueError("never distil from test alerts")
        y = float(a["label"])
        kept_q, stops = [], []
        for rec in t["trace"]:
            counts["steps"] += 1
            if "messages" not in rec:
                counts["no_prompt"] += 1
                continue
            if rec.get("error") or "text" not in rec:
                continue
            target = _target(rec["text"])
            if target is None:
                continue
            ex = {"messages": [*rec["messages"], {"role": "assistant", "content": target}], "alert_id": t["alert_id"],
                  "step": rec["step"], "kind": rec.get("action")}  # fmt: skip
            if rec.get("action") == "stop":
                counts["stops"] += 1
                stops.append(ex)
                continue
            counts["questions"] += 1
            before, after = rec.get("posterior_before"), rec.get("posterior_after")
            if (rec.get("outcome") == "reply" and before is not None and after is not None
                    and accepted(rec["canonical"], rec["eps"]) and abs(after - y) < abs(before - y)):  # fmt: skip
                kept_q.append(ex)
        examples += kept_q
        counts["kept_questions"] += len(kept_q)
        if keep_stops and kept_q:
            examples += stops
            counts["kept_stops"] += len(stops)
    return examples, counts
