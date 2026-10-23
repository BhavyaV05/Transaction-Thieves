"""Distillation filter: teacher planner traces -> SFT examples for the QLoRA student (C10 fine-tuning, step 2).

Owner: T3 (Akshat).  Spec: tracks.md 5.4 ("keep only teacher questions that (a) passed back-testing and (b) moved
the posterior toward the true label"), plan.md 6.11.

Input: traces.jsonl from teacher runs made with `run_llm_agent.py --keep-prompts` (each step keeps its exact
prompt messages, the raw JSON, the canonical question, eps, outcome and the answer), the alert labels (server
side), the acceptance rule and the fusion model. Output: chat-format examples {"messages": [... , assistant]}
whose assistant turn is the teacher's step re-serialised as compact JSON, tagged with the teacher model and
revision (E16 compares teacher vs student vs the prompted E18 backbone).
  question step kept  iff  it got a reply, its (canonical, eps) passes `accepted`, and
                           |posterior_after - y| < |posterior_before - y|
  stop step kept      iff  keep_stops and the same alert has at least one kept question (so the student learns
                           to stop after useful questions, not to stop at once)
The posteriors are recomputed here from the logged answers with the fusion model the system actually uses
(`ftm.llm.fusion_adapter.FusedLR` over T4's `ftm.fusion`, fitted on val), never the posteriors the teacher run
saw (those came from whatever fusion that run had, by default the NaiveLRFusion placeholder).
Only train-split alerts by default; test alerts are always refused.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Protocol

from ftm.llm.investigation import Fusion, NaiveLRFusion
from ftm.llm.jsonout import JsonOutputError, extract_json
from ftm.types import QuestionLogEntry, Reply


class RealisticAcceptance(Protocol):
    """Whether question (canonical program, eps) passed back-testing in REALISTIC mode (plan.md 6.8): the noisy
    answers a partner bank would really give, labels as the home bank knows them, TRAIN period only. Exact
    back-testing (no noise, true labels) accepts questions that are useless once the noise is added, so it must
    not be used here. T2's `ftm.backtest` produces the table; the adapter to it is pending (PR #24)."""

    def __call__(self, canonical: str, eps: float) -> bool: ...


@dataclass(frozen=True)
class Teacher:
    model: str
    revision: str  # weights revision (HF commit hash) of the served teacher

    def __post_init__(self):
        if not self.model or not self.revision:
            raise ValueError("teacher model and revision are both required (E16 compares teacher vs student)")


def _target(text: str) -> str | None:
    try:
        return json.dumps(extract_json(text), separators=(",", ":"), ensure_ascii=False)
    except JsonOutputError:
        return None


def _answer_log(recs: list[dict]) -> list[QuestionLogEntry]:
    """The question log of one alert up to now, rebuilt from its trace. Only answered questions can move a posterior."""
    log = []
    for r in recs:
        if r.get("outcome") != "reply":
            continue
        e = QuestionLogEntry(r["step"], str(r["bank"]), r["canonical"], float(r["eps"]))
        e.response = Reply(str(r["bank"]), r.get("answer_kind", "bool"), r["answer_value"], float(r["eps"]), 0.0)
        log.append(e)
    return log


def distill_examples(
    traces: Iterable[dict],
    alerts: dict[str, dict],
    accepted: RealisticAcceptance | None = None,
    *,
    fusion: Fusion,
    teacher: Teacher,
    splits: tuple[str, ...] = ("train",),
    keep_stops: bool = True,
) -> tuple[list[dict], dict]:
    """Returns (examples, counts). `alerts`: alert_id -> {"split", "label"}; each trace carries the alert's "p0"."""
    if accepted is None:
        raise ValueError("distillation needs a realistic back-test acceptance rule (RealisticAcceptance), none was "
                         "supplied; T2's adapter (PR #24) is pending, build one from a realistic-mode table")
    if type(fusion) is NaiveLRFusion:  # FusedLR subclasses it for the shared lookup, so exact type
        raise ValueError("posteriors must come from the fusion model the system uses (ftm.fusion, fitted on val), "
                         "not the NaiveLRFusion placeholder")
    if "test" in splits:
        raise ValueError("never distil from test alerts")
    tag = asdict(teacher)
    examples, counts = [], {"steps": 0, "no_prompt": 0, "questions": 0, "kept_questions": 0, "stops": 0, "kept_stops": 0}
    for t in traces:
        a = alerts.get(t["alert_id"])
        if a is None or a.get("label") is None or a["split"] not in splits:
            continue
        if a["split"] == "test":
            raise ValueError("never distil from test alerts")
        if "p0" not in t:
            raise ValueError(f"{t['alert_id']}: trace has no p0 (re-run the teacher with the current run_llm_agent.py)")
        y = float(a["label"])
        kept_q, stops = [], []
        asked: list[dict] = []  # question steps seen so far: the log the fusion sees
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
                  "step": rec["step"], "kind": rec.get("action"), "teacher": tag}  # fmt: skip
            if rec.get("action") == "stop":
                counts["stops"] += 1
                stops.append(ex)
                continue
            counts["questions"] += 1
            if rec.get("canonical") is None:
                continue
            before = fusion.posterior(t["p0"], _answer_log(asked))
            asked.append(rec)
            if rec.get("outcome") != "reply":
                continue
            after = fusion.posterior(t["p0"], _answer_log(asked))
            if accepted(rec["canonical"], rec["eps"]) and abs(after - y) < abs(before - y):
                kept_q.append(ex)
        examples += kept_q
        counts["kept_questions"] += len(kept_q)
        if keep_stops and kept_q:
            examples += stops
            counts["kept_stops"] += len(stops)
    return examples, {**counts, "teacher": tag, "fusion": type(fusion).__name__}
