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
This is the "realised" filter (E16 v1). It selects on the alert's own noisy outcome and in-sample, so it is kept only
for reproducing v1 (`filter="realised"`). The default is the "expected" filter (finding 6 of the 2026-11 audit):
  question step kept  iff  it got a reply, it was put to the right bank (the one that holds the subject, ftm.llm.routing),
                           its (canonical, eps) passes `accepted`, and its FAMILY has positive mean gain on the other
                           time folds of the train alerts (cross-fitted: the alert's own outcome never decides its own step)
  stop example added  when the rule says stopping is right at a state the teacher faced: the largest shift the accepted,
                           right-bank questions that fit the eps left can add cannot reach the decision threshold from
                           the re-fused posterior (this covers "no applicable accepted question"). The prompt is the teacher's
                           own, the target is its JSON with action stop. One per alert, no label involved.
The posteriors are recomputed here from the logged answers with the fusion model the system actually uses
(`ftm.llm.fusion_adapter.FusedLR` over T4's `ftm.fusion`, fitted on val), never the posteriors the teacher run
saw (those came from whatever fusion that run had, by default the NaiveLRFusion placeholder).
Only train-split alerts by default; test alerts are always refused.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Protocol

from ftm.dsl.parser import family as dsl_family
from ftm.dsl.parser import parse as dsl_parse
from ftm.llm.investigation import Fusion, LRTable, NaiveLRFusion, logit
from ftm.llm.jsonout import JsonOutputError, extract_json
from ftm.llm.routing import NO_BANK, expected_bank
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


def _answer_log(recs: list[dict], summary: dict | None = None, home=None) -> list[QuestionLogEntry]:
    """The question log of one alert up to now, rebuilt from its trace. Only answered questions can move a posterior.
    With `summary` the entries carry the expected bank, so fusion drops answers from the wrong bank."""
    log = []
    for r in recs:
        if r.get("outcome") != "reply":
            continue
        e = QuestionLogEntry(r["step"], str(r["bank"]), r["canonical"], float(r["eps"]))
        if summary is not None:
            e.expected_bank = expected_bank(r["canonical"], summary, home)
        e.response = Reply(str(r["bank"]), r.get("answer_kind", "bool"), r["answer_value"], float(r["eps"]), 0.0)
        log.append(e)
    return log


_ALERT_LINE = re.compile(r"## Alert\n(\{.*\})\n")
_EPS_LEFT = re.compile(r"eps left: ([0-9.]+)")


def prompt_summary(messages: list[dict]) -> dict | None:
    """The alert summary a teacher prompt showed (its "## Alert" section), None if absent."""
    for m in messages:
        mt = _ALERT_LINE.search(m.get("content", "")) if m.get("role") == "user" else None
        if mt:
            return json.loads(mt.group(1))
    return None


def _family(canonical: str) -> str:
    try:
        return dsl_family(dsl_parse(canonical, allow_restricted=True))
    except Exception:  # noqa: BLE001
        return canonical.split("(")[0]


def _reach(table: LRTable, summary: dict, home, eps_left: float) -> float:
    """Upper bound on how far the accepted, right-bank questions that fit `eps_left` can move the log-odds: the best
    answer of each (one entry per question, greedy by shift per eps)."""
    best: dict[str, tuple[float, float]] = {}  # canonical -> (shift, eps)
    for e in table.entries():
        if not e.accepted or e.eps > eps_left + 1e-9 or expected_bank(e.canonical, summary, home) == NO_BANK:
            continue
        shift = max((abs(math.log(v)) for v in e.lr.values() if v > 0), default=0.0)
        if shift > best.get(e.canonical, (0.0, 0.0))[0]:
            best[e.canonical] = (shift, e.eps)
    total, left = 0.0, eps_left
    for shift, eps in sorted(best.values(), key=lambda x: -x[0] / x[1]):
        if eps <= left + 1e-9:
            total, left = total + shift, left - eps
    return total


def _stop_target(text: str) -> str | None:
    """The teacher's own JSON for this state with the action turned into a stop (same keys, nulls elsewhere)."""
    try:
        obj = extract_json(text)
    except JsonOutputError:
        return None
    for k in ("question_id", "program", "bank", "rule_id"):
        if k in obj:
            obj[k] = None
    obj.update(action="stop", eps=0.0)
    if isinstance(obj.get("hypothesis"), dict):
        obj["hypothesis"]["rationale"] = "No listed question can move the probability across the decision threshold."
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _time_folds(alerts: dict[str, dict], ids: Iterable[str], k: int) -> dict[str, int]:
    """Fold index (0..k-1) by time order of the alerts (equal-count contiguous blocks)."""
    ids = sorted(set(ids), key=lambda i: (str(alerts[i].get("timestamp") or ""), i))
    return {i: min(k - 1, j * k // max(len(ids), 1)) for j, i in enumerate(ids)}


def distill_examples(
    traces: Iterable[dict],
    alerts: dict[str, dict],
    accepted: RealisticAcceptance | None = None,
    *,
    fusion: Fusion,
    teacher: Teacher,
    splits: tuple[str, ...] = ("train",),
    keep_stops: bool = True,
    allow_naive: bool = False,
    filter: str = "expected",  # noqa: A002 - "expected" | "realised" (E16 v1)
    table: LRTable | None = None,
    stop_threshold: float = 0.5,
    folds: int = 5,
    min_family_support: int = 20,
) -> tuple[list[dict], dict]:
    """Returns (examples, counts). `alerts`: alert_id -> {"split", "label"} (+ "bank", "timestamp" for the expected
    filter); each trace carries the alert's "p0". `table` (the back-test LRs) and `stop_threshold` are used by the
    expected filter's stop rule only."""
    if filter not in ("expected", "realised"):
        raise ValueError(f"filter must be expected or realised, not {filter!r}")
    if filter == "realised":
        return _distill_realised(traces, alerts, accepted, fusion=fusion, teacher=teacher, splits=splits,
                                 keep_stops=keep_stops, allow_naive=allow_naive)  # fmt: skip
    _check(accepted, fusion, splits, allow_naive)
    if table is None:
        raise ValueError("the expected filter needs the back-test table (for the stop rule's reach)")
    traces = [t for t in traces if (a := alerts.get(t["alert_id"])) is not None and a.get("label") is not None and a["split"] in splits]
    fold = _time_folds(alerts, (t["alert_id"] for t in traces), folds)
    tag = asdict(teacher)
    counts = {"steps": 0, "no_prompt": 0, "questions": 0, "candidates": 0, "kept_questions": 0, "wrong_bank": 0,
              "teacher_stops": 0, "stop_examples": 0, "alerts": len(traces)}  # fmt: skip
    cand: list[dict] = []  # right-bank, accepted, answered question steps
    qsteps: list[dict] = []
    stops: list[dict] = []
    for t in traces:
        a = alerts[t["alert_id"]]
        if "p0" not in t:
            raise ValueError(f"{t['alert_id']}: trace has no p0 (re-run the teacher with the current run_llm_agent.py)")
        y = float(a["label"])
        summary = next((prompt_summary(r["messages"]) for r in t["trace"] if "messages" in r), None)
        home = a.get("bank", (summary or {}).get("home_bank"))
        if summary is None or home is None:
            counts["no_prompt"] += len(t["trace"])
            continue
        asked: list[dict] = []
        stopped = False
        for rec in t["trace"]:
            counts["steps"] += 1
            if "messages" not in rec:
                counts["no_prompt"] += 1
                continue
            ex = None
            tgt = _target(rec["text"]) if rec.get("text") and not rec.get("error") else None
            if tgt is not None:
                ex = {"messages": [*rec["messages"], {"role": "assistant", "content": tgt}], "alert_id": t["alert_id"],
                      "step": rec["step"], "kind": rec.get("action"), "teacher": tag}  # fmt: skip
            log = _answer_log(asked, summary, home)
            p_now = fusion.posterior(t["p0"], log)
            if keep_stops and not stopped and rec.get("text"):  # the rule says stop here: no label involved
                eps_left = float(m.group(1)) if (m := _EPS_LEFT.search(rec["messages"][-1]["content"])) else 2.0
                reach = _reach(table, summary, home, eps_left)
                if abs(logit(p_now) - logit(stop_threshold)) > reach and (st := _stop_target(rec["text"])):
                    stops.append({"messages": [*rec["messages"], {"role": "assistant", "content": st}], "alert_id": t["alert_id"],
                                  "step": rec["step"], "kind": "stop", "teacher": tag, "rule": "cannot_cross"})  # fmt: skip
                    stopped = True
            if ex is None:
                continue
            if rec.get("action") == "stop":
                counts["teacher_stops"] += 1
                continue
            counts["questions"] += 1
            if rec.get("canonical") is None:
                continue
            asked.append(rec)
            if rec.get("outcome") != "reply":
                continue
            after = fusion.posterior(t["p0"], _answer_log(asked, summary, home))
            right = expected_bank(rec["canonical"], summary, home) == str(rec["bank"])
            counts["wrong_bank"] += not right
            row = {"ex": ex, "alert_id": t["alert_id"], "family": _family(rec["canonical"]), "gain": (p_now - y) ** 2 - (after - y) ** 2}
            qsteps.append(row)
            if right and accepted(rec["canonical"], rec["eps"]):
                cand.append(row)
    counts["candidates"] = len(cand)
    by_fold: dict[tuple[str, int], list[float]] = defaultdict(list)  # (family, fold) -> gains
    for r in cand:
        by_fold[(r["family"], fold[r["alert_id"]])].append(r["gain"])
    examples, family_gain = [], {}
    for r in cand:
        held = [g for (fam, f), gs in by_fold.items() if fam == r["family"] and f != fold[r["alert_id"]] for g in gs]
        if len(held) >= min_family_support and sum(held) / len(held) > 0:
            examples.append(r["ex"])
            family_gain[r["family"]] = round(sum(held) / len(held), 6)
    counts["kept_questions"] = len(examples)
    counts["stop_examples"] = len(stops)
    examples += stops
    return examples, {**counts, "filter": "expected", "teacher": tag, "fusion": type(fusion).__name__,
                      "folds": folds, "stop_threshold": stop_threshold, "last_fold_family_gain": family_gain}  # fmt: skip


def _check(accepted, fusion, splits, allow_naive) -> None:
    if accepted is None:
        raise ValueError("distillation needs a realistic back-test acceptance rule (RealisticAcceptance), none was "
                         "supplied; T2's adapter (PR #24) is pending, build one from a realistic-mode table")
    if type(fusion) is NaiveLRFusion and not allow_naive:  # FusedLR subclasses it for the shared lookup, so exact type
        raise ValueError("posteriors must come from the fusion model the system uses (ftm.fusion, fitted on val), "
                         "not the NaiveLRFusion placeholder (allow_naive=True when the system itself runs on it: the final run's \"naive default\" "
                         "fusion, configs/final/jobs.yaml, until the team adopts the fitted one)")
    if "test" in splits:
        raise ValueError("never distil from test alerts")


def _distill_realised(traces, alerts, accepted, *, fusion, teacher, splits, keep_stops, allow_naive):
    """The E16 v1 filter (steps that moved toward the realised label); kept for reproducing v1."""
    _check(accepted, fusion, splits, allow_naive)
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
    return examples, {**counts, "teacher": tag, "fusion": type(fusion).__name__, "filter": "realised"}
