"""Build the strategy library from many past alerts (C12, week 5). ReasoningBank-style successes AND failures.

Owner: T3 (Akshat).  Spec: plan.md 6.13, tracks.md 5.5 ("failure notes come only from back-test statistics over
many alerts, never from one noisy answer"; built on train + val, frozen before test).

Input: one record per (past alert, question asked), from planner / teacher runs on train and val joined with the
back-test table (hand-off #9) and the labels, server side:
    alert_id, split, shape (SituationShape.as_dict), receiver (alias, for the E11 account-keyed ablation only),
    question (canonical), family, eps, accepted (back-test), gain (Brier before - Brier after for that alert)
Grouping is by (shape key, question, eps). A group becomes:
    success  if accepted and mean_gain - z*se > 0           ("in situations like this, ask X first")
    failure  if support >= n_min and accepted and mean_gain + z*se < 0   ("X hurt here")
    nothing  otherwise (not enough evidence either way)
A question that was not accepted in back-testing never gets a note: the planner is only offered accepted questions, and
"rejected in back-testing" says nothing about the shape (the note used to be tautological).
Groups with fewer than `n_min` alerts never produce a note, so no single noisy answer can create one (shape keying:
n_min 20). Account keying (E11, the memorisation ablation) files by receiver alias, where a group is a few alerts at
most, so it defaults to n_min 1 and judges by the sign of the mean gain (no standard-error test): with the shape rules
its library came out empty.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

from ftm.dsl.parser import family as dsl_family
from ftm.dsl.parser import parse as dsl_parse
from ftm.library.shape import SituationShape
from ftm.library.store import Keying, Strategy, StrategyLibrary


def _text(kind: str, shape: SituationShape, question: str, n: int, gain: float, accepted: bool) -> str:
    if kind == "success":
        return (f"In cases like this ({shape.describe()}), asking `{question}` helped: mean Brier gain {gain:+.4f} "
                f"over {n} past alerts.")  # fmt: skip
    return f"In cases like this ({shape.describe()}), `{question}` did not help (mean Brier gain {gain:+.4f}; {n} past alerts)."


def build_library(records: list[dict], *, keying: Keying = "shape", n_min: int | None = None, z: float = 2.0) -> StrategyLibrary:
    n_min = n_min if n_min is not None else (20 if keying == "shape" else 1)
    if any(r["split"] == "test" for r in records):
        raise ValueError("the strategy library is built on train + val only (plan.md 6.13); test records found")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        who = SituationShape.from_dict(r["shape"]).key() if keying == "shape" else (r["receiver"],)
        groups[(who, r["question"], float(r["eps"]))].append(r)
    lib = StrategyLibrary(keying)
    for i, ((_, question, eps), rs) in enumerate(sorted(groups.items(), key=lambda kv: str(kv[0]))):
        n = len({r["alert_id"] for r in rs})
        if n < n_min:
            continue
        gains = [float(r["gain"]) for r in rs]
        mean = statistics.fmean(gains)
        se = statistics.stdev(gains) / math.sqrt(len(gains)) if len(gains) > 1 else math.inf
        if keying == "account":  # E11: an account is seen a handful of times at most; the sign of the mean gain is the note
            se = 0.0
        accepted = all(bool(r["accepted"]) for r in rs)
        if accepted and mean - z * se > 0:
            kind = "success"
        elif accepted and mean + z * se < 0:
            kind = "failure"
        else:
            continue
        shape = SituationShape.from_dict(rs[0]["shape"])
        lib.add(Strategy(
            strategy_id=f"S{i:04d}", kind=kind, text=_text(kind, shape, question, n, mean, accepted), question=question,
            family=rs[0].get("family"), eps=eps, support=n, mean_gain=mean, se_gain=se, accepted=accepted,
            shape=shape.as_dict() if keying == "shape" else {}, account=rs[0]["receiver"] if keying == "account" else None,
        ))  # fmt: skip
    return lib


def _family(canonical: str) -> str:
    """Family by the DSL parser (at_least(...) and other wrappers unwrap to the inner family); the bare function
    name only for a program that does not parse."""
    try:
        return dsl_family(dsl_parse(canonical, allow_restricted=True))
    except Exception:  # noqa: BLE001
        return canonical.split("(")[0]


def records_from_runs(steps: list[dict], alerts: dict[str, dict], accepted) -> list[dict]:
    """Library records from run-log step rows (ftm.eval.runlog schema, event == "step").

    alerts:   alert_id -> {"split", "p0", "label", "shape", "receiver"} (server side; labels only from train/val)
    accepted: callable(canonical, eps) -> bool, from the back-test table (hand-off #9)
    gain of a step = Brier(posterior before) - Brier(posterior after) for that alert's label; "before" is the
    previous step's posterior (p0 for the first step). Only answered, allowed questions count.
    """
    out, last = [], {}
    for r in sorted(steps, key=lambda r: (r["run_id"], r["alert_id"], r["step"])):
        a = alerts.get(r["alert_id"])
        if a is None or a["label"] is None or r["posterior"] is None:
            continue
        key = (r["run_id"], r["alert_id"])
        before = last.get(key, a["p0"])
        last[key] = r["posterior"]
        if r["gate"] != "allowed" or r["response_kind"] != "reply":
            continue
        y = float(a["label"])
        out.append({"alert_id": r["alert_id"], "split": a["split"], "shape": a["shape"], "receiver": a["receiver"],
                    "question": r["program_canonical"], "family": _family(r["program_canonical"]),
                    "eps": r["eps"], "accepted": bool(accepted(r["program_canonical"], r["eps"])),
                    "gain": (before - y) ** 2 - (r["posterior"] - y) ** 2})  # fmt: skip
    return out
