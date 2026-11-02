"""S1MB (System One Mosaic Benchmark, English v1): official scorer re-implemented with numpy, and System 1's rendering of its questions.

Owner: T3 (Akshat).  Spec: docs/t3_s1bench.md sections 2.2 and 7.

The scorer follows `evaluator/SCORING.md` and `evaluator/src/s1mb/{metrics,diagnostics}.py` of hotchpotch/S1MB (MIT; commit pinned in
configs/s1bench/datasets.yaml).  It is *validated* by recomputing the stored metrics of published result files from their stored
predictions (scripts/s1bench_validate_scorer.py), and by unit fixtures, before any number of ours is trusted.

A *line* (prepared by scripts/s1bench_prepare.py) is one decision of one case:
    case_id, question_id, task in {choice, noul, score}, state, system_prompt, instructions, options [{id, description, value}],
    target {option_id: probability}
`render` receives the inference fields only; the target never reaches a model (tested).  Benchmarks are (subset, task) pairs and their
ids are those of the official result files (e.g. "banking77-choice-test-v1").
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from statistics import median

import numpy as np

from ftm.fastmodel.s1bench import (
    PERM_SEEDS,
    ListedTwoStage,
    ece_from,
    make_decision,
    permutations,
    realign,
    registered,
)
from ftm.fastmodel.typed import TypedDecider

EPS = 1e-12
INFERENCE_FIELDS = ("case_id", "question_id", "task", "state", "system_prompt", "instructions", "options")


# ------------------------------------------------------------------------------------------------------------- scoring
def _t(x) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def _vec(opts: Sequence[dict], probs: dict) -> np.ndarray:
    return np.array([float(probs[o["id"]]) for o in opts])


def score_benchmark(task: str, lines: Sequence[dict], preds: dict[str, dict | None]) -> dict:
    """Official metrics of one benchmark.  `preds`: "case_id|question_id" -> {option_id: p} (None / missing = unanswered).
    The dataset-wide baselines use ALL lines; the primary metric and the diagnostics use the answered ones (as the official code)."""
    entries = [(ln["options"], ln["target"]) for ln in lines]
    rows = []
    for ln in lines:
        p = preds.get(f"{ln['case_id']}|{ln['question_id']}")
        if p is not None:
            rows.append((ln["options"], ln["target"], p))
    n = len(rows)
    out: dict = {"answered": n, "total": len(lines), "complete": n == len(lines)}
    if task == "noul":
        prevalence = sum(t["true"] for _, t in entries) / len(entries)
        tp = sum(t["true"] for _, t, p in rows if p["true"] >= 0.5)
        fp = sum(t["false"] for _, t, p in rows if p["true"] >= 0.5)
        fn = sum(t["true"] for _, t, p in rows if p["true"] < 0.5)
        tn = sum(t["false"] for _, t, p in rows if p["true"] < 0.5)
        recall = tp / (tp + fn) if tp + fn > EPS else None
        spec = tn / (tn + fp) if tn + fp > EPS else None
        ba = (recall + spec) / 2 if recall is not None and spec is not None else None
        base = sum((t["true"] - prevalence) ** 2 for _, t in entries) / len(entries)
        brier = sum((p["true"] - t["true"]) ** 2 for _, t, p in rows) / n if n else None
        out.update({
            "binary_brier": brier, "accuracy": (tp + tn) / n if n else None, "positive_recall": recall, "specificity": spec,
            "precision": tp / (tp + fp) if tp + fp > EPS else None, "balanced_accuracy": ba,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > EPS else None,
            "positive_prevalence": prevalence, "majority_accuracy_baseline": max(prevalence, 1 - prevalence), "constant_brier_baseline": base,
            "brier_skill": 1 - brier / base if base > EPS and brier is not None else None,
            "baseline_adjusted_skill": 2 * ba - 1 if ba is not None else None,
        })  # fmt: skip
        if n:  # extra: reliability of P(true) against the soft target, equal mass
            out["ece_true"] = ece_from([p["true"] for _, _, p in rows], [t["true"] for _, t, _ in rows])
    elif task == "choice":
        uniform = sum(1 / len(o) for o, _ in entries) / len(entries)
        ids = {x["id"] for o, _ in entries for x in o}
        fixed_id = max(sum(t.get(k, 1 / len(o)) for o, t in entries) / len(entries) for k in ids)
        width = max(len(o) for o, _ in entries)
        fixed_pos = max(sum(t[o[i]["id"]] if i < len(o) else 1 / len(o) for o, t in entries) / len(entries) for i in range(width))
        baseline = max(uniform, fixed_id, fixed_pos)
        mass, brier, xent, conf = [], [], [], []
        for o, t, p in rows:
            pv, tv = _vec(o, p), _vec(o, t)
            mass.append(float(tv[int(np.argmax(pv))]))  # ties -> first option, as the official max()
            brier.append(float(np.sum((pv - tv) ** 2)))
            xent.append(float(-np.sum(tv * np.log(np.clip(pv, 1e-12, 1.0)))))
            conf.append(float(pv.max()))
        acc = sum(mass) / n if n else None
        out.update({
            "target_mass_at_prediction": acc, "uniform_accuracy_baseline": uniform, "fixed_answer_accuracy_baseline": baseline,
            "baseline_adjusted_skill": (acc - baseline) / (1 - baseline) if acc is not None and baseline < 1 - EPS else None,
            "brier": float(np.mean(brier)) if n else None, "cross_entropy": float(np.mean(xent)) if n else None,
            "ece": ece_from(conf, mass) if n else None,
        })  # fmt: skip
    elif task == "score":

        def norm_value(opts, probs):
            sc = {x["id"]: float(x["value"]) for x in opts}
            lo, hi = min(sc.values()), max(sc.values())
            return sum((v - lo) / (hi - lo) * probs[k] for k, v in sc.items())

        targets = [norm_value(o, t) for o, t in entries]
        const = median(targets)
        base = sum(abs(x - const) for x in targets) / len(targets)
        errs = []
        for o, t, p in rows:
            sc = {x["id"]: x["value"] for x in o}
            errs.append(sum(v * (p[k] - t[k]) for k, v in sc.items()) / (max(sc.values()) - min(sc.values())))
        mae = sum(abs(e) for e in errs) / n if n else None
        out.update({
            "normalized_expected_score_mae": mae, "normalized_score_rmse": (sum(e * e for e in errs) / n) ** 0.5 if n else None,
            "constant_score_baseline": const, "constant_mae_baseline": base,
            "baseline_adjusted_skill": 1 - mae / base if base > EPS and mae is not None else None,
        })  # fmt: skip
    else:
        raise ValueError(f"unknown task {task!r}")
    skill = out["baseline_adjusted_skill"]
    out["baseline_adjusted_score"] = max(0.0, min(1.0, skill)) if skill is not None else None
    return out


PRIMARY = {"choice": "target_mass_at_prediction", "noul": "binary_brier", "score": "normalized_expected_score_mae"}


def aggregate(per_benchmark: dict[str, dict], tasks: dict[str, str], *, covered: Sequence[str] | None = None) -> dict:
    """Task scores and Task Avg over `covered` benchmarks (default: all given).  Per task: mean of the adjusted scores of the eligible ones
    (skill defined); Task Avg = mean of the three task scores, None if a task has no eligible benchmark."""
    ids = [b for b in (covered if covered is not None else per_benchmark) if b in per_benchmark]
    out: dict = {"n_benchmarks": len(ids), "tasks": {}}
    for task in ("choice", "noul", "score"):
        sc = [per_benchmark[b]["baseline_adjusted_score"] for b in ids if tasks[b] == task and per_benchmark[b]["baseline_adjusted_score"] is not None]
        n_t = sum(1 for b in ids if tasks[b] == task)
        out["tasks"][task] = {"score": float(np.mean(sc)) if sc else None, "eligible": len(sc), "benchmarks": n_t}
    vals = [v["score"] for v in out["tasks"].values()]
    out["task_avg"] = float(np.mean(vals)) if all(v is not None for v in vals) else None
    return out


# ----------------------------------------------------------------------------------------------------------- rendering
def state_text(state) -> str:
    """Flat dict of scalars -> "key: value" lines; anything else -> compact JSON (structure preserved)."""
    if isinstance(state, dict) and state and all(isinstance(v, str | int | float | bool) or v is None for v in state.values()):
        return "\n".join(f"{k}: {v}" for k, v in state.items())
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def option_labels(line: dict) -> list[str]:
    """What the model reads for each option, in the stored order: Noul -> the authored false/true definitions, Choice -> descriptions,
    Score -> "value: description" (the numeric value is part of the criterion)."""
    if line["task"] == "score":
        return [f"{o['value']:g}: {_t(o['description'])}" for o in line["options"]]
    return [_t(o["description"]) for o in line["options"]]


def render(line: dict, order: Sequence[int] | None = None) -> tuple[str, str, list[str]]:
    """(context, question, labels in SHOWN order) for one S1MB decision.  `order` permutes the shown options (shown j = stored order[j]).
    Only the inference fields are read."""
    ctx = state_text(line["state"])
    if line.get("system_prompt"):
        ctx = f"{line['system_prompt']}\n\n{ctx}"
    labels = option_labels(line)
    order = list(range(len(labels))) if order is None else list(order)
    return ctx, _t(line["instructions"]), [labels[i] for i in order]


def predict_line(decider: TypedDecider, line: dict, *, uid: str, order: Sequence[int] | None = None) -> list[float]:
    """Probabilities over the STORED option order (realigned when `order` permutes the shown options)."""
    ctx, question, labels = render(line, order)
    d = make_decision(f"s1mb::{uid}", question, labels, lettered=True)
    with registered(d):
        probs = decider.decide(d.name, ctx)
    p = list(probs.values())
    if order is None:
        return p
    return realign(p, list(order)).tolist()


def as_pred(line: dict, vec: Sequence[float]) -> dict:
    return {o["id"]: float(v) for o, v in zip(line["options"], vec, strict=True)}


def make_decider(base: TypedDecider, variant: str) -> TypedDecider:
    """`prod`: typed.py as is.  `listed`: stage 1 lists the labels of each group (doc section 2.4)."""
    if variant == "prod":
        return base
    if variant == "listed":
        return ListedTwoStage(base.model, base.temperatures, base.cap)
    raise ValueError(variant)


def run_line(decider: TypedDecider, ln: dict, uid: str, *, max_chars: int = 9000, permute: bool = False,
             clock: Callable[[], float] = time.perf_counter) -> dict:  # fmt: skip
    """Score one line.  Result: probs (stored order), seconds, or `error` (server failure, or the prompt is over `max_chars`: such a line is
    unscored and never truncated, so its benchmark is incomplete).  With `permute`, choice lines also get the 3 permuted arms
    ('perm_probs', realigned to the stored order); `uid` must be unique among concurrently running lines (decisions are registered by name)."""
    rec: dict = {"case_id": ln["case_id"], "question_id": ln["question_id"]}
    ctx, q, labels = render(ln)
    if len(ctx) + len(q) + sum(len(x) for x in labels) > max_chars:
        rec["error"] = "over_max_chars"
        return rec
    t0 = clock()
    try:
        rec["probs"] = predict_line(decider, ln, uid=uid)
        rec["seconds"] = clock() - t0
        if permute and ln["task"] == "choice" and len(labels) >= 2:
            rec["perm_probs"] = [predict_line(decider, ln, uid=f"{uid}p{k}", order=o) for k, o in enumerate(permutations(len(labels), PERM_SEEDS)[1:])]
    except Exception as e:  # server error / context overflow: unscored
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return rec
