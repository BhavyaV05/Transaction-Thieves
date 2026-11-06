"""External-validity tasks of S1BENCH Part A: items, typed decisions, per-item scoring, task reports.

Owner: T3 (Akshat).  Spec: docs/t3_s1bench.md sections 2.1, 2.3, 2.4.

Prepared data (scripts/s1bench_prepare.py) is read from JSONL, never from Parquet here.  Task kinds:
  multiclass / multiclass_oos   one single-label decision over the label names (CLINC150 adds `out of scope` as a 151st option)
  binary_yesno                  ToxicChat: yes/no, label 0 = yes (toxic)
  multilabel_yesno              UNFAIR-ToS: one yes/no decision per clause and unfair-term type (8 per clause)
`arms` are the option presentations scored for an item: `main` (production format: yes/no words for binary tasks, letters otherwise) and, for
the option-order test, `L0` (lettered identity; the same as `main` for lettered tasks) and `L1..L3` (three fixed random permutations).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ftm.fastmodel.calibration import fit_temperature, softmax_t
from ftm.fastmodel.s1bench import (
    PERM_SEEDS,
    Item,
    TaskSpec,
    auroc,
    binary_prf,
    core_metrics,
    latency_summary,
    make_decision,
    order_bias,
    permutations,
    realign,
    registered,
)
from ftm.fastmodel.typed import TypedDecider


@dataclass
class ExtTask:
    name: str
    kind: str
    specs: list[TaskSpec]  # one spec per decision type (unfair_tos: 8; the others: 1)
    meta: dict


def load_items(prepared: Path, name: str, part: str) -> tuple[dict, list[dict]]:
    meta = json.loads((prepared / name / "meta.json").read_text())
    rows = [json.loads(x) for x in (prepared / name / f"{part}.jsonl").read_text().splitlines() if x.strip()]
    return meta, rows


def build_task(meta: dict) -> ExtTask:
    kind = meta["kind"]
    if kind in ("multiclass", "multiclass_oos"):
        specs = [TaskSpec(meta["task"], meta["question"], tuple(meta["labels"]))]
    elif kind == "binary_yesno":
        specs = [TaskSpec(meta["task"], meta["question"], ("yes", "no"))]
    elif kind == "multilabel_yesno":
        specs = [TaskSpec(f"{meta['task']}/{lb}", meta["question"].format(label=lb), ("yes", "no")) for lb in meta["labels"]]
    else:
        raise ValueError(kind)
    return ExtTask(meta["task"], kind, specs, meta)


def items_for(task: ExtTask, rows: Sequence[dict]) -> list[list[Item]]:
    """Per spec, the items with their gold label index (into spec.labels)."""
    out = []
    for si in range(len(task.specs)):
        if task.kind == "multilabel_yesno":
            out.append([Item(f"{r['id']}#{si}", r["text"], 0 if si in r["label"] else 1, cluster=r["id"]) for r in rows])
        else:
            out.append([Item(r["id"], r["text"], int(r["label"]), cluster=r["id"]) for r in rows])
    return out


# ----------------------------------------------------------------------------------------------------- scoring one item
def arm_defs(spec: TaskSpec, with_perm: bool) -> list[tuple[str, list[int], bool]]:
    """(arm name, shown order as indices into spec.labels, lettered?).  `main` first; then L0 (if main is not lettered) and L1..L3."""
    k = len(spec.labels)
    ident = list(range(k))
    arms = [("main", ident, not spec.binary_yes_no)]
    if with_perm:
        if spec.binary_yes_no:
            arms.append(("L0", ident, True))
        for j, p in enumerate(permutations(k, PERM_SEEDS)[1:], 1):
            arms.append((f"L{j}", p, True))
    return arms


def score_item(decider: TypedDecider, spec: TaskSpec, item: Item, uid: str, *, with_perm: bool,
               clock: Callable[[], float] = time.perf_counter) -> dict:  # fmt: skip
    """All arms of one item.  Probabilities are returned in the CANONICAL label order (shown order realigned).  `seconds` is the wall time of the
    `main` decision only (all its passes)."""
    rec: dict = {"id": item.id, "label": item.label}
    seen: dict[tuple, list] = {}  # identical shown orders (every permutation of a 2-option task) are asked once
    for arm, order, lettered in arm_defs(spec, with_perm):
        key = (tuple(order), lettered)
        if key in seen and arm != "main":
            rec[arm] = seen[key]
            continue
        labels = [spec.labels[i] for i in order]
        d = make_decision(f"s1b::{uid}::{arm}", spec.question, labels, hint=spec.hint, lettered=lettered)
        t0 = clock()
        with registered(d):
            probs = list(decider.decide(d.name, item.text).values())
        dt = clock() - t0
        rec[arm] = seen[key] = realign(probs, order).tolist()
        if arm == "main":
            rec["seconds"] = dt
    if with_perm and "L0" not in rec:  # lettered task: the main arm IS the lettered identity arm
        rec["L0"] = rec["main"]
    return rec


# ---------------------------------------------------------------------------------------------------------- task reports
def fit_scalar_t(probs: np.ndarray, y: np.ndarray) -> float:
    return fit_temperature(np.log(np.clip(np.asarray(probs, float), 1e-12, 1.0)), np.asarray(y, int))


def rescale(probs: np.ndarray, t: float) -> np.ndarray:
    return softmax_t(np.log(np.clip(np.asarray(probs, float), 1e-12, 1.0)), t)


def _task_specific(task: ExtTask, P: np.ndarray, y: np.ndarray) -> dict:
    out: dict = {}
    pred = P.argmax(axis=1)
    if task.kind == "multiclass_oos":
        oos = task.meta["oos_label"]
        ins = y != oos
        out["in_scope_accuracy"] = float(np.mean(pred[ins] == y[ins])) if ins.any() else None
        tp = int(np.sum((pred == oos) & (y == oos)))
        out["oos_recall"] = tp / int(np.sum(y == oos)) if np.any(y == oos) else None
        out["oos_precision"] = tp / int(np.sum(pred == oos)) if np.any(pred == oos) else None
    if task.kind == "binary_yesno":
        pos = (y == 0).astype(int)  # yes = toxic
        out.update({f"toxic_{k}": v for k, v in binary_prf((pred == 0).astype(int), pos).items() if k in ("f1", "precision", "recall")})
        out["auroc"] = auroc(P[:, 0], pos)
    return out


def evaluate_task(task: ExtTask, per_spec_rows: Sequence[Sequence[dict]], calib_rows: Sequence[Sequence[dict]] | None) -> dict:
    """Metrics of one task from scored items (one list per spec).  `calib_rows`: scored calibration items (same structure) used ONLY to fit one T."""
    def stack(rows, arm):
        return np.array([r[arm] for r in rows], float), np.array([r["label"] for r in rows], int)

    P = np.concatenate([stack(rs, "main")[0] for rs in per_spec_rows]) if task.kind != "multilabel_yesno" else None
    out: dict = {"task": task.name, "kind": task.kind, "n_items": len(per_spec_rows[0]), "n_decisions": sum(len(r) for r in per_spec_rows)}
    if task.kind == "multilabel_yesno":
        Ps = [stack(rs, "main")[0] for rs in per_spec_rows]
        ys = [stack(rs, "main")[1] for rs in per_spec_rows]
        P, y = np.concatenate(Ps), np.concatenate(ys)
        pred_yes = [(p.argmax(1) == 0).astype(int) for p in Ps]
        gold_yes = [(yy == 0).astype(int) for yy in ys]
        per_type = [binary_prf(p, g) for p, g in zip(pred_yes, gold_yes, strict=True)]
        tp, fp, fn = (sum(t[k] for t in per_type) for k in ("tp", "fp", "fn"))
        out["micro_f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        out["macro_f1"] = float(np.mean([t["f1"] or 0.0 for t in per_type]))
        out["per_type"] = {s.name.split("/", 1)[1]: t for s, t in zip(task.specs, per_type, strict=True)}
        out["core"] = core_metrics(P, y)
    else:
        y = np.concatenate([stack(rs, "main")[1] for rs in per_spec_rows])
        out["core"] = core_metrics(P, y, task.specs[0].labels if len(task.specs[0].labels) <= 2 else None)
        out.update(_task_specific(task, P, y))
    out["latency"] = latency_summary([r["seconds"] for rs in per_spec_rows for r in rs if r.get("seconds") is not None])
    if calib_rows:
        if task.kind == "multilabel_yesno":
            Pc = np.concatenate([stack(rs, "main")[0] for rs in calib_rows])
            yc = np.concatenate([stack(rs, "main")[1] for rs in calib_rows])
        else:
            Pc, yc = stack(calib_rows[0], "main")
        t = fit_scalar_t(Pc, yc)
        out["temperature"] = t
        out["core_scaled"] = core_metrics(rescale(P, t), y)
    # option-order bias: needs L0..L3 on the same items
    if all(r.get("L1") is not None for r in per_spec_rows[0]):
        arms_ok = [rs for rs in per_spec_rows if all(r.get("L0") is not None and r.get("L1") is not None for r in rs)]
        if arms_ok:
            P0 = np.concatenate([np.array([r["L0"] for r in rs], float) for rs in arms_ok])
            Pp = [np.concatenate([np.array([r[f"L{j}"] for r in rs], float) for rs in arms_ok]) for j in range(1, len(PERM_SEEDS) + 1)]
            yy = np.concatenate([np.array([r["label"] for r in rs], int) for rs in arms_ok])
            out["order_bias"] = order_bias(P0, Pp, yy) | {"n_decisions": int(len(yy))}
    return out
