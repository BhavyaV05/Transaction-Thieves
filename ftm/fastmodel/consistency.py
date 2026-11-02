"""Consistency of the fast model under meaning-preserving perturbations of a decision's text (C13).

Owner: T3 (Akshat).  Spec: wave 4 (docs/t3_system1.md).  TypeSafe's Jev promises "similar answers for similar inputs"; this
measures it.  `perturb(decision, context, rng)` returns variants that mean the same thing:
  worth_asking / typology   the "Answers so far" lines in another order, the shape fields in another order, numbers written with a
                            trailing zero or fewer digits that do not change them
  injection_present         the planted (or benign) note moved to another carrier: the benign sentences around it swapped, or the
                            note alone; re-seeded bank id
  rule_applies              another cited-rule id format is NOT touched (it changes the meaning); only the leading jurisdiction sentence order
Accounts are aliased in the pipeline already, so "re-seeded aliases" means the bank id in the note and the alias-free shape order.
`consistency_stats` compares distributions: flip rate of the argmax and the mean total-variation distance.
"""

from __future__ import annotations

import random
import re

import numpy as np

SHAPE_RE = re.compile(r"(Shape of the counterparty's activity: )(.*?)(\. One-bank model score)", re.S)
NOTE_RE = re.compile(r'^(Partner bank )(\S+)( replied to a question with an answer and this note: ")(.*)(")$', re.S)


def _shuffle_shape(text: str, rng: random.Random) -> str:
    m = SHAPE_RE.search(text)
    if not m:
        return text
    parts = [x.strip() for x in m.group(2).split(",")]
    rng.shuffle(parts)
    return text[: m.start(2)] + ", ".join(parts) + text[m.end(2):]


def _reorder_answers(text: str, rng: random.Random) -> str:
    lines = text.split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.startswith("- ") and "answer" in ln]
    vals = [lines[i] for i in idx]
    rng.shuffle(vals)
    for i, v in zip(idx, vals, strict=True):
        lines[i] = v
    return "\n".join(lines)


def _rewrite_numbers(text: str) -> str:
    text = re.sub(r"eps left (\d+(?:\.\d+)?)", lambda m: f"eps left {float(m.group(1)):.1f}", text)
    return re.sub(r"One-bank model score (\d\.\d+)", lambda m: f"One-bank model score {float(m.group(1)):.3f}", text)


def _note_variant(text: str, rng: random.Random) -> str:
    m = NOTE_RE.match(text)
    if not m:
        return text
    sentences = re.split(r"(?<=[.!])\s+", m.group(4).strip())
    if len(sentences) > 1:
        rng.shuffle(sentences)  # the attack sentence and its benign carriers in another order
    return f"{m.group(1)}{rng.randint(0, 99)}{m.group(3)}{' '.join(sentences)}{m.group(5)}"


def perturb(decision: str, context: str, rng: random.Random, n: int = 2) -> list[str]:
    out = []
    for _ in range(n):
        if decision in ("worth_asking", "typology"):
            t = _rewrite_numbers(_shuffle_shape(_reorder_answers(context, rng), rng))
        elif decision == "injection_present":
            t = _note_variant(context, rng)
        else:
            t = context
        if t != context:
            out.append(t)
    return out


def consistency_stats(p_orig: np.ndarray, p_pert: np.ndarray) -> dict:
    """p_*: (n, k) label distributions of the original and its perturbed twin (paired rows)."""
    po, pp = np.asarray(p_orig, "f8"), np.asarray(p_pert, "f8")
    if po.shape != pp.shape or len(po) == 0:
        raise ValueError("paired (n, k) distributions needed")
    tv = 0.5 * np.abs(po - pp).sum(axis=1)
    return {"n": int(len(po)), "flip_rate": float((po.argmax(1) != pp.argmax(1)).mean()), "mean_tv": float(tv.mean()),
            "p90_tv": float(np.quantile(tv, 0.9))}  # fmt: skip
