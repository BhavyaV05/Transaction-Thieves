"""The pool of back-tested questions a non-LLM planner chooses from (hand-off #9 from C7).

Owner: T2 (Bhavya).  Built from the accepted rows of `lr_table.jsonl` (`ftm.backtest.engine.lr_table_rows`).

A question is only usable on an alert whose partner side it can address: a program about ALERT.receiver is for an
alert where the home bank sent the money (the receiver is at the partner), ALERT.sender for the opposite direction;
questions about a set slot work for both.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from ftm.backtest.lr import answer_key
from ftm.dsl.parser import Node, parse, sensitivity
from ftm.dsl.spec import ACCOUNT_SLOTS

DIRECTION_SLOT = {"out": "receiver", "in": "sender"}  # alert direction -> the partner-side account slot


@dataclass(frozen=True)
class Candidate:
    program: str  # canonical form
    eps: float
    family: str
    kind: str
    slot: str | None  # the account slot the program is about (None for set-only programs)
    lr: dict  # cell -> clipped LR
    bins: tuple
    p_cell: dict | None  # cell -> (P(cell | clean), P(cell | launder)); None if the table has none
    sensitivity: int

    @property
    def cost(self) -> float:
        """Budget cost: eps, and nothing for the no-noise arm (eps = inf)."""
        return self.eps if math.isfinite(self.eps) else 0.0

    @property
    def rank_cost(self) -> float:
        """What EIG divides by: eps, or one unit per question for the no-noise arm."""
        return self.eps if math.isfinite(self.eps) else 1.0

    def cell(self, value) -> str:
        return answer_key(value, self.bins or None)

    def log_lr(self, value) -> float | None:
        v = self.lr.get(self.cell(value))
        return math.log(v) if v is not None and v > 0 else None

    def applies_to(self, direction: str) -> bool:
        return self.slot is None or self.slot == DIRECTION_SLOT[direction]


def _account_slot(node: Node) -> str | None:
    while isinstance(node.subject, Node):
        node = node.subject
    return node.subject if node.subject in ACCOUNT_SLOTS else None


class QuestionPool:
    def __init__(self, candidates: Iterable[Candidate]):
        self.candidates = sorted(candidates, key=lambda c: (c.program, c.eps))

    @classmethod
    def from_rows(cls, rows: Iterable[dict], only_accepted: bool = True) -> QuestionPool:
        out = []
        for r in rows:
            if only_accepted and not r["accepted"]:
                continue
            node = parse(r["program_canonical"])
            pc = r.get("p_cell")
            out.append(
                Candidate(
                    program=r["program_canonical"], eps=float(r["eps"]), family=r["family"], kind=node.prim.kind,
                    slot=_account_slot(node), lr=dict(r["lr"]), bins=tuple(r["bins"] or ()),
                    p_cell={k: (v[0], v[1]) for k, v in pc.items()} if pc else None,
                    sensitivity=sensitivity(node),
                )  # fmt: skip
            )
        return cls(out)

    @classmethod
    def from_jsonl(cls, path: str | Path, only_accepted: bool = True) -> QuestionPool:
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        return cls.from_rows(rows, only_accepted)

    def at_eps(self, levels: Iterable[float]) -> QuestionPool:
        """The no-noise rows (eps = inf) offered at each of `levels`, with the same LRs: a table measured on exact answers, used
        on the noisy answers a bank gives. Rows at a finite eps stay as they are. A bank refuses eps = inf, so without this an
        `--exact-labels` table can never be asked."""
        levels = [float(e) for e in levels]
        return QuestionPool(
            replace(c, eps=e) for c in self.candidates for e in (levels if not math.isfinite(c.eps) else [c.eps])
        )

    def applicable(self, direction: str) -> list[Candidate]:
        return [c for c in self.candidates if c.applies_to(direction)]

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self):
        return iter(self.candidates)
