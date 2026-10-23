"""Enumerate every DSL program up to a size (the Enumerator planner's search space, plan.md 6.10).

Owner: T2 (Bhavya).  Spec: tracks.md 4.7 ("every DSL program up to size n, back-tested the same way (C7), picked by EIG").

The language has unbounded numeric parameters, so enumeration runs over a **grid** of values per parameter name
(`DEFAULT_GRID`, committed so the search space is part of the experiment's configuration). Size is
`ftm.dsl.parser.size`: one per call plus one per parameter that differs from its default, so `max_size` bounds the
number of free choices. Programs are returned in canonical form, de-duplicated, in a stable order.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence

from ftm.dsl.parser import Node, canonical, parse, size
from ftm.dsl.spec import ACCOUNT_SLOTS, PRIMS, REQUIRED, SET_SLOTS, P, Prim

DEFAULT_GRID: dict[str, tuple] = {
    "hours": (24, 72, 168),  # window lengths
    "min_frac": (0.5, 0.8),
    "k": (2, 3, 5),  # at_least thresholds
    "days": (3, 7),
    "mult": (3.0, 5.0),
    "tol": (0.01, 0.05),
    "unit": (100.0, 1000.0),
    "min_amount": (1000.0, 9000.0),
    "max_amount": (3000.0, 10000.0),
    "max_hops": (2, 3),
}


def _values(prim: Prim, key: str, p: P, grid: dict, bank_sets: Sequence[Sequence[int]]) -> list:
    """Values to try for one parameter, as raw (pre-typecheck) text fragments."""
    if p.kind == "enum":
        return list(p.choices)
    if p.kind == "window":
        return [f"{m}({h})" for m in p.choices for h in grid["hours"]]
    if p.kind == "set_slot":
        return [f"ALERT.{s}" for s in SET_SLOTS]
    if p.kind == "banks":
        return [str(list(b)) for b in bank_sets]
    vals = [v for v in grid.get(key, ()) if p.lo <= v <= p.hi]
    return [f"{v:g}" for v in vals]


def _calls(prim: Prim, subject: str, grid: dict, bank_sets, max_size: int) -> Iterable[str]:
    keys = list(prim.params)
    options = []
    for k in keys:
        p = prim.params[k]
        vals = _values(prim, k, p, grid, bank_sets)
        if p.default is not REQUIRED:
            vals = [None, *vals]  # the parameter may stay at its default
        options.append(vals)
    for combo in itertools.product(*options):
        if any(v is None and prim.params[k].default is REQUIRED for k, v in zip(keys, combo, strict=True)):
            continue
        if sum(v is not None for v in combo) + 1 > max_size:  # cheap pre-filter: a call plus its free parameters
            continue
        args = [subject, *(f"{k}={v}" for k, v in zip(keys, combo, strict=True) if v is not None)]
        yield f"{prim.name}({', '.join(args)})"


def enumerate_programs(max_size: int = 4, subjects: Sequence[str] = ("receiver",), grid: dict | None = None,
                       bank_sets: Sequence[Sequence[int]] = (), include_at_least: bool = True) -> list[Node]:  # fmt: skip
    """All askable programs of size <= max_size over the grid.

    `subjects`: the account slots to use (the alert's partner side: "receiver" for an outgoing alert, "sender" for an
    incoming one). `bank_sets`: lists of bank ids for `top_outflow_bank` (empty = that primitive is skipped).
    """
    grid = {**DEFAULT_GRID, **(grid or {})}
    for s in subjects:
        if s not in ACCOUNT_SLOTS:
            raise ValueError(f"unknown account slot {s!r}")
    seen: dict[str, Node] = {}

    def add(text: str):
        node = parse(text)
        if size(node) <= max_size:
            seen.setdefault(canonical(node), node)

    base: list[str] = []
    for prim in PRIMS.values():
        if prim.restricted or prim.subject == "count":
            continue
        slots = [f"ALERT.{s}" for s in (subjects if prim.subject == "account" else SET_SLOTS)]
        for slot in slots:
            for text in _calls(prim, slot, grid, bank_sets, max_size):
                add(text)
                if prim.kind in ("count", "set_count"):
                    base.append(text)
    if include_at_least:
        for inner in base:
            for k in grid["k"]:
                add(f"at_least({inner}, k={k})")
    return [seen[c] for c in sorted(seen)]
