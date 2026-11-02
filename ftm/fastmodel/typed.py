"""Typed decisions: the System-One-style API of the fast model (C13).

Owner: T3 (Akshat).  Spec: wave 4 (docs/t3_system1.md, faithfulness table).

TypeSafe's published description of its System One models ("Jev"): a decision is TYPED over an option set defined in advance, so the
answer can never be a type error; ALL option probabilities come back together, each decision carries a calibrated confidence; large
option sets are scored in two stages (score groups, then choose within the group).  This module is that interface for our fast model:

    TypedDecider.decide("worth_asking", text) -> {"yes": 0.83, "no": 0.17}     # the full calibrated distribution, sums to 1

The pipeline (ftm.graph.system1) consumes only these distributions and the conformal cascade built on them, never free text.
Option sets up to `cap` labels are read from ONE forward pass (the first output token's top logprobs, all options at once).  Above
`cap` the options are split into groups of `cap`; stage 1 scores the group letter, stage 2 scores the label letter inside each
group, and p(label) = p(group) * p(label | group) (one extra forward pass per group, all independent so they can run in parallel).
Not reproduced: Jev's non-autoregressive "parallel sampler" and its RL-for-calibration training (unpublished); multi-token options
(teacher-forced logprobs) are not needed for our option sets, which are all single tokens.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ftm.fastmodel.calibration import softmax_t
from ftm.fastmodel.decisions import DECISIONS, DecisionExample, messages
from ftm.fastmodel.model import FLOOR_LOGP, logps_from_variants

LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def group_split(n_labels: int, cap: int = len(LETTERS)) -> list[list[int]]:
    """Label indices in consecutive groups of at most `cap` (one group when n_labels <= cap)."""
    return [list(range(i, min(i + cap, n_labels))) for i in range(0, n_labels, cap)]


class TypedDecider:
    """`model`: a LogprobFastModel-like (`client`, `extra`, `top_logprobs`) with its fitted temperature; `temperatures`: per decision
    (overrides model.temperature). `cap`: the largest option set read in one pass (default: the 26 single-letter tokens)."""

    def __init__(self, model, temperatures: dict[str, float] | None = None, cap: int = len(LETTERS)):
        self.model, self.temperatures, self.cap = model, dict(temperatures or {}), cap

    def _t(self, decision: str) -> float:
        return self.temperatures.get(decision, self.model.temperature)

    def raw_logps(self, decision: str, text: str) -> np.ndarray:
        """Label log-probs before temperature, over DECISIONS[decision].labels."""
        d = DECISIONS[decision]
        if len(d.labels) <= self.cap:
            return self.model.raw_logps(DecisionExample(decision, text))
        return self._two_stage(d, text)

    def decide(self, decision: str, text: str) -> dict[str, float]:
        """The full calibrated distribution over the decision's enumerated labels."""
        lp = self.raw_logps(decision, text)
        p = softmax_t(lp[None, :], self._t(decision))[0]
        return dict(zip(DECISIONS[decision].labels, p.tolist(), strict=True))

    # -- two-stage scoring (option sets above the cap) ------------------------------------------------------------
    def _read(self, msgs: list[dict], tokens: Sequence[str]) -> np.ndarray:
        r = self.model.client.chat(msgs, max_tokens=1, temperature=0.0,
                                   extra={**self.model.extra, "logprobs": True, "top_logprobs": self.model.top_logprobs})  # fmt: skip
        try:
            top = r.raw["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError):
            top = []
        t2i = {t.lower(): i for i, t in enumerate(tokens)}
        seen: list[list[float]] = [[] for _ in tokens]
        for item in top:
            i = t2i.get(item["token"].strip().lower())
            if i is not None:
                seen[i].append(float(item["logprob"]))
        return logps_from_variants(seen)

    def _two_stage(self, d, text: str) -> np.ndarray:
        groups = group_split(len(d.labels), self.cap)
        ex = DecisionExample(d.name, text)
        base = messages(ex)
        glogp = self._read(_ask(base, d.question, [f"{LETTERS[g]} = labels {d.labels[idx[0]]} .. {d.labels[idx[-1]]}" for g, idx in enumerate(groups)]),
                           LETTERS[: len(groups)])  # fmt: skip
        glogp = glogp - np.logaddexp.reduce(glogp)
        out = np.full(len(d.labels), FLOOR_LOGP)
        for g, idx in enumerate(groups):
            inner = self._read(_ask(base, d.question, [f"{LETTERS[j]} = {d.labels[i]}" for j, i in enumerate(idx)]), LETTERS[: len(idx)])
            inner = inner - np.logaddexp.reduce(inner)
            for j, i in enumerate(idx):
                out[i] = glogp[g] + inner[j]
        return out


def _ask(base: list[dict], question: str, options: list[str]) -> list[dict]:
    head = base[-1]["content"].split("\n\nQuestion:")[0]
    return [*base[:-1], {"role": "user", "content": f"{head}\n\nQuestion: {question}\nAnswer with exactly one letter: " + ", ".join(options) + "."}]


def entropy(p: dict[str, float]) -> float:
    return -sum(v * math.log(v) for v in p.values() if v > 0)
