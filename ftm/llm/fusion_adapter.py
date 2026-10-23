"""Adapter from our `Fusion` protocol (ftm.llm.investigation) to T4's fusion model (`ftm.fusion.fuse`). PR #7 review.

Owner: T3 (Akshat).  Spec: plan.md 6.9; tracks.md 5.4, 6.4.

Per alert: the accepted, answered log entries become one (alert, family, log LR) row each, `answer_features` sums
them per family into X, and `Fusion(mode).predict(p0, X)` gives the posterior. Same selection as NaiveLRFusion
(allowed reply with an accepted back-test LR; everything else contributes nothing), so `uses()` is inherited.
`Fusion("naive")` gives the same posteriors as `NaiveLRFusion` (tests/test_fusion_adapter.py).

Fitted modes (`shared`, `per_family`) need T2's back-test table (LR + family per accepted question) for the val
answers to be fitted on: pass the fitted model in. A fitted `per_family` model must have `families` set (the fit
does not set it); a family in the log that the model does not know raises rather than dropping evidence.
"""

from __future__ import annotations

import math

import numpy as np

from ftm.fusion import fuse
from ftm.llm.investigation import LRTable, NaiveLRFusion
from ftm.types import QuestionLogEntry

UNKNOWN_FAMILY = "_unknown"  # accepted entry whose back-test row has no family (only reached in naive mode)


class FusedLR(NaiveLRFusion):
    def __init__(self, model: fuse.Fusion, table: LRTable | None = None):
        if model.mode == "per_family" and model.fitted and not model.families:
            raise ValueError("a fitted per_family model needs `families` (the order of its weights)")
        super().__init__(table)
        self.model = model

    @classmethod
    def from_params(cls, params: dict, table: LRTable | None = None) -> FusedLR:
        """Rebuild a model from `Fusion.params()` (e.g. saved JSON of the one fitted on val)."""
        m = fuse.Fusion(params["mode"], list(params["families"]), params["a"], np.asarray(params["w"], dtype="f8"),
                        params["c"], params["ridge"], params["fitted"])  # fmt: skip
        return cls(m, table)

    def _family(self, e: QuestionLogEntry) -> str:
        entry = self.table.get(e.program, e.eps)
        return entry.family or UNKNOWN_FAMILY

    def posterior(self, p0: float, log: list[QuestionLogEntry]) -> float:
        used = [(e, lr) for e in log if (lr := self._lr(e)) is not None and lr > 0]
        fams = self.model.families or None
        if self.model.mode != "naive" and any(self._family(e) == UNKNOWN_FAMILY for e, _ in used):
            raise KeyError("an accepted question has no family in the back-test table; fitted fusion needs one")
        X, _ = fuse.answer_features(["a"], ["a"] * len(used), [self._family(e) for e, _ in used],
                                    [math.log(lr) for _, lr in used], families=fams)  # fmt: skip
        return float(self.model.predict([p0], X)[0])
