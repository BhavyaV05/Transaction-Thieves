"""p0-conditional fusion: logit P(y) = a + b*logit(p0) + shrink * sum_f w_f * x_f.

Owner: T2 (rewrite of 1 Nov, finding 2).  Companion of `ftm.fusion.fuse` (T4), which fits on validation answers.

Why a new one.  Naive fusion multiplies marginal back-test LRs onto logit p0.  The LRs were estimated without conditioning on
p0 and near-duplicate questions (dormant 3d/7d, new_account 2d/5d, returns_home 24h/72h) are summed as if independent.  Here
the weights are fitted on the TRAIN alerts' own per-alert back-test rows (`write_alert_rows`), jointly with p0:
  * x_f = the summed log-LR of the accepted answers of family f for that alert (one answer per question; at fit time one eps
    level per (alert, question) is drawn from the levels the table accepts, since a run never gets the same question twice);
  * one weight w_f per family, so a family of near-duplicates counts about once; ridge pulls b toward 1 and each w_f toward 1
    (naive Bayes) with `ridge` pseudo-observations; the intercept a is free;
  * `shrink` in [0, 1] scales the whole evidence term at inference. It is chosen on val fit+cal alerts only (`choose_shrink`),
    never on eval or test. `base="p0"` replaces a + b*logit(p0) by logit(p0), for when the score is already calibrated for the
    period it is applied to (the weights were still fitted conditionally on p0).
Numbers only: no LLM text enters.  Params serialise to JSON with a content hash (`to_json` / `from_json`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from ftm.fusion.fuse import logit, sigmoid

BASES = ("fitted", "p0")
SHRINKS = (0.0, 0.25, 0.5, 0.75, 1.0)
RIDGES = (1.0, 10.0, 100.0, 1000.0)


@dataclass
class ConditionalFusion:
    families: list = field(default_factory=list)
    a: float = 0.0
    b: float = 1.0
    w: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ridge: float = 10.0
    shrink: float = 1.0
    base: str = "fitted"
    default_w: float = 1.0  # weight of a family the fit never saw (the ridge prior: trust the back-tested LR)
    fitted: bool = False
    p0_clip: float = 1e-6  # logit(p0) is taken at p0 clipped to [c, 1 - c]; headroom_v1 uses 0.1 (val p0 has exact 0/1 scores)
    meta: dict = field(default_factory=dict)  # provenance (p0 file, table, n rows); part of the hash

    def _check(self, p0, X) -> tuple[np.ndarray, np.ndarray]:
        p0, X = np.asarray(p0, dtype="f8"), np.asarray(X, dtype="f8")
        if X.ndim == 1:
            X = X[:, None]
        if len(X) != len(p0) or X.shape[1] != len(self.families):
            raise ValueError("X needs one row per alert and one column per family")
        if np.isnan(p0).any() or np.isnan(X).any():
            raise ValueError("NaN in fusion input")
        return p0, X

    def fit(self, p0, X, y, families: Sequence[str], sample_weight=None) -> ConditionalFusion:
        self.families = list(families)
        p0, X = self._check(p0, X)
        y = np.asarray(y, dtype="f8")
        sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype="f8")
        D = np.column_stack([np.ones(len(y)), logit(p0), X])  # a, b, w...
        prior = np.r_[0.0, 1.0, np.ones(X.shape[1])]
        pen = np.r_[0.0, np.ones(1 + X.shape[1])] * self.ridge
        n = sw.sum()

        def loss(th):
            z = D @ th
            r = th - prior
            g = D.T @ (sw * (sigmoid(z) - y)) / n + pen * r / n
            return float((sw * (np.logaddexp(0.0, z) - y * z)).sum() / n + 0.5 * (pen * r * r).sum() / n), g

        res = minimize(loss, prior.copy(), jac=True, method="L-BFGS-B")
        if not res.success:
            raise RuntimeError(f"conditional fusion fit did not converge: {res.message}")
        self.a, self.b, self.w = float(res.x[0]), float(res.x[1]), res.x[2:].copy()
        self.fitted = True
        return self

    def logit_base(self, p0) -> np.ndarray:
        z0 = logit(np.clip(np.asarray(p0, dtype="f8"), self.p0_clip, 1 - self.p0_clip))
        return z0 if self.base == "p0" else self.a + self.b * z0

    def predict(self, p0, X) -> np.ndarray:
        p0, X = self._check(p0, X)
        w = self.w if self.fitted else np.ones(len(self.families))
        return sigmoid(self.logit_base(p0) + self.shrink * (X @ w))

    def max_weight(self) -> float:
        """Largest |weight| times the shrink: the `max_weight` of the stop rule (`ftm.decision.threshold.stop_bounds`)."""
        w = np.abs(self.w) if self.fitted else np.ones(len(self.families))
        return float(max(w.max(initial=0.0), abs(self.default_w))) * abs(self.shrink)

    def weight(self, family: str) -> float:
        if family in self.families and self.fitted:
            return float(self.w[self.families.index(family)])
        return self.default_w

    def params(self) -> dict:
        return {"kind": "conditional_fusion", "families": list(self.families), "a": self.a, "b": self.b,
                "w": [float(x) for x in self.w], "ridge": self.ridge, "shrink": self.shrink, "base": self.base,
                "default_w": self.default_w, "fitted": self.fitted, "meta": self.meta,
                **({"p0_clip": self.p0_clip} if self.p0_clip != 1e-6 else {})}  # fmt: skip  (absent = default: old hashes hold)

    def content_hash(self) -> str:
        return hashlib.blake2b(json.dumps(self.params(), sort_keys=True).encode(), digest_size=8).hexdigest()

    def to_json(self, path: str | Path) -> str:
        """Write the params plus their hash; returns the hash."""
        h = self.content_hash()
        Path(path).write_text(json.dumps({**self.params(), "hash": h}, indent=1, sort_keys=True))
        return h

    @classmethod
    def from_params(cls, p: dict) -> ConditionalFusion:
        if p.get("kind") != "conditional_fusion":
            raise ValueError("not a conditional_fusion params file")
        return cls(list(p["families"]), p["a"], p["b"], np.asarray(p["w"], dtype="f8"), p["ridge"], p["shrink"], p["base"],
                   p.get("default_w", 1.0), p["fitted"], p.get("p0_clip", 1e-6), dict(p.get("meta", {})))  # fmt: skip

    @classmethod
    def from_json(cls, path: str | Path) -> ConditionalFusion:
        p = json.loads(Path(path).read_text())
        m = cls.from_params(p)
        if m.content_hash() != p.get("hash"):
            raise ValueError(f"{path}: params do not match their hash (edited after fitting?)")
        return m


def fit_ridge_cv(p0, X, y, families: Sequence[str], ridges: Sequence[float] = RIDGES, n_folds: int = 5) -> ConditionalFusion:
    """Fit with the ridge chosen by contiguous-block CV Brier (rows in time order; blocks keep time neighbours together)."""
    p0, X, y = np.asarray(p0, dtype="f8"), np.asarray(X, dtype="f8"), np.asarray(y, dtype="f8")
    blocks = np.array_split(np.arange(len(y)), n_folds)
    score = {}
    for r in ridges:
        pred = np.empty(len(y))
        for te in blocks:
            tr = np.setdiff1d(np.arange(len(y)), te)
            pred[te] = ConditionalFusion(ridge=r).fit(p0[tr], X[tr], y[tr], families).predict(p0[te], X[te])
        score[r] = float(np.mean((pred - y) ** 2))
    best = min(ridges, key=lambda r: score[r])
    m = ConditionalFusion(ridge=best).fit(p0, X, y, families)
    m.meta["ridge_cv_brier"] = {f"{r:g}": s for r, s in score.items()}
    return m


def choose_shrink(model: ConditionalFusion, p0, X, y, shrinks: Sequence[float] = SHRINKS,
                  bases: Sequence[str] = BASES) -> dict:  # fmt: skip
    """Pick (shrink, base) by Brier on the rows given. Call it on val fit+cal alerts only; the rows it picks on are in-sample
    for the choice, so do not score the same rows. Sets model.shrink / model.base and returns the grid."""
    y = np.asarray(y, dtype="f8")
    grid = {}
    for base in bases:
        for s in shrinks:
            model.shrink, model.base = float(s), base
            grid[f"{base}/{s:g}"] = float(np.mean((model.predict(p0, X) - y) ** 2))
    best = min(grid, key=grid.get)  # ties go to the earlier (fitted base, smaller shrink first)
    model.base, s = best.split("/")
    model.shrink = float(s)
    model.meta["shrink_grid_brier"] = grid
    return grid


def alert_matrix(rows, table, p0_of: dict, families: Sequence[str] | None = None, draws: int = 1, seed: int = 0,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:  # fmt: skip
    """Per-alert design from back-test rows.  `rows`: dicts (alert_id, canonical, eps, answer_bin, label) of the questions
    in `table` (an `LRTable`; only accepted entries count).  Per (alert, question) one answered eps level is drawn among the
    accepted levels (a run never asks the same question twice); x_f sums that family's log-LRs.  Returns
    (p0, X, y, families, alert_ids), one row per askable alert per draw (alerts without a p0 dropped)."""
    from math import log

    acc: dict[tuple[str, str], dict[float, tuple[str, float]]] = {}
    labels: dict[str, bool] = {}  # every alert any question was put to, answered or not (they carry x = 0)
    for r in rows:
        labels[r["alert_id"]] = bool(r["label"])
        e = table.get(r["canonical"], float(r["eps"]))
        if e is None or not e.accepted or r.get("answer_bin") is None:
            continue
        lr = e.lr.get(r["answer_bin"])
        if lr is None or lr <= 0:
            continue
        acc.setdefault((r["alert_id"], r["canonical"]), {})[float(r["eps"])] = (e.family or "_unknown", log(lr))
    fams = sorted({v[0] for d in acc.values() for v in d.values()}) if families is None else list(families)
    fi = {f: j for j, f in enumerate(fams)}
    ids = [a for a in p0_of if a in labels]  # in the order of p0_of (time order for CV)
    ai = {a: i for i, a in enumerate(ids)}
    rng = np.random.default_rng(seed)
    X = np.zeros((draws * len(ids), len(fams)))  # row i*draws + k: draw k of alert i, so blocks stay in time order
    for (aid, _), d in sorted(acc.items()):
        if aid not in ai:
            continue
        levels = sorted(d)
        for k in range(draws):
            f, v = d[levels[int(rng.integers(len(levels)))]]
            if f in fi:
                X[ai[aid] * draws + k, fi[f]] += v
    ids_all = [a for a in ids for _ in range(draws)]
    p0 = np.array([p0_of[a] for a in ids_all])
    y = np.array([labels[a] for a in ids_all], dtype="f8")
    return p0, X, y, fams, ids_all
