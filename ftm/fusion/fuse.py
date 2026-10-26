"""C8 evidence fusion, v0: logit p = a*logit(p0) + sum_f w_f * x_f + c, where x_f is the summed log-LR of the
answers of question family f for that alert. Only numbers go in: the one-bank score and back-tested
log-likelihood ratios. No LLM text, no reply notes (plan.md 6.9; invariant "LLM text never enters").

Owner: T4 (Shreyas).  Spec: plan.md 6.9; tracks.md 6.4.

Modes:
- naive:      a = 1, w = 1, c = 0. Bayes' rule with independent answers (the "summed log-LRs" ablation).
- shared:     one weight for all families, fitted with a and c (v0).
- per_family: one weight per family (v1, week 7). Already supported: weights are ridge-shrunk toward the
              naive value 1, so a family with little data stays close to "trust the back-tested LR".
Fitted modes are fit jointly on validation alerts, so overlapping questions and the base score are not
double-counted. A family not seen in fitting gets the shared prior weight (1 after shrinkage).

v1 (week 10): `prior`. The ridge pulls a toward 1 in both cases; it pulls the weights w toward `prior`:
- "naive" (default, unchanged): w -> 1, "trust the back-tested LR unless validation says otherwise".
- "zero":  w -> 0, "ignore the evidence unless validation supports it". The right prior if naive fusion hurts
           (G2 on real data: docs/t4_fusion_findings.md). With a huge ridge the model is p0 up to the free intercept c.
`fit_cv` picks (ridge, prior) by cluster-grouped cross-validated Brier inside the fit set. The grid runs from weak to
strong shrinkage under both priors; its strongest ridge is close to naive fusion under "naive" and close to p0 under
"zero", so the choice is "use the evidence, partly use it, or leave p0 alone". The hyperparameters are chosen on validation only (never
test), and the chosen setting must be pre-registered in prereg/ before G3.

Input format for answers (proposal for hand-off #9 with T2): one row per answered question,
(alert_id, family, log_lr), where log_lr = log LR_f(answer) from the back-test table, already clipped at the
privacy ceiling. Declines and timeouts contribute nothing (no row).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

P_CLIP = 1e-6
PRIORS = ("naive", "zero")
RIDGES = (1.0, 10.0, 100.0, 1000.0)  # pseudo-observations: the ridge is divided by n like the data term


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype="f8"), P_CLIP, 1 - P_CLIP)
    return np.log(p / (1 - p))


def sigmoid(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype="f8")))


def _numeric(x, name: str) -> np.ndarray:
    a = np.asarray(x)
    if not np.issubdtype(a.dtype, np.number) and a.dtype != bool:
        raise TypeError(f"{name} must be numeric (got {a.dtype}); text never enters fusion")
    a = a.astype("f8")
    if np.isnan(a).any():
        raise ValueError(f"{name} contains NaN")
    return a


def answer_features(alert_ids: Sequence, ans_alert_id: Sequence, ans_family: Sequence, ans_log_lr,
                    families: Sequence | None = None) -> tuple[np.ndarray, list]:  # fmt: skip
    """Sum log-LRs per (alert, family). Returns X (n_alerts x n_families) and the family order.

    Alerts with no answers get a zero row. Families not in `families` (when given) raise, so a typo in a
    family name cannot silently drop evidence.
    """
    llr = _numeric(ans_log_lr, "log_lr")
    fams = list(dict.fromkeys(ans_family)) if families is None else list(families)
    fi = {f: j for j, f in enumerate(fams)}
    ai = {a: i for i, a in enumerate(alert_ids)}
    if len(ai) != len(alert_ids):
        raise ValueError("alert_ids must be unique")
    X = np.zeros((len(alert_ids), len(fams)))
    for a, f, v in zip(ans_alert_id, ans_family, llr, strict=True):
        if a not in ai:
            raise KeyError(f"answer for unknown alert {a!r}")
        if f not in fi:
            raise KeyError(f"unknown family {f!r}")
        X[ai[a], fi[f]] += v
    return X, fams


@dataclass
class Fusion:
    mode: str = "naive"
    families: list = field(default_factory=list)
    a: float = 1.0
    w: np.ndarray = field(default_factory=lambda: np.zeros(0))
    c: float = 0.0
    ridge: float = 1.0
    fitted: bool = False
    prior: str = "naive"  # where the ridge pulls the weights w: "naive" (1) or "zero" (0)

    def _check(self, p0, X):
        p0 = _numeric(p0, "p0")
        X = _numeric(X, "X")
        if X.ndim == 1:
            X = X[:, None]
        if len(X) != len(p0):
            raise ValueError("p0 and X need one row per alert")
        if self.families and X.shape[1] != len(self.families):
            raise ValueError(f"X has {X.shape[1]} family columns, model has {len(self.families)}")
        return p0, X

    def _weights(self, k: int) -> np.ndarray:
        if self.mode == "naive" or not self.fitted:
            return np.ones(k)
        if self.mode == "shared":
            return np.full(k, float(self.w[0]))
        return self.w

    def fit(self, p0, X, y, sample_weight=None) -> Fusion:
        """Penalised log-loss fit of (a, w, c). Ridge pulls a toward 1 and w toward 1 (prior "naive", Bayes) or
        0 (prior "zero", ignore the evidence); c is free."""
        p0, X = self._check(p0, X)
        y = _numeric(y, "y")
        if self.prior not in PRIORS:
            raise ValueError(f"prior must be one of {PRIORS}, got {self.prior!r}")
        if self.mode == "naive":
            self.fitted = True
            return self
        if self.mode not in ("shared", "per_family"):
            raise ValueError(f"unknown mode {self.mode!r}")
        sw = np.ones(len(y)) if sample_weight is None else _numeric(sample_weight, "sample_weight")
        z0 = logit(p0)
        F = X.sum(axis=1, keepdims=True) if self.mode == "shared" else X
        D = np.column_stack([z0, F, np.ones(len(y))])  # columns: a, w..., c
        prior = np.r_[1.0, np.full(F.shape[1], 1.0 if self.prior == "naive" else 0.0), 0.0]
        pen = np.r_[np.ones(1 + F.shape[1]), 0.0] * self.ridge
        n = sw.sum()

        def loss(theta):
            z = D @ theta
            # log(1 + e^z) - y z, stable
            ll = np.logaddexp(0.0, z) - y * z
            r = theta - prior
            g = D.T @ (sw * (sigmoid(z) - y)) / n + pen * r / n
            return float((sw * ll).sum() / n + 0.5 * (pen * r * r).sum() / n), g

        res = minimize(loss, prior.copy(), jac=True, method="L-BFGS-B")
        if not res.success:
            raise RuntimeError(f"fusion fit did not converge: {res.message}")
        self.a, self.w, self.c = float(res.x[0]), res.x[1:-1].copy(), float(res.x[-1])
        self.fitted = True
        return self

    def predict(self, p0, X) -> np.ndarray:
        p0, X = self._check(p0, X)
        return sigmoid(self.a * logit(p0) + X @ self._weights(X.shape[1]) + self.c)

    def params(self) -> dict:
        w = [1.0] if self.mode == "naive" or not self.fitted else self.w.tolist()
        return {"mode": self.mode, "families": list(self.families), "a": self.a, "w": w, "c": self.c,
                "ridge": self.ridge, "prior": self.prior, "fitted": self.fitted}  # fmt: skip


def posterior(p0: float, log_lrs_by_family: dict, model: Fusion | None = None) -> float:
    """Running posterior for the planner's stop rule: p0 plus the answers so far, one alert."""
    model = model or Fusion("naive")
    fams = model.families or list(log_lrs_by_family)
    unknown = set(log_lrs_by_family) - set(fams)
    if model.families and unknown:
        raise KeyError(f"families not in the model: {sorted(unknown)}")
    x = np.array([[float(log_lrs_by_family.get(f, 0.0)) for f in fams]]) if fams else np.zeros((1, 0))
    return float(model.predict([p0], x)[0])


def shuffled_answers_test(p0, X, y, mode: str = "shared", folds=None, n_folds: int = 5, seed: int = 0,
                          ridge: float = 1.0) -> dict:  # fmt: skip
    """Leakage check (tracks.md 6.4): permute answer rows across alerts; the Brier gain should go to ~0.

    Cross-fitted: fit on k-1 folds, score the held-out fold, for both the real and the shuffled answers,
    so in-sample optimism does not fake a gain. Pass `folds` (one fold label per alert, e.g. from
    clusters) to keep related alerts in one fold. Gain = Brier(p0) - Brier(fused); positive = helps.
    """
    p0, X, y = _numeric(p0, "p0"), _numeric(X, "X"), _numeric(y, "y")
    X = X[:, None] if X.ndim == 1 else X
    rng = np.random.default_rng(seed)
    if folds is None:
        folds = rng.integers(0, n_folds, len(y))
    folds = np.asarray(folds)
    Xs = X[rng.permutation(len(y))]
    out = {}
    for name, feats in (("real", X), ("shuffled", Xs)):
        pred = np.empty(len(y))
        for f in np.unique(folds):
            tr, te = folds != f, folds == f
            m = Fusion(mode, ridge=ridge).fit(p0[tr], feats[tr], y[tr])
            pred[te] = m.predict(p0[te], feats[te])
        out[f"gain_{name}"] = float(np.mean((p0 - y) ** 2) - np.mean((pred - y) ** 2))
    return out


def fold_labels(clusters, n_folds: int, seed: int) -> np.ndarray:
    """A fold per alert, with every cluster whole in one fold (related alerts are never split between fit and score)."""
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    k = max(1, min(n_folds, len(uniq)))
    fold_of = dict(zip(np.random.default_rng(seed).permutation(uniq), np.arange(len(uniq)) % k, strict=True))
    return np.array([fold_of[c] for c in clusters])


def fit_cv(p0, X, y, clusters, mode: str = "per_family", ridges: Sequence[float] = RIDGES,
           priors: Sequence[str] = PRIORS, n_folds: int = 5, seed: int = 0, families: Sequence | None = None,
           ) -> tuple[Fusion, dict]:  # fmt: skip
    """Fit `mode` fusion with (ridge, prior) chosen by cluster-grouped cross-validated Brier inside the fit set.

    Every (prior, ridge) of the grid is fitted on k-1 folds of whole clusters and scored on the held-out fold;
    the pair with the lowest pooled held-out Brier wins (ties go to the earlier pair of the grid, so list the
    cautious setting first), and is then refitted on all the rows given. The rows given are the only data used,
    so call it on validation alerts only: never on test, and never with the rows it will be scored on (for an
    honest score on validation, wrap it in an outer cross-fit: `ftm.eval.g2.crossfit_posterior(select="cv")`).
    The chosen setting must be pre-registered in prereg/ before G3.

    Returns (fitted Fusion, info) with info = {"ridge", "prior", "n_folds", "cv_brier": {"prior/ridge": Brier}}.
    """
    p0, X, y = _numeric(p0, "p0"), _numeric(X, "X"), _numeric(y, "y")
    X = X[:, None] if X.ndim == 1 else X
    folds = fold_labels(clusters, n_folds, seed)
    nf = len(np.unique(folds))
    if nf < 2:
        raise ValueError("fit_cv needs at least two clusters")
    grid = [(pr, r) for pr in priors for r in ridges]
    scores: dict[str, float] = {}
    for pr, r in grid:
        pred = np.empty(len(y))
        for f in np.unique(folds):
            tr, te = folds != f, folds == f
            pred[te] = Fusion(mode, ridge=r, prior=pr).fit(p0[tr], X[tr], y[tr]).predict(p0[te], X[te])
        scores[f"{pr}/{r:g}"] = float(np.mean((pred - y) ** 2))
    best_pr, best_r = min(grid, key=lambda g: scores[f"{g[0]}/{g[1]:g}"])  # min keeps the first of equal scores
    model = Fusion(mode, families=list(families or []), ridge=best_r, prior=best_pr).fit(p0, X, y)
    return model, {"ridge": float(best_r), "prior": best_pr, "n_folds": nf, "cv_brier": scores}
