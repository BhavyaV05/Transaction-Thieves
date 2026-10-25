"""C9: the learned policy. A gradient-boosted model predicts the Brier gain of asking a question, the planner asks the best.

Owner: T2 (Bhavya).  Spec: plan.md 6.10, tracks.md 4.7 ("gradient-boosted ranker trained on validation to predict Brier
gain from (alert features, question)"), week 9.

Training data come from the simulator, on the `fit` half (in time) of the validation alerts; the other half is where every
planner is scored (`split_val`). For each alert, `ftm.orchestrator.policy_data.collect_policy_data` walks a short random rollout (up to `depth` questions
chosen at random among the affordable ones) and, at every state, asks EVERY affordable question (up to `max_options`) of the
partner bank. The label of a (state, question) row is the Brier gain that answer actually gave,
`(p - y)^2 - (p' - y)^2`, with p the posterior before and p' after, y the alert label. The model predicts that gain from the
state and the question, so ranking by prediction / eps is a planner and a prediction <= `min_gain` is a stop rule.

The prediction is the closed-form expected gain of `EIGPlanner` (the `eig` feature) plus a boosted correction: the trees fit
the residual, so the model learns where the back-tested LRs are optimistic or pessimistic (a family whose LR was measured on
few answers, a question that is often declined) and zero rounds IS the EIG planner. The number of rounds is chosen by
cross-validation over whole alerts. With little data it stops early and behaves like EIG; that is the intended failure mode.

Those questions are asked of the banks without the harness, as the back-test does, so they are charged to eps_cal (summed in
`PolicyData.eps_cal`), never to an alert. Labels are validation labels; the test split is never read. The collector lives in
`ftm/orchestrator/policy_data.py`: nothing under `ftm/planners` may import a bank (tests/test_harness.py).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ftm.fusion.fuse import logit
from ftm.planners.pool import Candidate
from ftm.planners.session import PlanSession
from ftm.planners.simple import MIN_GAIN, _Base, expected_brier_gain

KINDS = ("bool", "count", "kary", "set_count")
FEATURES = ("logit_p", "p0", "eps_left", "n_asked", "direction_out", "log_amount", "partner_bank", "eps", "kind",
            "family", "sensitivity", "n_cells", "max_abs_log_lr", "eig", "eig_per_eps")  # fmt: skip
CATEGORICAL = ("kind", "family", "partner_bank")
EIG_COL = FEATURES.index("eig")
PARAMS = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 40,
          "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "cat_smooth": 20.0,
          "min_data_per_group": 40, "verbose": -1, "deterministic": True, "num_threads": 1}  # fmt: skip


def split_val(alerts: Sequence, fit_frac: float = 0.5) -> tuple[list, list]:
    """The validation alerts in time order, cut into `fit` (the first `fit_frac`, where the policy is trained) and `eval`
    (the rest, where every planner is scored). The same cut as T1's `val_part` fit / (cal + eval) split, so a later
    estimate never trains on its own future."""
    ordered = sorted(alerts, key=lambda a: (a.binding.t0, a.alert_id))
    cut = int(len(ordered) * fit_frac)
    return ordered[:cut], ordered[cut:]


class Featurizer:
    """(state, question) -> the numbers the model sees. `families` is the vocabulary seen in training; a family it has not
    seen gets code -1, which LightGBM reads as missing."""

    def __init__(self, families: Sequence[str] = ()):
        self.families = list(families)
        self._fam = {f: i for i, f in enumerate(self.families)}

    def row(self, s: PlanSession, c: Candidate) -> list[float]:
        return self.row_from(s.p, float(s.alert.p0), s.eps_left, len(s.families_asked), s.direction, s.bank,
                             float(s.alert.summary.get("amount") or 0.0), c)  # fmt: skip

    def row_from(self, p: float, p0: float, eps_left: float, n_asked: int, direction: str, bank: int, amount: float,
                 c: Candidate) -> list[float]:  # fmt: skip
        gain = expected_brier_gain(p, c)
        lrs = [abs(math.log(v)) for v in c.lr.values() if v > 0]
        half_decades = round(2 * math.log10(1.0 + max(amount, 0.0))) / 2  # coarse on purpose: an exact amount is an alert id
        return [float(logit(p)), p0, eps_left, float(n_asked), 1.0 if direction == "out" else 0.0,
                half_decades, float(bank), c.rank_cost, float(KINDS.index(c.kind) if c.kind in KINDS else -1),
                float(self._fam.get(c.family, -1)), float(c.sensitivity), float(len(c.lr)), max(lrs, default=0.0),
                gain, gain / c.rank_cost]  # fmt: skip

    def to_dict(self) -> dict:
        return {"families": self.families}


@dataclass
class PolicyData:
    X: np.ndarray
    y: np.ndarray  # realised Brier gain of the answer
    alert_ids: list[str]
    families: list[str]
    eps_cal: float  # eps of the questions asked to build the rows (back-test accounting)
    n_queries: int
    n_declined: int = 0


@dataclass
class LearnedPolicy:
    """The fitted regression and the featurizer that goes with it."""

    featurizer: Featurizer
    model: object  # lightgbm.Booster
    meta: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, data: PolicyData, params: dict | None = None, rounds: int = 300, seed: int = 0, n_folds: int = 5,
            patience: int = 20) -> LearnedPolicy:  # fmt: skip
        """Boost the residual of the realised gain over EIG's expected gain (`init_score`), so zero rounds IS the EIG planner.
        The number of rounds is chosen by cross-validation over alerts (whole alerts held out, never rows of one alert split
        between fitting and checking), stopping after `patience` rounds without improvement."""
        import lightgbm as lgb

        if len(data.y) < 2:
            raise ValueError("no policy rows: nothing to fit (is the pool empty, or are no partner banks reachable?)")
        p = {**PARAMS, "seed": seed, **(params or {})}
        eig = data.X[:, EIG_COL]
        ds = lgb.Dataset(data.X, data.y, init_score=eig, feature_name=list(FEATURES), categorical_feature=list(CATEGORICAL),
                         free_raw_data=False)  # fmt: skip
        alerts = np.array(data.alert_ids)
        uniq = np.unique(alerts)
        used, cv_l2 = rounds, None
        if len(uniq) >= 2 * n_folds:
            fold_of = dict(zip(np.random.default_rng(seed).permutation(uniq), np.arange(len(uniq)) % n_folds, strict=True))
            fold = np.array([fold_of[a] for a in alerts])
            folds = [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(n_folds)]
            res = lgb.cv(p, ds, num_boost_round=rounds, folds=folds, callbacks=[lgb.early_stopping(patience, verbose=False)])
            curve = next(iter(res.values()))
            used, cv_l2 = max(1, len(curve)), float(min(curve))
        booster = lgb.train(p, ds, num_boost_round=used)
        meta = {"rows": int(len(data.y)), "alerts": len(uniq), "mean_gain": float(data.y.mean()), "eps_cal": data.eps_cal,
                "queries": data.n_queries, "declined": data.n_declined, "rounds": used, "cv_l2": cv_l2,
                "l2_eig_only": float(np.mean((data.y - eig) ** 2)), "seed": seed}  # fmt: skip
        return cls(Featurizer(data.families), booster, meta)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """EIG's expected gain plus the learned correction."""
        X = np.asarray(X, dtype="f8").reshape(-1, len(FEATURES))
        return X[:, EIG_COL] + np.asarray(self.model.predict(X))

    def importance(self) -> dict:
        gain = self.model.feature_importance(importance_type="gain")
        total = float(sum(gain)) or 1.0
        return {n: float(g) / total for n, g in zip(FEATURES, gain, strict=True)}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": self.model.model_to_string(), "featurizer": self.featurizer.to_dict(),
                                    "meta": self.meta}), encoding="utf-8")  # fmt: skip
        return path

    @classmethod
    def load(cls, path: str | Path) -> LearnedPolicy:
        import lightgbm as lgb

        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(Featurizer(d["featurizer"]["families"]), lgb.Booster(model_str=d["model"]), d.get("meta", {}))


@dataclass
class LearnedPolicyPlanner(_Base):
    """Asks the question with the largest predicted gain per eps; stops when no question is predicted to help."""

    policy: LearnedPolicy | None = None
    min_gain: float = MIN_GAIN
    name: str = "learned"

    def choose(self, s: PlanSession, options: list[Candidate]) -> Candidate | None:
        if self.policy is None:
            raise ValueError("LearnedPolicyPlanner needs a fitted policy (scripts/train_policy.py)")
        if not options:
            return None
        X = np.array([self.policy.featurizer.row(s, c) for c in options])
        pred = self.policy.predict(X)
        scored = [(float(g) / c.rank_cost, c.program, -c.rank_cost, i) for i, (g, c) in enumerate(zip(pred, options, strict=True))]
        best = max(scored)
        i = best[3]
        return options[i] if pred[i] > self.min_gain else None
