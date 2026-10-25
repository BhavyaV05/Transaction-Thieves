"""RouteLLM / FrugalGPT-style routing between the fast tier and the 8B, and its comparison with the conformal cascade (E10).

Owner: T2 (Bhavya).  Spec: plan.md 7.3 and 8 (E10: "RouteLLM-style"), tracks.md 4.9, week 10.  Refs: Ong et al. RouteLLM 2024; Chen
et al. FrugalGPT 2023. The conformal cascade it is compared with is T3's `ftm/fastmodel/cascade.py`.

What a router is: a score for "the cheap model's answer is right", a threshold on it, and the dear model for everything below.
RouteLLM trains the score on preference data and picks the threshold so that a chosen share of calls goes to the strong model;
FrugalGPT scores the cheap answer and accepts it above a threshold. Here:

  learned   a ridge logistic regression on the fast tier's own confidence (top probability, margin to the second label,
            normalised entropy) and the decision type, fitted on the validation `fit` part to predict whether the fast argmax is right
  maxprob   no training: the score is the fast tier's top probability (FrugalGPT's confidence rule)

The threshold is the quantile of the score on the `cal` part that sends a chosen share of rows to the 8B, so the cost is set by
the operator, as in RouteLLM. Differences from the conformal cascade, which the report keeps visible: the router always answers
(it never abstains to a safe default) and has no coverage guarantee, while the conformal cascade bounds the marginal error of an
accepted answer by 2 alpha and returns the decision's safe default when neither tier is sure. `route_report` puts them side by
side at the same share of rows not settled by the fast tier. Same split discipline as `ftm/fastmodel/splits.py`: temperature and
router on `fit`, thresholds and conformal sets on `cal`, every reported number on `eval`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from ftm.fastmodel.calibration import fit_temperature, softmax_t
from ftm.fastmodel.cascade import CascadeDecision, ConformalCascade, ConformalSet, evaluate
from ftm.fastmodel.decisions import DECISIONS, DecisionExample
from ftm.fastmodel.splits import PARTS, check_parts

KINDS = ("learned", "maxprob")


def confidence_features(probs: np.ndarray) -> np.ndarray:
    """(top probability, margin to the runner-up, entropy / log k) of one label distribution."""
    p = np.sort(np.asarray(probs, "f8"))[::-1]
    k = len(p)
    ent = -float(np.sum(p * np.log(np.clip(p, 1e-12, 1.0)))) / np.log(k) if k > 1 else 0.0
    return np.array([p[0], p[0] - (p[1] if k > 1 else 0.0), ent])


@dataclass
class ConfidenceRouter:
    kind: str = "learned"
    ridge: float = 1.0
    decisions: list[str] = field(default_factory=list)
    w: np.ndarray = field(default_factory=lambda: np.zeros(0))
    threshold: float = 0.0  # defer when score < threshold

    def _x(self, decision: str, probs: np.ndarray) -> np.ndarray:
        onehot = np.array([float(decision == d) for d in self.decisions])
        return np.concatenate([[1.0], confidence_features(probs), onehot])

    def fit(self, decisions: Sequence[str], probs: Sequence[np.ndarray], correct: Sequence[bool]) -> ConfidenceRouter:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        self.decisions = sorted(set(decisions))
        if self.kind == "maxprob" or not len(decisions):
            return self
        X = np.stack([self._x(d, p) for d, p in zip(decisions, probs, strict=True)])
        y = np.asarray(correct, "f8")
        pen = np.r_[0.0, np.full(X.shape[1] - 1, self.ridge)]  # the intercept is free

        def loss(w):
            z = X @ w
            ll = np.logaddexp(0.0, z) - y * z
            g = X.T @ (1 / (1 + np.exp(-z)) - y) / len(y) + pen * w / len(y)
            return float(ll.mean() + 0.5 * (pen * w * w).sum() / len(y)), g

        res = minimize(loss, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B")
        self.w = res.x
        return self

    def score(self, decision: str, probs: np.ndarray) -> float:
        """Estimated probability that the fast argmax is right (maxprob: the top probability itself)."""
        if self.kind == "maxprob":
            return float(np.max(probs))
        return float(1 / (1 + np.exp(-self._x(decision, probs) @ self.w)))

    def set_share(self, scores_cal: Sequence[float], large_share: float) -> float:
        """Threshold that sends `large_share` of the calibration rows to the large tier."""
        s = np.sort(np.asarray(scores_cal, "f8"))
        if not len(s) or large_share <= 0:
            self.threshold = -np.inf
        elif large_share >= 1:
            self.threshold = np.inf
        else:
            k = int(round(large_share * len(s)))  # the k lowest scores defer
            self.threshold = float(s[k - 1] + 1e-12) if k > 0 else -np.inf
        return self.threshold

    def defers(self, decision: str, probs: np.ndarray) -> bool:
        return self.score(decision, probs) < self.threshold

    def decide(self, decision: str, fast_probs: dict[str, float], large_probs_fn) -> CascadeDecision:
        """Same result type as the conformal cascade, so `ftm.fastmodel.cascade.evaluate` reads it. Never `default`."""
        labels = DECISIONS[decision].labels
        p = np.array([fast_probs[x] for x in labels])
        if not self.defers(decision, p):
            return CascadeDecision(decision, labels[int(p.argmax())], "fast")
        lp = large_probs_fn()
        return CascadeDecision(decision, max(lp, key=lp.get), "large")


def _error(answers: Sequence[str], labels: Sequence[str]) -> float:
    return float(np.mean([a != y for a, y in zip(answers, labels, strict=True)])) if len(labels) else float("nan")


def route_report(examples: Sequence[DecisionExample], parts: Sequence[str], fast_logps: Sequence[np.ndarray],
                 large_probs: Sequence[dict | None], alpha: float = 0.05, kinds: Sequence[str] = KINDS,
                 shares: Sequence[float] = tuple(np.round(np.linspace(0, 1, 11), 2)), ridge: float = 1.0) -> dict:  # fmt: skip
    """Router cascades against the conformal cascade on the validation parts. `fast_logps[i]` is the fast tier's label
    log-probabilities for row i, `large_probs[i]` the 8B's label -> probability (None where it was not scored: only `fit` rows
    may lack it). Returns, per router kind, the error-versus-share curve on `eval`, and the matched comparison."""
    check_parts(examples, parts)
    exs, parts = list(examples), list(parts)
    if len(fast_logps) != len(exs) or len(large_probs) != len(exs):
        raise ValueError("one fast and one large entry per example")
    idx = {p: [i for i, q in enumerate(parts) if q == p] for p in PARTS}
    for p in ("cal", "eval"):
        if any(large_probs[i] is None for i in idx[p]):
            raise ValueError(f"the 8B must be scored on every {p} row")
    decisions = sorted({e.decision for e in exs})
    temps = {d: 1.0 for d in decisions}
    for d in decisions:
        rows = [i for i in idx["fit"] if exs[i].decision == d]
        if rows:
            y = np.array([DECISIONS[d].labels.index(exs[i].label) for i in rows])
            temps[d] = fit_temperature(np.stack([np.asarray(fast_logps[i], "f8") for i in rows]), y)

    def fast_p(i: int) -> np.ndarray:
        return softmax_t(np.asarray(fast_logps[i], "f8")[None, :], temps[exs[i].decision])[0]

    def as_dict(i: int) -> dict[str, float]:
        return dict(zip(DECISIONS[exs[i].decision].labels, fast_p(i).tolist(), strict=True))

    # the conformal cascade, as T3 builds it
    names_cal = [exs[i].decision for i in idx["cal"]]
    y_cal = [exs[i].label for i in idx["cal"]]
    cascade = ConformalCascade(ConformalSet(alpha).fit(names_cal, [as_dict(i) for i in idx["cal"]], y_cal),
                               ConformalSet(alpha).fit(names_cal, [large_probs[i] for i in idx["cal"]], y_cal))  # fmt: skip

    def run_conformal(rows):
        return [cascade.decide(exs[i].decision, as_dict(i), lambda i=i: large_probs[i]) for i in rows]

    cal_res, ev_res = run_conformal(idx["cal"]), run_conformal(idx["eval"])
    y_ev = [exs[i].label for i in idx["eval"]]
    unsettled_cal = float(np.mean([r.source != "fast" for r in cal_res])) if cal_res else float("nan")
    conf_by = evaluate(ev_res, y_ev, alpha)
    n_ev = len(ev_res)
    wrong = sum(r.answer != y for r, y in zip(ev_res, y_ev, strict=True) if r.source in ("fast", "large"))
    conformal = {"unsettled_share_cal": unsettled_cal, "kept_fast": float(np.mean([r.source == "fast" for r in ev_res])) if n_ev else None,
                 "to_large": float(np.mean([r.source == "large" for r in ev_res])) if n_ev else None,
                 "to_default": float(np.mean([r.source == "default" for r in ev_res])) if n_ev else None,
                 "error_accepted_and_wrong": wrong / n_ev if n_ev else None,  # the quantity bounded by 2 alpha; a default is not "wrong"
                 # a default counted as the answer it is (the decision's safe label), the like-for-like number against a router
                 "error_default_as_answer": _error([r.answer for r in ev_res], y_ev), "by_decision": conf_by}  # fmt: skip

    fast_ans = [DECISIONS[exs[i].decision].labels[int(fast_p(i).argmax())] for i in idx["eval"]]
    large_ans = [max(large_probs[i], key=large_probs[i].get) for i in idx["eval"]]
    out = {"alpha": alpha, "temperatures": temps, "rows": {p: len(idx[p]) for p in PARTS}, "conformal": conformal,
           "fast_only_error": _error(fast_ans, y_ev), "large_only_error": _error(large_ans, y_ev), "routers": {}}  # fmt: skip
    for kind in kinds:
        fit_rows = idx["fit"]
        router = ConfidenceRouter(kind, ridge).fit([exs[i].decision for i in fit_rows], [fast_p(i) for i in fit_rows],
                                                   [DECISIONS[exs[i].decision].labels[int(fast_p(i).argmax())] == exs[i].label for i in fit_rows])  # fmt: skip
        s_cal = [router.score(exs[i].decision, fast_p(i)) for i in idx["cal"]]
        s_ev = np.array([router.score(exs[i].decision, fast_p(i)) for i in idx["eval"]])

        def at_share(share: float, router=router, s_cal=s_cal) -> dict:
            router.set_share(s_cal, share)
            defer = np.array([router.defers(exs[i].decision, fast_p(i)) for i in idx["eval"]])
            ans = [la if d else fa for d, fa, la in zip(defer, fast_ans, large_ans, strict=True)]
            return {"target_share": float(share), "large_share": float(defer.mean()) if len(defer) else None, "error": _error(ans, y_ev)}

        curve = [at_share(s) for s in shares]
        matched = at_share(unsettled_cal)
        out["routers"][kind] = {
            "auc_fast_correct_eval": _auc(s_ev, [a == y for a, y in zip(fast_ans, y_ev, strict=True)]),
            "curve": curve, "matched_to_conformal": matched,
            "router_beats_conformal_at_matched_share": (None if math.isnan(conformal["error_default_as_answer"])
                                                        else bool(matched["error"] < conformal["error_default_as_answer"])),
        }  # fmt: skip
    return out


def _auc(scores: np.ndarray, positive: Sequence[bool]) -> float | None:
    """Rank AUC of `scores` for predicting `positive` (does the router's score separate right from wrong fast answers?)."""
    pos = np.asarray(positive, bool)
    if pos.all() or not pos.any():
        return None
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    s = scores[order]
    i = 0
    while i < len(s):  # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
