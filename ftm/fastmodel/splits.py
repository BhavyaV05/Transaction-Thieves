"""Which validation part fits what in the fast-model cascade (C13). T4 review of PR #10.

Owner: T3 (Akshat).  Spec: plan.md 6.14; part definitions: ftm/baselines/README.md (T1, `val_part`).

T1 cuts the val time span into `fit` (first half), `cal` (next quarter) and `eval` (last quarter). Here:
  fit   the temperature of each decision type (the only thing fitted by gradient / likelihood)
  cal   the conformal sets of BOTH tiers (fast model and 8B), so their q is not tuned on rows that set T
  eval  every number reported; nothing above has seen these rows
The 8B tier's probabilities are computed on the same `cal` rows as the fast model's. Rows from train or test
never enter: a part outside fit/cal/eval raises. A decision type with too few `cal` rows falls back to the full
label set (it always defers); the report flags it, with n per part, so a thin type is not read as a result.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ftm.fastmodel.calibration import fit_temperature, softmax_t, top_label_ece
from ftm.fastmodel.cascade import ConformalCascade, ConformalSet, evaluate
from ftm.fastmodel.decisions import DECISIONS, DecisionExample
from ftm.fastmodel.model import LogprobFastModel

PARTS = ("fit", "cal", "eval")


def check_parts(examples: Sequence[DecisionExample], parts: Sequence[str]) -> None:
    if len(examples) != len(parts):
        raise ValueError("one val part per example")
    bad = sorted(set(parts) - set(PARTS))
    if bad:
        raise ValueError(f"val part must be one of {PARTS}, got {bad} (train and test rows never enter calibration)")
    for ex in examples:
        if ex.label not in DECISIONS[ex.decision].labels:
            raise ValueError(f"{ex.decision}: example {ex.alert_id!r} has label {ex.label!r}, not one of its labels")


def _majority_acc(fit_labels: Sequence[str], labels: Sequence[str]) -> float:
    """Accuracy of the constant label that is the most common in `fit_labels`, on `labels`."""
    if not fit_labels or not labels:
        return float("nan")
    top = max(set(fit_labels), key=list(fit_labels).count)
    return sum(y == top for y in labels) / len(labels)


def calibrate_cascade(examples: Sequence[DecisionExample], parts: Sequence[str], fast: LogprobFastModel,
                      large, alpha: float = 0.05, retire_margin: float = 0.0) -> tuple[dict[str, float], ConformalCascade, dict]:  # fmt: skip
    """Fit T per decision on `fit`, both conformal tiers on `cal`, report on `eval`. Returns (temps, cascade, report).

    `large`: a LogprobFastModel-like (`raw_logps`: its own T is fitted on `fit`, like the fast tier's) or a `predict_proba`
    model (the E10 stated-confidence baseline, no T). A tier whose argmax accuracy on `cal` does not exceed the majority label
    of `fit` by more than `retire_margin` is RETIRED for that decision (`cascade.fast_off` / `large_off`): its conformal set can
    only be a singleton by luck of the majority, so the decision goes straight to the next tier. Eval plays no part in it."""
    check_parts(examples, parts)
    rows = {p: [ex for ex, q in zip(examples, parts, strict=True) if q == p] for p in PARTS}
    raw = {p: [fast.raw_logps(ex) for ex in rows[p]] for p in PARTS}  # one fast forward pass per row
    decisions = sorted({ex.decision for ex in examples})
    logit_large = hasattr(large, "raw_logps")
    raw_l = {p: [large.raw_logps(ex) for ex in rows[p]] for p in PARTS} if logit_large else None

    def probs(ex, logps, temps):
        d = DECISIONS[ex.decision]
        return dict(zip(d.labels, softmax_t(logps[None, :], temps[ex.decision])[0].tolist(), strict=True))

    def fit_temps(raws):
        t = {d: 1.0 for d in decisions}
        for d in decisions:
            idx = [i for i, ex in enumerate(rows["fit"]) if ex.decision == d]
            if idx:
                y = np.array([DECISIONS[d].labels.index(rows["fit"][i].label) for i in idx])
                t[d] = fit_temperature(np.stack([raws["fit"][i] for i in idx]), y)
        return t

    temps = fit_temps(raw)
    temps_l = fit_temps(raw_l) if logit_large else {d: 1.0 for d in decisions}
    cal = rows["cal"]
    fast_cal = [probs(ex, lp, temps) for ex, lp in zip(cal, raw["cal"], strict=True)]
    large_cal = ([probs(ex, lp, temps_l) for ex, lp in zip(cal, raw_l["cal"], strict=True)] if logit_large
                 else [large.predict_proba(ex) for ex in cal])  # the large tier on the same cal rows as the fast model
    names, labels = [ex.decision for ex in cal], [ex.label for ex in cal]
    fast_sets = ConformalSet(alpha).fit(names, fast_cal, labels)
    large_sets = ConformalSet(alpha).fit(names, large_cal, labels)

    def argmax_acc(d, plist):
        idx = [i for i, ex in enumerate(cal) if ex.decision == d]
        return (sum(max(plist[i], key=plist[i].get) == cal[i].label for i in idx) / len(idx)) if idx else float("nan")

    fast_off, large_off, retire = set(), set(), {}
    for d in decisions:
        maj = _majority_acc([ex.label for ex in rows["fit"] if ex.decision == d], [ex.label for ex in cal if ex.decision == d])
        fa, la = argmax_acc(d, fast_cal), argmax_acc(d, large_cal)
        retire[d] = {"cal_majority_acc": maj, "cal_fast_acc": fa, "cal_large_acc": la}
        if not fa > maj + retire_margin:
            fast_off.add(d)
        if not la > maj + retire_margin:
            large_off.add(d)
    cascade = ConformalCascade(fast_sets, large_sets, fast_off, large_off)

    ev = rows["eval"]
    results = [cascade.decide(ex.decision, probs(ex, lp, temps),
                              (lambda ex=ex, lp=lp: probs(ex, lp, temps_l)) if logit_large else (lambda ex=ex: large.predict_proba(ex)))
               for ex, lp in zip(ev, raw_l["eval"] if logit_large else raw["eval"], strict=True)]  # fmt: skip
    by = evaluate(results, [ex.label for ex in ev], alpha)
    report = {}
    for d in decisions:
        labels_d = DECISIONS[d].labels
        i_ev = [i for i, ex in enumerate(ev) if ex.decision == d]
        row = {"n_fit": sum(ex.decision == d for ex in rows["fit"]), "n_cal": int(fast_sets.n.get(d, 0)),
               "n_eval": len(i_ev), "temperature": temps[d], "temperature_large": temps_l[d],
               "fast_full_set": fast_sets.fell_back(d), "large_full_set": large_sets.fell_back(d),
               "fast_retired": d in fast_off, "large_retired": d in large_off, **retire[d]}  # fmt: skip
        row["no_fit_rows"] = row["n_fit"] == 0  # T stayed 1.0
        row["always_defer"] = row["fast_full_set"] and row["large_full_set"]  # both tiers always return the full set
        if i_ev:
            y = np.array([labels_d.index(ev[i].label) for i in i_ev])
            row["top_label_ece_eval"] = top_label_ece(np.stack([softmax_t(raw["eval"][i][None, :], temps[d])[0] for i in i_ev]), y)
            row["top_label_ece_eval_T1"] = top_label_ece(np.stack([softmax_t(raw["eval"][i][None, :], 1.0)[0] for i in i_ev]), y)
            row.update(by[d])
        report[d] = row
    eval_rows = [{"decision": ex.decision, "label": ex.label, "answer": r.answer, "source": r.source,
                  **({"context": ex.context} if ex.decision == "injection_present" else {})} for ex, r in zip(ev, results, strict=True)]
    return temps, cascade, {"alpha": alpha, "decisions": report, "temperatures_large": temps_l, "eval_rows": eval_rows,
                            "fast_retired": sorted(fast_off), "large_retired": sorted(large_off),
                            "flagged": sorted(d for d, r in report.items() if r["fast_full_set"] or r["large_full_set"] or r["no_fit_rows"])}  # fmt: skip
