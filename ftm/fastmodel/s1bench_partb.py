"""S1BENCH Part B: System 1 against open System One models on OUR decisions (validation rows only).

Owner: T3 (Akshat).  Spec: docs/t3_s1bench.md sections 3, 4 and 5 (selection rule written before any run).

Rows: `data/decisions/hi_medium_s1v1/decisions_val.jsonl` (JSON lines; `val_part` fit / cal / eval).  The row index `i` and the optional per-(decision, part)
cap are the ones of `scripts/score_fast_baseline.py`, so its score files (`fast.jsonl`: `i`, `logps`/`variants`) join to the open models' score files
(`i`, `probs`, `seconds`) by `i`.  A candidate is either logit-based (System 1, base Qwen3-4B: calibrated by a temperature fitted on `fit`, the
cascade's calibration) or probability-based (an open model: its OWN output, plus a temperature-rescaled twin for a like-for-like calibration check).
No train row and no test row is read here.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ftm.eval.stats import holm
from ftm.fastmodel.calibration import fit_temperature, softmax_t
from ftm.fastmodel.decisions import DECISIONS
from ftm.fastmodel.model import logps_from_variants
from ftm.fastmodel.s1bench import (
    Candidate,
    collapse_flags,
    ece_equal_mass,
    paired_cluster_bootstrap,
    per_row_brier,
    select_candidate,
)

PARTS = ("fit", "cal", "eval")
DECISION_NAMES = tuple(DECISIONS)  # worth_asking, typology, rule_applies, injection_present (stop is exact, not learned)
ECE_BINS = 10


def load_rows(path: Path, cap: int | None = None, seed: int = 0) -> list[dict]:
    """Val decision rows with their `val_part`, in file order, `i` = index among the learned decisions.  Same cap rule as score_fast_baseline.load_rows."""
    rows = []
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r["decision"] in DECISIONS:
            rows.append({**r, "i": len(rows)})
    if any(r.get("split") != "val" for r in rows):
        raise ValueError("S1BENCH Part B reads validation rows only")
    if any(r["val_part"] not in PARTS for r in rows):
        raise ValueError(f"val_part must be one of {PARTS}")
    if cap is None:
        return rows
    rng = random.Random(seed)
    keep: set[int] = set()
    groups: dict[tuple[str, str], list[int]] = {}
    for r in rows:
        groups.setdefault((r["decision"], r["val_part"]), []).append(r["i"])
    for key in sorted(groups):
        idx = groups[key]
        keep.update(idx if len(idx) <= cap else rng.sample(idx, cap))
    return [r for r in rows if r["i"] in keep]


@dataclass
class Scores:
    """One candidate's raw scores by row index: `logps` (logit-based) or `probs` (probability-based), in DECISIONS[d].labels order."""

    name: str
    kind: str  # "logps" | "probs"
    by_i: dict[int, np.ndarray]
    seconds: dict[int, float]
    params_m: float
    latency_s: float | None  # median serial seconds per decision
    hardware: str = ""


def read_scores(path: Path, name: str, kind: str, params_m: float, latency_s: float | None, hardware: str = "") -> Scores:
    by_i, secs = {}, {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if kind == "logps":
            by_i[r["i"]] = np.asarray(logps_from_variants(r["variants"]) if r.get("variants") else r["logps"], float)
        else:
            by_i[r["i"]] = np.asarray(r["probs"], float)
        if "seconds" in r:
            secs[r["i"]] = r["seconds"]
    if latency_s is None and secs:
        latency_s = float(np.median(list(secs.values())))
    return Scores(name, kind, by_i, secs, params_m, latency_s, hardware)


def labels_of(decision: str) -> tuple[str, ...]:
    return DECISIONS[decision].labels


def gold(rows: Sequence[dict]) -> np.ndarray:
    return np.array([labels_of(r["decision"]).index(r["label"]) for r in rows], int)


def temperatures_on_fit(sc: Scores, rows: Sequence[dict]) -> dict[str, float]:
    """One T per decision, fitted on the `fit` rows only (bounds as ftm.fastmodel.calibration)."""
    out = {}
    for d in DECISION_NAMES:
        fit = [r for r in rows if r["decision"] == d and r["val_part"] == "fit" and r["i"] in sc.by_i]
        if len(fit) < 20:
            out[d] = 1.0
            continue
        X = np.stack([sc.by_i[r["i"]] if sc.kind == "logps" else np.log(np.clip(sc.by_i[r["i"]], 1e-12, 1)) for r in fit])
        out[d] = fit_temperature(X, gold(fit))
    return out


def probs_for(sc: Scores, rows: Sequence[dict], mode: str, temps: dict[str, float] | None = None) -> np.ndarray:
    """(n, k) probabilities of one decision's rows.  mode: own (logps: softmax_T with T on fit; probs: as returned), raw (T = 1), scaled (T on fit for either kind)."""
    d = rows[0]["decision"]
    X = np.stack([sc.by_i[r["i"]] for r in rows])
    if sc.kind == "probs":
        P = X / X.sum(axis=1, keepdims=True)
        if mode == "scaled":
            return softmax_t(np.log(np.clip(P, 1e-12, 1)), (temps or {}).get(d, 1.0))
        return P
    t = 1.0 if mode == "raw" else (temps or {}).get(d, 1.0)
    return softmax_t(X, t)


def decision_rows(rows: Sequence[dict], decision: str, parts: Sequence[str], have: set[int]) -> list[dict]:
    return [r for r in rows if r["decision"] == decision and r["val_part"] in parts and r["i"] in have]


def common_rows(rows: Sequence[dict], cands: Sequence[Scores]) -> list[dict]:
    have = set.intersection(*[set(c.by_i) for c in cands])
    return [r for r in rows if r["i"] in have]


def fit_prior(rows: Sequence[dict], decision: str) -> tuple[np.ndarray, int]:
    fit = [r for r in rows if r["decision"] == decision and r["val_part"] == "fit"]
    y = gold(fit)
    counts = np.bincount(y, minlength=len(labels_of(decision)))
    return counts / counts.sum(), int(counts.argmax())


def metrics_table(cands: Sequence[Scores], rows: Sequence[dict], parts: Sequence[str], mode: str = "own") -> dict:
    """Per candidate, decision and part: n, accuracy, majority accuracy (fit majority), Brier, prior Brier, log loss, ECE (equal mass, 10 bins),
    prediction diversity.  `parts` is the list of parts to report separately (each one on its own)."""
    from ftm.fastmodel.s1bench import core_metrics

    rows = common_rows(rows, cands)
    out: dict = {}
    for c in cands:
        temps = temperatures_on_fit(c, rows)
        out[c.name] = {"temperatures": temps, "decisions": {}}
        for d in DECISION_NAMES:
            if not any(r["decision"] == d and r["val_part"] == "fit" for r in rows):
                continue
            prior, maj = fit_prior(rows, d)
            for part in parts:
                rs = decision_rows(rows, d, [part], set(c.by_i))
                if not rs:
                    continue
                P, y = probs_for(c, rs, mode, temps), gold(rs)
                m = core_metrics(P, y, labels_of(d))
                m["ece"] = ece_equal_mass(P, y, ECE_BINS)
                m["fit_majority_accuracy"] = float(np.mean(y == maj))
                out[c.name]["decisions"].setdefault(d, {})[part] = m
    return out


def build_candidates(cands: Sequence[Scores], rows: Sequence[dict], parts: Sequence[str] = ("fit", "cal"), mode: str = "own") -> tuple[list[Candidate], dict]:
    """Selection inputs on `parts` only (fit+cal): per-row Brier arrays, ECE, collapse flags.  Rows are the intersection over all candidates."""
    rows = common_rows(rows, cands)
    out, clusters = [], {}
    for d in DECISION_NAMES:
        rs = decision_rows(rows, d, parts, set.intersection(*[set(c.by_i) for c in cands]))
        clusters[d] = [r["alert_id"] for r in rs]
    for c in cands:
        temps = temperatures_on_fit(c, rows)
        cand = Candidate(c.name, c.params_m, c.latency_s if c.latency_s is not None else float("inf"))
        for d in DECISION_NAMES:
            rs = decision_rows(rows, d, parts, set(c.by_i))
            if not rs:
                continue
            P, y = probs_for(c, rs, mode, temps), gold(rs)
            prior, maj = fit_prior(rows, d)
            cand.brier_by_decision[d] = per_row_brier(P, y)
            cand.ece_by_decision[d] = ece_equal_mass(P, y, ECE_BINS)
            cand.collapsed[d] = collapse_flags(P, y, prior, maj)["collapsed"]
        out.append(cand)
    return out, clusters


def run_selection(cands: Sequence[Scores], rows: Sequence[dict], *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Doc section 4 on fit+cal.  A decision where EVERY candidate is collapsed is dropped (no candidate has skill there) before eligibility."""
    cs, clusters = build_candidates(cands, rows)
    decisions = [d for d in DECISION_NAMES if all(d in c.brier_by_decision for c in cs)]
    skill = [d for d in decisions if any(not c.collapsed[d] for c in cs)]
    dropped = [d for d in decisions if d not in skill]
    for c in cs:
        c.brier_by_decision = {d: c.brier_by_decision[d] for d in skill}
        c.ece_by_decision = {d: c.ece_by_decision[d] for d in skill}
    rep = select_candidate(cs, {d: clusters[d] for d in skill}, n_boot=n_boot, seed=seed) if skill else {"selected": None, "ranked": [], "ineligible": [c.name for c in cs]}
    rep["no_candidate_has_skill"] = dropped
    rep["collapsed"] = {c.name: [d for d in decisions if c.collapsed.get(d)] for c in cs}
    rep["parts"] = ["fit", "cal"]
    return rep


def eval_contrast(chosen: Scores, others: Sequence[Scores], rows: Sequence[dict], mode: str = "own", *, n_boot: int = 2000, seed: int = 0) -> dict:
    """On `eval`, once: per decision, paired cluster-bootstrap Brier difference chosen - other with Holm over the whole family (other x decision)."""
    allc = [chosen, *others]
    rows = common_rows(rows, allc)
    out: dict = {}
    pvals: dict[str, float] = {}
    for o in others:
        for d in DECISION_NAMES:
            rs = decision_rows(rows, d, ["eval"], set.intersection(*[set(c.by_i) for c in allc]))
            if not rs:
                continue
            y = gold(rs)
            a = per_row_brier(probs_for(chosen, rs, mode, temperatures_on_fit(chosen, rows)), y)
            b = per_row_brier(probs_for(o, rs, mode, temperatures_on_fit(o, rows)), y)
            r = paired_cluster_bootstrap(a, b, [r["alert_id"] for r in rs], n_boot=n_boot, seed=seed)
            out[f"{o.name}|{d}"] = r
            pvals[f"{o.name}|{d}"] = max(r["p_two_sided"], 1 / n_boot)
    if pvals:
        for k, v in holm(pvals).items():
            out[k]["p_holm"] = v["p_adj"]
            out[k]["reject_holm"] = v["reject"]
    return out
