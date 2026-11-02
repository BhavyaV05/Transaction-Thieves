"""Part B: row loading and joins, calibration modes, collapse checks, selection on fit+cal, eval contrast."""

import json
import random

import numpy as np
import pytest

from ftm.fastmodel.decisions import DECISIONS
from ftm.fastmodel.s1bench_partb import (
    Scores,
    common_rows,
    eval_contrast,
    load_rows,
    metrics_table,
    probs_for,
    read_scores,
    run_selection,
    temperatures_on_fit,
)


def write_rows(path, n_per_part=60, seed=0):
    rng = random.Random(seed)
    lines = []
    for part in ("fit", "cal", "eval"):
        for k in range(n_per_part):
            for d in ("worth_asking", "rule_applies"):
                lines.append({"alert_id": f"{part}{k}", "split": "val", "decision": d, "context": "c", "label": rng.choice(["yes", "no"]), "val_part": part})
    lines.append({"alert_id": "t0", "split": "val", "decision": "stop", "context": "c", "label": "yes", "val_part": "fit"})  # not a learned decision: skipped
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))


def make_scores(rows, name, skill, kind="probs", seed=0, params=100.0, lat=0.01, acc=1.0):
    """Informative by construction: favours the true label with strength `skill`, but is wrong on a fraction 1 - acc of the rows."""
    rng = np.random.default_rng(seed)
    by_i = {}
    for r in rows:
        y = DECISIONS[r["decision"]].labels.index(r["label"])
        if rng.uniform() > acc:
            y = 1 - y
        p = np.full(2, (1 - skill) / 2)
        p[y] += skill
        p = np.clip(p + rng.normal(0, 0.02, 2), 0.01, None)
        p = p / p.sum()
        by_i[r["i"]] = p if kind == "probs" else np.log(p)
    return Scores(name, kind, by_i, {}, params, lat)


def test_load_rows_indexes_learned_decisions_and_caps_like_score_fast_baseline(tmp_path):
    f = tmp_path / "v.jsonl"
    write_rows(f)
    rows = load_rows(f)
    assert len(rows) == 360 and [r["i"] for r in rows] == list(range(360)) and all(r["decision"] != "stop" for r in rows)
    capped = load_rows(f, cap=10, seed=0)
    assert len(capped) == 2 * 3 * 10 and capped == load_rows(f, cap=10, seed=0) and {r["i"] for r in capped} <= set(range(360))
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"alert_id": "x", "split": "train", "decision": "worth_asking", "context": "c", "label": "yes", "val_part": "fit"}) + "\n")
    with pytest.raises(ValueError):
        load_rows(bad)


def test_read_scores_logps_and_probs(tmp_path):
    f = tmp_path / "s.jsonl"
    f.write_text(json.dumps({"i": 0, "variants": [[-1.0, -2.0], [-0.5]]}) + "\n" + json.dumps({"i": 1, "logps": [-0.1, -2.3]}) + "\n")
    s = read_scores(f, "m", "logps", 4000, None)
    assert s.by_i[0] == pytest.approx([np.logaddexp(-1.0, -2.0), -0.5]) and s.by_i[1] == pytest.approx([-0.1, -2.3])
    g = tmp_path / "p.jsonl"
    g.write_text(json.dumps({"i": 0, "probs": [0.3, 0.7], "seconds": 0.5}) + "\n" + json.dumps({"i": 1, "probs": [0.6, 0.4], "seconds": 0.1}) + "\n")
    p = read_scores(g, "o", "probs", 400, None)
    assert p.latency_s == pytest.approx(0.3) and p.kind == "probs"


def test_calibration_modes_own_raw_scaled(tmp_path):
    f = tmp_path / "v.jsonl"
    write_rows(f)
    rows = load_rows(f)
    over = make_scores(rows, "sys1", 0.5, kind="logps", acc=0.7)
    over.by_i = {i: v * 6 for i, v in over.by_i.items()}  # overconfident logits
    t = temperatures_on_fit(over, rows)
    assert t["worth_asking"] > 1.5
    rs = [r for r in rows if r["decision"] == "worth_asking" and r["val_part"] == "cal"]
    own, raw = probs_for(over, rs, "own", t), probs_for(over, rs, "raw", t)
    assert own.max(1).mean() < raw.max(1).mean()
    prob_model = make_scores(rows, "open", 0.5)
    assert probs_for(prob_model, rs, "own", {}) == pytest.approx(np.stack([prob_model.by_i[r["i"]] for r in rs]))
    assert probs_for(prob_model, rs, "scaled", {"worth_asking": 5.0}).max(1).mean() < probs_for(prob_model, rs, "own").max(1).mean()


def test_selection_prefers_skill_drops_collapsed_and_never_reads_eval(tmp_path):
    f = tmp_path / "v.jsonl"
    write_rows(f, n_per_part=150)
    rows = load_rows(f)
    # labels are random coin flips here, so make candidate probabilities informative by construction (they see the label)
    good = make_scores(rows, "good", 0.7, params=4000, lat=0.02, acc=0.9)
    ok = make_scores(rows, "ok", 0.3, seed=1, params=400, lat=0.5, acc=0.6)
    const = Scores("const", "probs", {r["i"]: np.array([0.95, 0.05]) for r in rows}, {}, 10.0, 0.001)
    rep = run_selection([good, ok, const], rows, n_boot=300)
    assert rep["selected"] == "good" and "const" in rep["ineligible"] and rep["collapsed"]["const"]
    # eval rows cannot change the choice: corrupt good's eval scores
    broken = Scores("good", "probs", {i: (np.array([0.5, 0.5]) if rows[i]["val_part"] == "eval" else v) for i, v in good.by_i.items()}, {}, 4000, 0.02)
    assert run_selection([broken, ok, const], rows, n_boot=300)["selected"] == "good"
    # a decision where every candidate collapses is dropped, not blamed on the candidates
    only_const = run_selection([const, Scores("const2", "probs", {r["i"]: np.array([0.9, 0.1]) for r in rows}, {}, 5.0, 0.001)], rows, n_boot=100)
    assert only_const["selected"] is None and set(only_const["no_candidate_has_skill"]) == {"worth_asking", "rule_applies"}


def test_metrics_table_and_eval_contrast_use_common_rows_and_holm(tmp_path):
    f = tmp_path / "v.jsonl"
    write_rows(f, n_per_part=120)
    rows = load_rows(f)
    a, b = make_scores(rows, "a", 0.8, acc=0.9), make_scores(rows, "b", 0.2, seed=2, acc=0.55)
    partial = Scores("c", "probs", {i: v for i, v in a.by_i.items() if i % 2 == 0}, {}, 1.0, 0.1)
    assert all(r["i"] % 2 == 0 for r in common_rows(rows, [a, partial]))
    tab = metrics_table([a, b], rows, ["fit", "cal", "eval"])
    assert tab["a"]["decisions"]["worth_asking"]["eval"]["accuracy"] > tab["b"]["decisions"]["worth_asking"]["eval"]["accuracy"]
    assert tab["a"]["decisions"]["worth_asking"]["eval"]["fit_majority_accuracy"] > 0
    c = eval_contrast(a, [b], rows, n_boot=300)
    k = "b|worth_asking"
    assert c[k]["ci_hi"] < 0 and c[k]["reject_holm"] and c[k]["p_holm"] <= 0.05
