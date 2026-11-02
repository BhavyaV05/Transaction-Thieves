"""External-validity tasks: typed items, per-item scoring with order arms, task reports."""

import json

import numpy as np
import pytest

from ftm.fastmodel.model import LogprobFastModel
from ftm.fastmodel.s1bench import Item, TaskSpec
from ftm.fastmodel.s1bench_ext import (
    arm_defs,
    build_task,
    evaluate_task,
    items_for,
    load_items,
    rescale,
    score_item,
)
from ftm.fastmodel.typed import TypedDecider
from tests.test_s1bench import PromptClient


def dec(favour=None):
    return TypedDecider(LogprobFastModel(PromptClient(favour=favour), temperature=1.0))


def test_arm_defs_for_lettered_and_yes_no_tasks():
    multi = TaskSpec("m", "q", ("a", "b", "c"))
    assert [a[0] for a in arm_defs(multi, False)] == ["main"] and [a[0] for a in arm_defs(multi, True)] == ["main", "L1", "L2", "L3"]
    yn = TaskSpec("y", "q", ("yes", "no"))
    arms = arm_defs(yn, True)
    assert [a[0] for a in arms] == ["main", "L0", "L1", "L2", "L3"] and arms[0][2] is False and all(a[2] for a in arms[1:]) and arms[2][1] == [1, 0]


def test_score_item_realigns_permuted_arms_and_times_the_main_decision():
    spec = TaskSpec("m", "Which?", ("alpha", "beta", "gamma"))
    rec = score_item(dec("beta"), spec, Item("i", "text", 1), "u", with_perm=True)
    for arm in ("main", "L0", "L1", "L2", "L3"):
        assert int(np.argmax(rec[arm])) == 1 and sum(rec[arm]) == pytest.approx(1)  # an order-invariant model stays invariant after realignment
    assert rec["seconds"] >= 0
    yn = score_item(dec("yes"), TaskSpec("y", "Toxic?", ("yes", "no")), Item("j", "text", 0), "v", with_perm=True)
    assert yn["main"][0] > 0.9 and yn["L0"] == yn["L1"] or yn["L1"]  # lettered arms exist for the binary task
    assert score_item(dec(), spec, Item("k", "t", 0), "w", with_perm=False).keys() >= {"main", "seconds", "label"}


def test_load_build_and_items(tmp_path):
    d = tmp_path / "t"
    d.mkdir()
    meta = {"task": "t", "kind": "multilabel_yesno", "labels": ["A", "B"], "question": "Is it of type {label}?"}
    (d / "meta.json").write_text(json.dumps(meta))
    (d / "eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [{"id": "e0", "text": "x", "label": [1]}, {"id": "e1", "text": "y", "label": []}]))
    m, rows = load_items(tmp_path, "t", "eval")
    task = build_task(m)
    assert [s.name for s in task.specs] == ["t/A", "t/B"] and task.specs[1].question == "Is it of type B?"
    items = items_for(task, rows)
    assert [[it.label for it in its] for its in items] == [[1, 1], [0, 1]] and items[0][0].cluster == "e0"  # label 0 = yes
    with pytest.raises(ValueError):
        build_task({"task": "x", "kind": "weird"})


def _scored(P, y, secs=0.01):
    return [{"id": str(i), "label": int(yy), "main": list(p), "seconds": secs} for i, (p, yy) in enumerate(zip(P, y, strict=True))]


def test_evaluate_multiclass_oos_metrics():
    meta = {"task": "c", "kind": "multiclass_oos", "labels": ["a", "b", "out of scope"], "question": "q", "oos_label": 2}
    task = build_task(meta)
    y = [0, 1, 2, 2, 0]
    P = [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.5, 0.3, 0.2], [0.1, 0.1, 0.8]]
    r = evaluate_task(task, [_scored(P, y)], None)
    assert r["in_scope_accuracy"] == pytest.approx(2 / 3) and r["oos_recall"] == pytest.approx(0.5) and r["oos_precision"] == pytest.approx(0.5)
    assert r["core"]["accuracy"] == pytest.approx(3 / 5) and r["latency"]["median_s"] == 0.01 and "order_bias" not in r


def test_evaluate_binary_and_multilabel_and_calibration_split():
    task = build_task({"task": "tox", "kind": "binary_yesno", "labels": ["yes", "no"], "question": "q"})
    y = [0, 0, 1, 1, 1, 1]
    P = [[0.9, 0.1], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.6, 0.4], [0.1, 0.9]]
    r = evaluate_task(task, [_scored(P, y)], [_scored(P, y)])
    assert r["toxic_f1"] == pytest.approx(2 * 1 / (2 * 1 + 1 + 1)) and 0.5 < r["auroc"] <= 1 and r["temperature"] > 0 and "core_scaled" in r
    ml = build_task({"task": "u", "kind": "multilabel_yesno", "labels": ["A", "B"], "question": "q {label}"})
    sa = _scored([[0.9, 0.1], [0.2, 0.8]], [0, 1])
    sb = _scored([[0.3, 0.7], [0.1, 0.9]], [1, 1])
    r2 = evaluate_task(ml, [sa, sb], None)
    assert r2["micro_f1"] == 1.0 and r2["per_type"]["A"]["f1"] == 1.0 and r2["per_type"]["B"]["f1"] is None and r2["n_decisions"] == 4


def test_order_bias_block_is_built_from_arms():
    task = build_task({"task": "m", "kind": "multiclass", "labels": ["a", "b"], "question": "q"})
    rows = [{"id": "0", "label": 0, "main": [0.8, 0.2], "L0": [0.8, 0.2], "L1": [0.6, 0.4], "L2": [0.6, 0.4], "L3": [0.6, 0.4], "seconds": 0.0}]
    r = evaluate_task(task, [rows], None)
    assert r["order_bias"]["mean_tv"] == pytest.approx(0.2) and r["order_bias"]["accuracy_range"] == 0.0


def test_rescale_is_a_temperature_on_log_probs():
    p = np.array([[0.8, 0.2]])
    assert rescale(p, 1.0) == pytest.approx(p) and rescale(p, 100.0)[0, 0] == pytest.approx(0.5, abs=0.01) and rescale(p, 0.5)[0, 0] > 0.9
