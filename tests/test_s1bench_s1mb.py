"""S1MB scorer (hand-computed fixtures, cross-check against published rows when the data is present) and rendering."""

import json
from pathlib import Path

import numpy as np
import pytest

from ftm.fastmodel.model import LogprobFastModel
from ftm.fastmodel.s1bench_s1mb import (
    INFERENCE_FIELDS,
    aggregate,
    make_decider,
    option_labels,
    predict_line,
    render,
    run_line,
    score_benchmark,
    state_text,
)
from ftm.fastmodel.typed import TypedDecider
from tests.test_s1bench import PromptClient


def noul_line(i, t_true):
    return {"case_id": f"c{i}", "question_id": "q", "task": "noul", "state": {"x": str(i)}, "system_prompt": "", "instructions": "Is it?",
            "options": [{"id": "false", "description": "no it is not", "value": None}, {"id": "true", "description": "yes it is", "value": None}],
            "target": {"false": 1 - t_true, "true": t_true}}  # fmt: skip


def choice_line(i, correct, n=3):
    return {"case_id": f"c{i}", "question_id": "q", "task": "choice", "state": "s", "system_prompt": "", "instructions": "Pick",
            "options": [{"id": f"o{k}", "description": f"option {k}", "value": None} for k in range(n)],
            "target": {f"o{k}": float(k == correct) for k in range(n)}}  # fmt: skip


def score_line(i, hi):
    ops = [{"id": f"l{k}", "description": f"level {k}", "value": float(v)} for k, v in enumerate([0, 2, 4])]
    t = [0.0, 0.0, 1.0] if hi else [1.0, 0.0, 0.0]
    return {"case_id": f"c{i}", "question_id": "q", "task": "score", "state": "s", "system_prompt": "", "instructions": "Rate", "options": ops,
            "target": {o["id"]: p for o, p in zip(ops, t, strict=True)}}  # fmt: skip


def key(ln):
    return f"{ln['case_id']}|{ln['question_id']}"


def test_noul_metrics_hand_values_with_soft_targets():
    lines = [noul_line(0, 1.0), noul_line(1, 0.0), noul_line(2, 0.0), noul_line(3, 0.5)]
    preds = {key(lines[0]): {"false": 0.2, "true": 0.8}, key(lines[1]): {"false": 0.9, "true": 0.1},
             key(lines[2]): {"false": 0.4, "true": 0.6}, key(lines[3]): {"false": 0.5, "true": 0.5}}  # fmt: skip
    m = score_benchmark("noul", lines, preds)
    # predicted true: c0 (tp 1), c2 (fp 1), c3 (tp .5, fp .5); predicted false: c1 (tn 1)
    assert (m["accuracy"], m["precision"], m["positive_recall"], m["specificity"]) == pytest.approx((2.5 / 4, 1.5 / 3, 1.0, 1 / 2.5))
    assert m["balanced_accuracy"] == pytest.approx((1.0 + 0.4) / 2) and m["baseline_adjusted_skill"] == pytest.approx(2 * 0.7 - 1)
    assert m["binary_brier"] == pytest.approx(np.mean([0.04, 0.01, 0.36, 0.0]))
    prev = 1.5 / 4
    base = np.mean([(1 - prev) ** 2, prev**2, prev**2, (0.5 - prev) ** 2])
    assert m["constant_brier_baseline"] == pytest.approx(base) and m["brier_skill"] == pytest.approx(1 - m["binary_brier"] / base)
    assert m["complete"] and m["baseline_adjusted_score"] == pytest.approx(0.4)


def test_choice_metrics_and_baselines():
    lines = [choice_line(0, 0), choice_line(1, 1), choice_line(2, 2), choice_line(3, 0)]
    best = {key(ln): {f"o{k}": float(k == int(np.argmax(list(ln["target"].values())))) for k in range(3)} for ln in lines}
    m = score_benchmark("choice", lines, best)
    assert m["target_mass_at_prediction"] == 1.0 and m["uniform_accuracy_baseline"] == pytest.approx(1 / 3)
    assert m["fixed_answer_accuracy_baseline"] == pytest.approx(0.5)  # always o0 / position 0 is right twice in four
    assert m["baseline_adjusted_skill"] == pytest.approx((1 - 0.5) / 0.5)
    always_o2 = {key(ln): {"o0": 0.0, "o1": 0.0, "o2": 1.0} for ln in lines}
    m2 = score_benchmark("choice", lines, always_o2)
    assert m2["target_mass_at_prediction"] == 0.25 and m2["baseline_adjusted_score"] == 0.0  # clipped at zero
    tie = {key(ln): {"o0": 0.4, "o1": 0.4, "o2": 0.2} for ln in lines}
    assert score_benchmark("choice", lines, tie)["target_mass_at_prediction"] == pytest.approx(0.5)  # ties -> first option


def test_score_metrics_and_degenerate_baseline():
    lines = [score_line(0, True), score_line(1, False), score_line(2, True)]
    perfect = {key(ln): ln["target"] for ln in lines}
    m = score_benchmark("score", lines, perfect)
    assert m["normalized_expected_score_mae"] == 0 and m["baseline_adjusted_skill"] == 1.0
    # constant median of normalised target values [1, 0, 1] is 1 -> baseline MAE 1/3
    assert m["constant_score_baseline"] == 1.0 and m["constant_mae_baseline"] == pytest.approx(1 / 3)
    mid = {key(ln): {"l0": 0.0, "l1": 1.0, "l2": 0.0} for ln in lines}  # always 0.5 normalised
    m2 = score_benchmark("score", lines, mid)
    assert m2["normalized_expected_score_mae"] == pytest.approx(0.5) and m2["baseline_adjusted_skill"] == pytest.approx(1 - 0.5 / (1 / 3))
    same = [score_line(0, True), score_line(1, True)]  # constant targets: baseline MAE 0 -> adjustment undefined
    assert score_benchmark("score", same, {key(ln): ln["target"] for ln in same})["baseline_adjusted_skill"] is None


def test_unanswered_judgments_make_the_benchmark_incomplete():
    lines = [noul_line(0, 1.0), noul_line(1, 0.0)]
    m = score_benchmark("noul", lines, {key(lines[0]): {"false": 0.1, "true": 0.9}})
    assert not m["complete"] and m["answered"] == 1 and m["total"] == 2


def test_aggregate_clips_per_benchmark_and_requires_all_tasks():
    per = {"a": {"baseline_adjusted_score": 0.5}, "b": {"baseline_adjusted_score": 0.0}, "c": {"baseline_adjusted_score": None}, "d": {"baseline_adjusted_score": 1.0},
           "e": {"baseline_adjusted_score": 0.25}}  # fmt: skip
    tasks = {"a": "choice", "b": "choice", "c": "noul", "d": "noul", "e": "score"}
    r = aggregate(per, tasks)
    assert r["tasks"]["choice"]["score"] == 0.25 and r["tasks"]["noul"] == {"score": 1.0, "eligible": 1, "benchmarks": 2}
    assert r["task_avg"] == pytest.approx((0.25 + 1.0 + 0.25) / 3)
    assert aggregate(per, tasks, covered=["a", "b"])["task_avg"] is None  # no noul/score benchmark covered
    assert aggregate(per, tasks, covered=["a", "d", "e"])["n_benchmarks"] == 3


# -- rendering ----------------------------------------------------------------------------------------------
def test_render_never_reads_the_target_and_keeps_authored_definitions():
    ln = noul_line(0, 1.0)
    ln["target"] = None  # would raise if touched
    ctx, q, labels = render(ln)
    assert labels == ["no it is not", "yes it is"] and q == "Is it?" and ctx == "x: 0"
    assert set(INFERENCE_FIELDS) <= set(ln) and "target" not in INFERENCE_FIELDS
    sc = score_line(0, True)
    assert option_labels(sc) == ["0: level 0", "2: level 1", "4: level 2"]
    assert render(choice_line(0, 1), order=[2, 0, 1])[2] == ["option 2", "option 0", "option 1"]
    assert state_text({"a": "x", "b": 3}) == "a: x\nb: 3" and state_text({"a": {"n": 1}}) == '{"a": {"n": 1}}' and state_text("raw") == "raw"
    assert "SYS" in render({**ln, "system_prompt": "SYS"})[0]


def test_predict_line_returns_stored_order_even_when_options_are_permuted():
    ln = choice_line(0, 1, n=4)
    dec = TypedDecider(LogprobFastModel(PromptClient(favour="option 2"), temperature=1.0))
    p = predict_line(dec, ln, uid="u0")
    assert int(np.argmax(p)) == 2 and sum(p) == pytest.approx(1)
    for k, order in enumerate([[3, 1, 0, 2], [2, 3, 1, 0]]):
        q = predict_line(dec, ln, uid=f"u{k + 1}", order=order)  # shown order permuted, result realigned to stored order
        assert int(np.argmax(q)) == 2


def test_run_line_records_errors_and_over_long_prompts_without_truncating():
    dec = TypedDecider(LogprobFastModel(PromptClient(favour="option 1"), temperature=1.0))
    ok = run_line(dec, choice_line(0, 1), "a", permute=True)
    assert ok["probs"] and len(ok["perm_probs"]) == 3 and "error" not in ok and ok["seconds"] >= 0
    long = choice_line(1, 0)
    long["state"] = "x" * 20000
    assert run_line(dec, long, "b", max_chars=9000)["error"] == "over_max_chars"

    class Boom:
        def chat(self, *a, **k):
            raise RuntimeError("context length exceeded")

    r = run_line(TypedDecider(LogprobFastModel(Boom())), choice_line(2, 0), "c")
    assert "context length exceeded" in r["error"] and "probs" not in r
    assert make_decider(dec, "prod") is dec and type(make_decider(dec, "listed")).__name__ == "ListedTwoStage"
    with pytest.raises(ValueError):
        make_decider(dec, "other")


# -- cross-check against the published rows (needs the downloaded data; skipped otherwise) -----------------------
PREP = Path("/home/akshat/projects/s1bench_data/prepared")


@pytest.mark.skipif(not (PREP / "published" / "alibiserikbay__JevK5").exists(), reason="published S1MB rows not downloaded")
def test_scorer_reproduces_stored_metrics_of_a_published_row():
    d = PREP / "published" / "alibiserikbay__JevK5"
    checked = 0
    for f in sorted(d.glob("*.json"))[:60]:
        j = json.loads(f.read_text())
        src = PREP / "s1mb" / f"{j['benchmark']}.jsonl"
        if not src.exists() or j["abstentions"]:
            continue
        lines = [json.loads(x) for x in src.read_text().splitlines()]
        m = score_benchmark(lines[0]["task"], lines, {k: v for k, v in j["predictions"].items() if v is not None})
        for k, v in j["metrics"].items():
            if isinstance(v, int | float) and k in m and m[k] is not None:
                assert m[k] == pytest.approx(v, abs=1e-9), (j["benchmark"], k)
        checked += 1
    assert checked > 20
