"""T3 weeks 5-6: strategy library (C12) and the Reflexion / ReWOO ports."""

import json
import random
from datetime import datetime, timedelta

import pytest

from ftm.external.reflexion import Reflexion
from ftm.external.rewoo import Rewoo, parse_plan, solver_probability
from ftm.harness.interface import Investigator, ScriptedToolBox
from ftm.library.build import build_library
from ftm.library.shape import SituationShape, shape_from_rows
from ftm.library.store import FrozenLibraryError, LibraryToolBox, StrategyLibrary
from ftm.llm.mock import MockLLM
from ftm.planners.llm import LLMPlanner
from ftm.types import AlertContext, Limits, Reply

T0 = datetime(2022, 9, 5, 12)
Q_PASS = "passthrough(ALERT.receiver, min_frac=0.8, within_h=24)"
Q_FAN = "fan_in(ALERT.receiver, min_senders=5, window_h=48)"


def rows(n_payers=5, hours=30):
    out = [{"src_alias": f"A{i:07x}", "dst_alias": "R", "from_bank": 220, "to_bank": 70, "amount_paid": 900.0 + i,
            "timestamp": T0 - timedelta(hours=hours - i)} for i in range(n_payers)]  # fmt: skip
    out.append({"src_alias": "R", "dst_alias": "X", "from_bank": 70, "to_bank": 1588, "amount_paid": 4500.0, "timestamp": T0})
    return out


# -- shape ------------------------------------------------------------------------------------------------
def test_shape_buckets_and_no_account_in_key():
    s = shape_from_rows(rows(), "R", T0, typology="fan_in")
    assert s.key() == ("fan_in", "4-7", "1", "3-4", "100-1k varied", "1-3d")  # 4,500 out vs ~900 in: CV ~0.96
    flat = [r for r in rows() if r["dst_alias"] == "R"]
    assert shape_from_rows(flat, "R", T0).amount == "100-1k flat"
    assert "R" not in s.key() and "A0000001" not in str(s.key())
    assert SituationShape.from_dict(s.as_dict()) == s
    assert shape_from_rows(rows(1, hours=2), "R", T0).key()[1:] == ("1", "1", "3-4", "1k-10k varied", "<6h")
    assert s.with_typology(None).typology == "any"


# -- build + search ---------------------------------------------------------------------------------------
def records(shape, question, gain_mean, n, accepted=True, split="train", seed=0, receiver="R"):
    rng = random.Random(seed)
    return [{"alert_id": f"{question[:4]}{seed}-{i}", "split": split, "shape": shape.as_dict(), "receiver": receiver,
             "question": question, "family": question.split("(")[0], "eps": 0.5, "accepted": accepted,
             "gain": gain_mean + rng.gauss(0, 0.01)} for i in range(n)]  # fmt: skip


def test_build_successes_failures_and_min_support():
    sh = shape_from_rows(rows(), "R", T0, typology="fan_in")
    recs = (records(sh, Q_PASS, 0.02, 30) + records(sh, Q_FAN, -0.02, 30, seed=1)
            + records(sh, "n_tx(ALERT.sender, direction=in, min_amount=1000, window_h=72)", 0.05, 30, accepted=False, seed=2)
            + records(sh, "fan_out(ALERT.sender, min_receivers=5, window_h=48)", 0.5, 5, seed=3))  # fmt: skip
    lib = build_library(recs, n_min=20)
    kinds = {s.question.split("(")[0]: s.kind for s in lib.strategies}
    assert kinds == {"passthrough": "success", "fan_in": "failure", "n_tx": "failure"}  # fan_out: 5 alerts < n_min
    assert all("A0" not in s.text and "R" not in s.text.split("`")[0][:3] for s in lib.strategies)
    hits = lib.search(sh, k=3)
    assert hits[0].kind == "success" and hits[0].question == Q_PASS
    assert lib.search(sh.with_typology("cycle")) == []  # other typology: no match
    assert lib.search(sh.with_typology(None))  # unknown typology matches any


def test_build_refuses_test_records_and_noisy_singletons():
    sh = shape_from_rows(rows(), "R", T0)
    with pytest.raises(ValueError):
        build_library(records(sh, Q_PASS, 0.02, 30, split="test"))
    assert len(build_library(records(sh, Q_PASS, -1.0, 1), n_min=1)) == 0  # one alert: se is infinite, no note


def test_nearest_shape_fallback():
    near = shape_from_rows(rows(5), "R", T0, typology="fan_in")
    lib = build_library(records(near, Q_PASS, 0.02, 30), n_min=20)
    far_query = shape_from_rows(rows(9), "R", T0, typology="fan_in")  # in_deg 8+ instead of 4-7
    assert far_query.key() != near.key() and lib.search(far_query)[0].question == Q_PASS


def test_freeze_save_load_hash(tmp_path):
    sh = shape_from_rows(rows(), "R", T0, typology="fan_in")
    lib = build_library(records(sh, Q_PASS, 0.02, 30), n_min=20)
    h = lib.freeze()
    with pytest.raises(FrozenLibraryError):
        lib.add(lib.strategies[0])
    assert lib.save(tmp_path / "lib.json") == h
    again = StrategyLibrary.load(tmp_path / "lib.json")
    assert again.frozen and again.content_hash() == h and len(again) == 1
    doc = json.loads((tmp_path / "lib.json").read_text())
    doc["strategies"][0]["text"] = "edited"
    (tmp_path / "lib.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        StrategyLibrary.load(tmp_path / "lib.json")


def test_account_keyed_ablation_memorises_receiver():
    sh = shape_from_rows(rows(), "R", T0)
    lib = build_library(records(sh, Q_PASS, 0.02, 30, receiver="A0bad001"), keying="account", n_min=20)
    tools = LibraryToolBox(ScriptedToolBox(), lib)
    assert tools.search_library({"receiver": "A0bad001"}) and not tools.search_library({"receiver": "A0other1"})


def test_planner_reads_library_and_refreshes_on_typology():
    sh = shape_from_rows(rows(), "R", T0, typology="fan_in")
    lib = build_library(records(sh, Q_PASS, 0.02, 30), n_min=20)
    summary = {"receiver": "R", "evidence": "e", "partner_banks": [220], "shape": sh.with_typology(None).as_dict()}
    alert = AlertContext("V1", "70", T0.isoformat(), "val", 0.2, summary)
    step = json.dumps({"hypothesis": {"typology": "fan_in", "rationale": "r"}, "action": "compose", "question_id": None,
                       "program": Q_FAN, "bank": 220, "eps": 0.5, "rule_id": None})  # fmt: skip
    stop = json.dumps({"hypothesis": {"typology": "fan_in", "rationale": "r"}, "action": "stop", "question_id": None,
                       "program": None, "bank": None, "eps": 0.0, "rule_id": None})  # fmt: skip
    llm = MockLLM(script=[step, stop])
    LLMPlanner(llm, use_library=True).investigate(alert, LibraryToolBox(ScriptedToolBox([Reply("220", "bool", True, 0.5, 0.3)]), lib), Limits())
    assert all("## Strategies" in r["messages"][1]["content"] for r in llm.requests)
    assert Q_PASS in llm.requests[1]["messages"][1]["content"]


# -- Reflexion ----------------------------------------------------------------------------------------------
ALERT = AlertContext("V2", "70", T0.isoformat(), "val", 0.2, {"receiver": "R", "evidence": "e", "partner_banks": [220, 1588]})


def test_reflexion_retries_with_memory_and_shared_budget():
    llm = MockLLM(script=[
        f"Thought 1: x\nAction 1: ask_bank[220 | 0.5 | {Q_PASS}]", "Thought 2: y\nAction 2: finish[0.3]",
        "RETRY: only one weak answer.", "I relied on one noisy answer; next ask about payer counts.",
        f"Thought 1: z\nAction 1: ask_bank[1588 | 0.5 | {Q_FAN}]", "Thought 2: w\nAction 2: finish[0.7]",
        "CONFIDENT. Two signals agree.",
    ])  # fmt: skip
    r = Reflexion(llm, max_trials=3)
    assert isinstance(r, Investigator)
    v = r.investigate(ALERT, ScriptedToolBox([Reply("220", "bool", True, 0.5, 0.2), Reply("1588", "bool", True, 0.5, 0.2)]),
                      Limits())  # fmt: skip
    assert v.probability == 0.7 and len(v.questions_log) == 2 and v.llm_calls == 7
    trial2 = llm.requests[4]["messages"][1]["content"]
    assert "Reflections from your earlier attempts" in trial2 and "payer counts" in trial2
    assert "Answers already obtained" in trial2 and "passthrough" in trial2


def test_reflexion_stops_when_confident_or_out_of_calls():
    llm = MockLLM(script=["Thought: a\nAction: finish[0.2]", "CONFIDENT"])
    assert Reflexion(llm).investigate(ALERT, ScriptedToolBox(), Limits()).llm_calls == 2
    v = Reflexion(MockLLM(script=["Thought: a\nAction: finish[0.2]"] * 5), min_calls_for_retry=4).investigate(
        ALERT, ScriptedToolBox(), Limits(max_llm_calls=3))
    assert v.llm_calls == 1 and v.probability == 0.2  # 2 calls left < 4: no retry


# -- ReWOO --------------------------------------------------------------------------------------------------
def test_rewoo_plan_work_solve_in_two_calls():
    plan = (f"Plan: check pass-through\n#E1 = ask_bank[220 | 0.5 | {Q_PASS}]\nPlan: rules\n#E2 = lookup_rules[314b sharing]\n"
            f"Plan: payers\n#E3 = ask_bank[1588 | 2.0 | {Q_FAN}]")  # fmt: skip
    assert [(n, c.tool) for _, n, c in parse_plan(plan)] == [(1, "ask_bank"), (2, "lookup_rules"), (3, "ask_bank")]
    llm = MockLLM(script=[plan, "finish[0.55]"])
    r = Rewoo(llm)
    assert isinstance(r, Investigator)
    v = r.investigate(ALERT, ScriptedToolBox([Reply("220", "bool", True, 0.5, 0.2)]), Limits())
    assert v.llm_calls == 2 and v.probability == 0.55
    assert [e.gate_decision for e in v.questions_log] == ["allowed", "over_budget"]  # 2.0 > 1.5 left
    solver_in = llm.requests[1]["messages"][1]["content"]
    assert "#E1 = ask_bank" in solver_in and "Evidence: bank 220 answered True" in solver_in


@pytest.mark.parametrize("text,p", [("finish[0.3]", 0.3), ("0.42", 0.42), ("35%", 0.35),
                                    ("ask_bank[0 | 0.5 | x]", None), ("I think 0.2 maybe", None), (None, None)])  # fmt: skip
def test_solver_probability_never_takes_a_stray_number(text, p):
    assert solver_probability(text) == p


def test_records_from_runs_gain_is_brier_change_per_answered_step():
    from ftm.library.build import records_from_runs

    sh = shape_from_rows(rows(), "R", T0).as_dict()
    base = {"run_id": "r", "alert_id": "V1", "eps": 0.5, "gate": "allowed", "response_kind": "reply"}
    steps = [{**base, "step": 1, "program_canonical": Q_PASS, "posterior": 0.5},
             {**base, "step": 2, "program_canonical": Q_FAN, "posterior": 0.5, "gate": "blocked", "response_kind": None},
             {**base, "step": 3, "program_canonical": "n_tx(x)", "posterior": 0.9}]  # fmt: skip
    alerts = {"V1": {"split": "val", "p0": 0.2, "label": 1, "shape": sh, "receiver": "R"}}
    recs = records_from_runs(steps, alerts, accepted=lambda c, e: c == Q_PASS)
    assert [r["question"] for r in recs] == [Q_PASS, "n_tx(x)"]
    assert recs[0]["gain"] == pytest.approx(0.8**2 - 0.5**2) and recs[0]["accepted"]
    assert recs[1]["gain"] == pytest.approx(0.5**2 - 0.1**2) and not recs[1]["accepted"]
    assert records_from_runs(steps, {"V1": {**alerts["V1"], "label": None}}, lambda c, e: True) == []
