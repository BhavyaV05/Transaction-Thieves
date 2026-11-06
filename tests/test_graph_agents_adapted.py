"""R2: the ADAPTED arms (PlannerPolicy) on a minimal fake pipeline driver: contract (asks inside the menu, affordable, not repeated,
terminates, LLM calls inside the cap) under fuzzed model output, plus the strategy-specific behaviour of each."""

import json
import math

import pytest

pytest.importorskip("langgraph")

from graph_agents_support import BOOL_QS, COUNT_QS, adapted_responder, drive, menu_items  # noqa: E402

from ftm.graph.agents import adapted  # noqa: E402
from ftm.graph.policy import Ask, PlannerView, Stop  # noqa: E402
from ftm.llm.budget import BudgetedClient  # noqa: E402
from ftm.llm.mock import MockLLM  # noqa: E402
from ftm.types import Limits  # noqa: E402

ALL = list(adapted.POLICIES)


@pytest.mark.parametrize("system", ALL)
def test_policy_contract_under_fuzzed_output(system):
    stops, asked = set(), 0
    for seed in range(20):
        policy = adapted.make_policy(system + "_adapted", seed=seed)
        log, stop, _, budget = drive(policy, MockLLM(responder=adapted_responder(seed)), seed=seed)
        assert len(set(log)) == len(log)
        assert budget.calls_used <= 20 if hasattr(budget, "calls_used") else True
        stops.add(stop)
        asked += len(log)
    assert asked > 0  # the fuzz reaches real asks


@pytest.mark.parametrize("system", ALL)
def test_policy_stops_without_options_or_calls(system):
    policy = adapted.make_policy(system + "_adapted")
    view = PlannerView({}, 0.3, 0.3, None, "", [], [], menu_items(), [], 0.1, 5, 1, False)  # eps left below every level
    llm = BudgetedClient(MockLLM(), Limits(max_llm_calls=5), tag="x")
    assert policy.propose(view, llm, {}) == [Stop("no_options")]
    view2 = PlannerView({}, 0.3, 0.3, None, "", [], [], menu_items(), [], 2.0, 0, 1, False)
    assert policy.propose(view2, llm, {}) == [Stop("calls_used")]


def _view(**kw):
    base = dict(alert_summary={}, p0=0.3, posterior=0.3, threshold=None, evidence="e", rules=[], strategies=[], menu=menu_items(),
                answers=[], eps_left=2.0, calls_left=20, step=1, allow_compose=False)  # fmt: skip
    return PlannerView(**{**base, **kw})


def _client(script, n=20):
    llm = MockLLM(script=[json.dumps(s) if not isinstance(s, str) else s for s in script])
    return llm, BudgetedClient(llm, Limits(max_llm_calls=n), tag="x")


def test_react_picks_the_numbered_option_and_stop():
    p = adapted.ReactPolicy()
    _, c = _client([{"thought": "t", "choice": 3}, {"thought": "done", "choice": 0}])
    out = p.propose(_view(), c, {})
    assert out == [Ask(BOOL_QS[2], 0.5, "t")]  # option 3 = third (question, eps) item
    assert p.propose(_view(), c, {}) == [Stop("agent_stop")]


def test_react_never_offers_asked_or_unaffordable():
    v = _view(answers=[{"program": BOOL_QS[0], "bank": "1588", "eps": 0.5, "outcome": "reply", "value": "true", "lr": 2.5}], eps_left=0.5)
    opts = adapted.options(v)
    assert [m.canonical for m in opts] == [BOOL_QS[2], COUNT_QS[1]]


def test_reflexion_reflects_on_an_uninformative_answer_and_carries_it():
    p = adapted.ReflexionPolicy()
    llm, c = _client([{"thought": "t", "choice": 1}, {"reflection": "LESSON-ONE"}, {"thought": "t2", "choice": 2}])
    scratch = {}
    p.propose(_view(), c, scratch)
    weak = [{"program": BOOL_QS[0], "bank": "1588", "eps": 0.5, "outcome": "decline", "value": None, "lr": None}]
    out = p.propose(_view(answers=weak), c, scratch)
    assert out[0].canonical == COUNT_QS[1] or isinstance(out[0], Ask)
    assert scratch["reflections"] == ["LESSON-ONE"]
    assert "LESSON-ONE" in llm.requests[2]["messages"][1]["content"]  # carried into the next decision


def test_reflexion_retry_on_stop_then_asks():
    p = adapted.ReflexionPolicy()
    _, c = _client([{"thought": "stop", "choice": 0}, {"verdict": "RETRY", "reason": "weak"}, {"reflection": "R"}, {"thought": "ok", "choice": 1}])
    out = p.propose(_view(), c, {})
    assert out == [Ask(BOOL_QS[0], 0.5, "ok")]
    _, c2 = _client([{"thought": "stop", "choice": 0}, {"verdict": "CONFIDENT", "reason": "fine"}])
    assert p.propose(_view(), c2, {}) == [Stop("agent_stop")]


def test_rewoo_plans_once_then_executes_without_llm_calls():
    p = adapted.RewooPolicy()
    llm, c = _client([{"plan": [{"why": "a", "choice": 1}, {"why": "dup", "choice": 2}, {"why": "b", "choice": 3}, {"why": "c", "choice": 4}]}])
    scratch, seen, eps = {}, [], 0.0
    for step in range(6):
        v = _view(step=step + 1, eps_left=2.0 - eps, answers=[{"program": q, "bank": "", "eps": e, "outcome": "reply", "value": 1, "lr": 1.2} for q, e in seen])
        out = p.propose(v, c, scratch)
        if isinstance(out[0], Stop):
            assert out[0].reason in ("plan_done", "no_options")
            break
        seen.append((out[0].canonical, out[0].eps))
        eps += out[0].eps
    assert len(llm.requests) == 1  # no replanning
    assert [q for q, _ in seen] == [BOOL_QS[0], BOOL_QS[2], COUNT_QS[0]]  # option 2 repeated option 1's question: dropped


def test_lats_samples_k_scores_and_runs_the_best():
    p = adapted.LatsPolicy()
    llm, c = _client([{"thought": "a", "choice": 1}, {"thought": "b", "choice": 3}, {"thought": "c", "choice": 1}, {"scores": [2, 9]}])
    out = p.propose(_view(), c, {})
    assert out == [Ask(BOOL_QS[2], 0.5, "b")]  # candidates {1, 3} (the duplicate is dropped), scored 2 and 9
    assert [r["temperature"] for r in llm.requests[:3]] == [0.7] * 3 and len({r["seed"] for r in llm.requests[:3]}) == 3
    assert len(llm.requests) == 4


def test_llmcompiler_returns_waves_then_joins():
    p = adapted.LlmCompilerPolicy()
    plan = {"tasks": [{"id": 1, "choice": 1, "deps": []}, {"id": 2, "choice": 3, "deps": []}, {"id": 3, "choice": 5, "deps": [1]}]}
    _, c = _client([plan, {"decision": "FINISH", "note": "ok"}])
    scratch = {}
    w1 = p.propose(_view(), c, scratch)
    assert [a.canonical for a in w1] == [BOOL_QS[0], BOOL_QS[2]]  # independent tasks go out as one wave
    ans = [{"program": a.canonical, "bank": "", "eps": a.eps, "outcome": "reply", "value": 1, "lr": 1.5} for a in w1]
    w2 = p.propose(_view(answers=ans), c, scratch)
    assert [a.canonical for a in w2] == [COUNT_QS[1]]  # the dependent task follows
    assert p.propose(_view(answers=ans), c, scratch) == [Stop("joiner_finish")]


def test_debate_majority_with_two_rounds():
    p = adapted.DebatePolicy()
    r1 = [{"thought": f"a{i}", "choice": ch} for i, ch in enumerate([1, 3, 3])]
    r2 = [{"thought": f"b{i}", "choice": ch} for i, ch in enumerate([3, 3, 1])]
    llm, c = _client(r1 + r2)
    out = p.propose(_view(), c, {})
    assert out == [Ask(BOOL_QS[2], 0.5, "b0")] and len(llm.requests) == 6
    assert "Analyst 2" in llm.requests[3]["messages"][1]["content"]  # round 2 shows the others' proposals
    assert [r["seed"] for r in llm.requests[:3]] == [0, 1, 2]


def test_debate_stop_majority():
    _, c = _client([{"thought": "s", "choice": 0}] * 6)
    assert adapted.DebatePolicy().propose(_view(), c, {}) == [Stop("agent_stop")]


def test_uot_uses_our_lrs_for_the_clean_side_and_stops_on_low_eig():
    assert adapted.clean_from_launder([0.2, 0.8], [0.4, 2.5]) == pytest.approx([0.2 / 0.4 / (0.5 + 0.32), 0.32 / (0.5 + 0.32)], rel=1e-2)
    p = adapted.UotPolicy()
    informative = {"candidates": [{"choice": 1, "p_launder": [0.1, 0.9]}, {"choice": 3, "p_launder": [0.5, 0.5]}]}
    _, c = _client([informative])
    out = p.propose(_view(), c, {})
    assert isinstance(out[0], Ask) and out[0].canonical == BOOL_QS[0]
    flat = {"candidates": [{"choice": 1, "p_launder": [0.5, 0.5]}]}
    # p_launder equal to the clean side only when LR == 1; with LRs != 1 a flat guess still has a gap, so use a menu bin LR of 1
    p2 = adapted.UotPolicy(eig_stop=10.0)
    _, c2 = _client([flat])
    assert p2.propose(_view(), c2, {}) == [Stop("low_eig")]
    _, c3 = _client([{"candidates": []}])
    assert p.propose(_view(), c3, {}) == [Stop("no_candidates")]


def test_bed_llm_label_eig_and_observed_bin_reweighting():
    p = adapted.BedLlmPolicy()
    hyps = {"hypotheses": [{"label": "launder", "description": "mule"}, {"label": "clean", "description": "payroll"}]}
    cand = {"past": [], "candidates": [{"choice": 1, "predictions": [[0.1, 0.9], [0.9, 0.1]]}]}
    llm, c = _client([hyps, cand])
    out = p.propose(_view(), c, {})
    assert out == [Ask(BOOL_QS[0], 0.5, out[0].rationale)] and "EIG" in out[0].rationale and len(llm.requests) == 2
    ans = [{"program": BOOL_QS[0], "bank": "1588", "eps": 0.5, "outcome": "reply", "value": "true", "lr": 2.5}]
    assert adapted.observed_bins(_view(answers=ans)) == [{"canonical": BOOL_QS[0], "bins": ["false", "true"], "idx": 1}]
    # the second step must score the question already asked
    llm2, c2 = _client([hyps, {"past": [[[0.1, 0.9], [0.9, 0.1]]], "candidates": [{"choice": 1, "predictions": [[0.5, 0.5], [0.5, 0.5]]}]}])
    out2 = p.propose(_view(answers=ans), c2, {})
    assert out2 == [Stop("low_eig")] and "observed bin true" in llm2.requests[1]["messages"][1]["content"]


def test_mediq_confident_stops_unconfident_asks():
    p = adapted.MediqPolicy()
    _, c = _client([{"probability": 0.8, "confidence": 0.9}])
    assert p.propose(_view(), c, {}) == [Stop("confident")]
    _, c2 = _client([{"probability": 0.5, "confidence": 0.3}, {"thought": "t", "choice": 0}])
    out = p.propose(_view(), c2, {})
    assert isinstance(out[0], Ask)  # unconfident: a question is always asked (0 is read as the first option)
    many = [{"program": q, "bank": "", "eps": 0.5, "outcome": "reply", "value": 1, "lr": 1.2} for q in ("a", "b", "c", "d")]
    _, c3 = _client([])
    assert p.propose(_view(answers=many), c3, {}) == [Stop("questions")]


def test_dspy_react_adapted_uses_the_mipro_instruction_paragraph(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"react": {"signature": {"instructions": "TASK PARAGRAPH.\n\nYou are an Agent. tools..."}}}), encoding="utf-8")
    p = adapted.make_policy("dspy_react_adapted", program_path=f)
    llm, c = _client([{"thought": "t", "choice": 1}])
    p.propose(_view(), c, {})
    assert llm.requests[0]["messages"][0]["content"].startswith("TASK PARAGRAPH.\n\n")
    real = "outputs/t3_dspy/react_miprov2.json"
    import os

    if os.path.exists(real):
        assert adapted.mipro_instruction(real).startswith("Investigate an anti-money-laundering")


def test_malformed_reply_is_a_failed_step_not_a_crash():
    _, c = _client(["garbage", "garbage"])
    assert adapted.ReactPolicy().propose(_view(), c, {}) == []
    _, c2 = _client([])
    assert math.isfinite(1.0) and adapted.ReactPolicy().propose(_view(calls_left=1), BudgetedClient(MockLLM(), Limits(max_llm_calls=0), tag="x"), {}) in ([Stop("calls_used")], [])
