"""The planner prompt budget (docs/t3_prompt_budget.md): menu ranking by expected Brier reduction, the evidence floor, the cap rule, the profile."""

import math

import pytest

from ftm.fusion.conditional import ConditionalFusion
from ftm.fusion.gain import Cells, UnitFusion, expected_gain
from ftm.graph.menu_rank import compact_lr, rank_menu, ranked_questions
from ftm.graph.policy import MenuItem
from ftm.llm.profile import load_profile, prompt_cap
from ftm.planners.llm_dsl import V0Dsl
from ftm.planners.llm_prompt import Mode, PlannerState, build_messages_with_estimate

DSL = V0Dsl()


def _item(prog, eps, lr, p_cell, fam="f", edges=(), qid="B1"):
    return MenuItem(qid, prog, eps, "11", fam, sorted(lr.items()), p_cell, edges)


GOOD = _item("good(ALERT.receiver)", 1.0, {"true": 3.0, "false": 0.5}, {"true": (0.1, 0.5), "false": (0.9, 0.5)}, qid="B1")
WEAK = _item("weak(ALERT.receiver)", 1.0, {"true": 1.3, "false": 0.8}, {"true": (0.4, 0.6), "false": (0.6, 0.4)}, qid="B2")
OVER = _item("over(ALERT.receiver)", 0.5, {"true": 9.0, "false": 0.1}, {"true": (0.5, 0.5), "false": (0.5, 0.5)}, qid="B3")  # overshoots an uninformative answer
NOCELL = _item("nocell(ALERT.receiver)", 0.5, {"true": 2.0, "false": 0.5}, None, qid="B4")


def test_rank_menu_orders_by_gain_and_drops_nonpositive_and_unmeasured():
    shown, st = rank_menu([WEAK, OVER, NOCELL, GOOD], None, 0.3, 2.0, set(), 10)
    assert [m.qid for m in shown][0] == "B1" and "B3" not in [m.qid for m in shown] and "B4" not in [m.qid for m in shown]
    assert st["dropped_nonpositive"] == 1 and st["dropped_no_cells"] == 1 and st["offered"] == 4
    assert st["gains"] == sorted(st["gains"], reverse=True)


def test_rank_menu_top_k_asked_and_afford():
    shown, st = rank_menu([GOOD, WEAK], None, 0.3, 2.0, set(), 1)
    assert [m.qid for m in shown] == ["B1"] and st["beyond_top_k"] == 1
    shown, st = rank_menu([GOOD, WEAK], None, 0.3, 2.0, {"good(ALERT.receiver)"}, 5)
    assert [m.qid for m in shown] == ["B2"]  # asked questions are not menu items
    shown, st = rank_menu([GOOD, WEAK], None, 0.3, 0.5, set(), 5)
    assert shown == [] and st["offered"] == 0  # eps 1 is unaffordable


def test_gain_is_the_one_system1_uses():
    from ftm.orchestrator import decision_set

    assert decision_set.expected_gain is expected_gain
    m = ConditionalFusion(families=["f"], w=__import__("numpy").array([0.5]), shrink=0.5, fitted=True)
    z = 0.4
    g = expected_gain(m, z, Cells("f", dict(GOOD.lr_bins), GOOD.p_cell))
    shown, st = rank_menu([GOOD], m, 1 / (1 + math.exp(-z)), 2.0, set(), 3)
    assert st["gains"][0] == pytest.approx(round(g, 6)) and expected_gain(UnitFusion(), z, Cells("f", {}, None)) is None


def test_compact_lr_encodings():
    assert compact_lr(GOOD) == "yes:3.00 no:0.50"
    b = _item("n(ALERT.receiver)", 1.0, {"b0": 0.5, "b1": 1.25, "b2": 2.0}, None, edges=(0.0, 4.0))
    assert compact_lr(b) == "<0:0.50 0:1.25 4:2.00"
    c = _item("c(ALERT.receiver)", 1.0, {"0": 0.9, "1": 1.1}, None)
    assert compact_lr(c) == "0:0.90 1:1.10"


def test_ranked_questions_one_entry_per_question_with_kept_levels():
    lo = _item("good(ALERT.receiver)", 0.5, {"true": 1.5, "false": 0.8}, {"true": (0.3, 0.4), "false": (0.7, 0.6)}, qid="B1")
    raw = [{"id": "B1", "program": "good(ALERT.receiver)", "family": "f", "eps": [0.5, 1.0], "bank": 11}]
    q = ranked_questions(raw, [GOOD, lo])
    assert len(q) == 1 and q[0]["eps"] == [1.0, 0.5] and q[0]["compact"]
    assert q[0]["lr_text"] == "eps 1: yes:3.00 no:0.50 | eps 0.5: yes:1.50 no:0.80"


def _state(rows, questions=()):
    return PlannerState(alert_summary={"amount": 1.0}, evidence_text="\n".join(f"2022-09-0{i % 9 + 1} 10:00 | A{i:07d} -> A{i + 1:07d} | 10.00 Euro | ACH | bank 1 -> bank 2" for i in range(rows)),
                        partner_banks=[2], eps_left=2.0, calls_left=5,
                        questions=[{"id": f"B{i}", "program": f"n_tx(ALERT.receiver, window=after({i}))", "family": "f", "eps": [1.0], "bank": 2,
                                    "lr_text": "eps 1: yes:2.00 no:0.50", "compact": True} for i in range(questions)],
                        posterior=0.4)


def test_evidence_floor_cuts_menu_before_trail_rows():
    st = _state(30, questions=20)
    fit_old: dict = {}
    _, est = build_messages_with_estimate(Mode.COMPOSE, st, DSL, 100000, 1.0, fit_old)
    cap = est - 300
    legacy: dict = {}
    build_messages_with_estimate(Mode.COMPOSE, st, DSL, cap, 1.0, legacy, 0)
    floor: dict = {}
    build_messages_with_estimate(Mode.COMPOSE, st, DSL, cap, 1.0, floor, 30)
    assert legacy["evidence_rows"] < 30 and legacy["questions_kept"] == 20  # legacy: the trail goes first
    assert floor["evidence_rows"] == 30 and floor["questions_kept"] < 20 and not floor["evidence_floor_broken"]


def test_evidence_floor_breaks_only_as_a_last_resort():
    st = _state(30, questions=3)
    rep: dict = {}
    msgs, est = build_messages_with_estimate(Mode.COMPOSE, st, DSL, 4000, 1.0, rep, 30)
    assert est <= 4000 and rep["questions_kept"] == 1 and rep["evidence_floor_broken"] and rep["evidence_rows"] < 30


def test_prompt_cap_rule_and_profile():
    assert prompt_cap() == 6750 == load_profile("tuned")["max_prompt_tokens"]
    assert prompt_cap(max_model_len=4096) < prompt_cap()
    assert load_profile("tuned")["menu_top_k"] == 8 and load_profile("tuned")["min_evidence_rows"] == 40
    assert load_profile("none") == {}  # legacy: 6,000 cap, whole menu, old trimming order


def test_planner_with_a_budget_shows_the_ranked_compact_menu_and_legacy_is_unchanged():
    from ftm.harness.interface import ScriptedToolBox
    from ftm.llm.backtest_table import accepted_menu
    from ftm.llm.investigation import LREntry, LRTable, NaiveLRFusion
    from ftm.llm.mock import MockLLM
    from ftm.planners.llm import LLMPlanner
    from ftm.types import AlertContext, Limits
    from tests.test_planner_quality import SUMMARY, canon

    dorm, new = "dormant(ALERT.receiver, days=7)", "new_account(ALERT.receiver, days=2)"
    t = LRTable([LREntry(canon(dorm), 1.0, True, {"true": 0.5, "false": 1.4}, "dormant", None, {"true": (0.3, 0.1), "false": (0.7, 0.9)}),
                 LREntry(canon(new), 2.0, True, {"true": 9.0, "false": 0.1}, "new_account", None, {"true": (0.5, 0.5), "false": (0.5, 0.5)})])
    menu = accepted_menu(t)
    al = AlertContext("V1", "70", "2022-09-05T12:00:00", "val", 0.3, {**SUMMARY, "sender_bank": 70, "receiver_bank": 220})

    def first_prompt(**kw):
        llm = MockLLM(script=['{"hypothesis": {"typology": "none", "rationale": "r"}, "action": "stop", "question_id": null, "program": null, '
                              '"bank": null, "eps": 0.0, "rule_id": null}'])
        p = LLMPlanner(llm, questions=menu, fusion=NaiveLRFusion(t), **kw)
        p.investigate(al, ScriptedToolBox(), Limits())
        return llm.requests[0]["messages"][1]["content"], p

    ranked, p = first_prompt(menu_top_k=5)
    legacy, _ = first_prompt()
    assert "dormant" in ranked and "new_account" not in ranked.split("## Back-tested")[1].split("## Question language")[0]  # overshooting question dropped
    assert "eps 1: yes:0.50 no:1.40" in ranked and "LR yes x0.50, no x1.40" in legacy
    assert p.stats.as_dict()["menu_budget"]["dropped_nonpositive"] == 1.0
