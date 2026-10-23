"""T3 planner-quality fixes after the first real runs: select menu from the back-test table, retry feedback with
examples, refusal handling and the stop rule, ReAct wrap-up call. Synthetic data only."""

import json

from ftm.external.react import React
from ftm.harness.interface import ScriptedToolBox
from ftm.llm.backtest_table import ACCOUNT_SLOTS, accepted_menu, applicable_to
from ftm.llm.investigation import LREntry, LRTable, NaiveLRFusion
from ftm.llm.mock import MockLLM
from ftm.planners.llm import LLMPlanner
from ftm.planners.llm_dsl import V0Dsl, examples_by_function, retry_hint
from ftm.planners.llm_prompt import Mode, PlannerState, build_messages
from ftm.types import AlertContext, Limits, Reply

DSL = V0Dsl()
Q_DORM = "dormant(ALERT.receiver, days=7)"
Q_NEW = "new_account(ALERT.sender, days=2)"
Q_SET = "overlap(ALERT.receiver, dir=out, window=after(72), among=ALERT.payers)"
SUMMARY = {"sender": "A0000002", "receiver": "A0000001", "amount": 9500.0, "evidence": "row1\nrow2",
           "partner_banks": [220, 1588]}  # fmt: skip


def alert(p0=0.2):
    return AlertContext("V000001", "70", "2022-09-05T12:00:00", "val", p0, SUMMARY)


def canon(q):
    return DSL.validate(q).canonical


def table():
    return LRTable([
        LREntry(canon(Q_DORM), 1.0, True, {"true": 0.6, "false": 1.5}, "dormant"),
        LREntry(canon(Q_DORM), 2.0, True, {"true": 0.4, "false": 1.9}, "dormant"),
        LREntry(canon(Q_NEW), 2.0, True, {"true": 0.3, "false": 1.8}, "new_account"),
        LREntry(canon(Q_NEW), 0.5, False, {"true": 1.0, "false": 1.0}, "new_account"),
        LREntry(canon(Q_SET), 2.0, True, {"b0": 0.5, "b1": 2.0}, "overlap:out", (1.0,)),
        LREntry(canon("passed_on(ALERT.receiver, window=after(24), min_frac=0.8)"), 0.5, False, {"true": 3.0}, "passed_on"),
    ])  # fmt: skip


def step(action="compose", program=Q_DORM, bank=220, eps=1.0, qid=None):
    return json.dumps({"hypothesis": {"typology": "fan_in", "rationale": "r"}, "action": action, "question_id": qid,
                       "program": program if action == "compose" else None, "bank": bank if action != "stop" else None,
                       "eps": eps if action != "stop" else 0.0, "rule_id": None})  # fmt: skip


STOP = step("stop")


def reply(v, eps=1.0, days=0.5):
    return Reply(bank="220", kind="bool", value=v, eps_spent=eps, latency_days=days, note=None)


class GateBlocked(PermissionError):
    pass


class RefusingTools(ScriptedToolBox):
    def ask_bank(self, bank, program, eps):
        self.calls.append((bank, program, eps))
        raise GateBlocked(f"bank {bank} is not a partner bank")


# -- the select menu --------------------------------------------------------------------------------------
def test_accepted_menu_merges_eps_keeps_only_accepted_and_has_stable_ids():
    m = accepted_menu(table())
    assert [x["program"] for x in m] == sorted((x["program"] for x in m), key=lambda p: next(i["family"] for i in m if i["program"] == p) + p)
    assert {x["program"] for x in m} == {canon(Q_DORM), canon(Q_NEW), canon(Q_SET)}  # rejected rows are not offered
    dorm = next(x for x in m if x["program"] == canon(Q_DORM))
    assert dorm["eps"] == [1.0, 2.0] and dorm["family"] == "dormant" and "yes x0.60" in dorm["lr_text"] and "no x1.50" in dorm["lr_text"]
    assert [x["id"] for x in m] == ["B01", "B02", "B03"] and accepted_menu(table()) == m


def test_menu_applicability_follows_slot_binding():
    m = {x["program"]: x for x in accepted_menu(table())}
    assert applicable_to(m[canon(Q_DORM)], ACCOUNT_SLOTS)
    assert not applicable_to(m[canon(Q_SET)], ACCOUNT_SLOTS)  # needs ALERT.payers, not bound for this alert
    assert applicable_to(m[canon(Q_SET)], ACCOUNT_SLOTS | {"payers"})


def _state(menu, **kw):
    return PlannerState(alert_summary={"amount": 1.0}, evidence_text="row", partner_banks=[220], eps_left=2.0, calls_left=20,
                        questions=menu, **kw)  # fmt: skip


def test_compose_prompt_shows_menu_with_eps_and_lr_but_no_backtest_shows_none():
    menu = accepted_menu(table())
    user = build_messages(Mode.COMPOSE, _state(menu), DSL, 6000)[1]["content"]
    assert "## Back-tested questions" in user and "B01" in user and "ask at eps 1 or 2" in user and "LR yes" in user
    plain = build_messages(Mode.NO_BACKTEST, _state([{k: x[k] for k in ("id", "program", "family")} for x in menu]), DSL, 6000)[1]["content"]
    assert "Back-tested" not in plain and "LR " not in plain and "ask at eps" not in plain
    # the planner itself strips back-test fields in the ablation, whatever the caller passes
    p = LLMPlanner(MockLLM(script=[STOP]), mode="no_backtest", questions=menu)
    assert all(set(q) <= {"id", "program", "family", "note"} for q in p.questions.values())
    p.investigate(alert(), ScriptedToolBox(), Limits())
    assert "LR " not in p.llm.requests[0]["messages"][1]["content"]


def test_select_asks_at_a_back_tested_eps_and_moves_the_posterior():
    menu = accepted_menu(table())
    qid = next(x["id"] for x in menu if x["program"] == canon(Q_DORM))
    llm = MockLLM(script=[step("select", qid=qid, eps=0.5), STOP])  # 0.5 was not back-tested for this question
    tools = ScriptedToolBox([reply(False)])
    p = LLMPlanner(llm, questions=menu, fusion=NaiveLRFusion(table()))
    v = p.investigate(alert(0.2), tools, Limits())
    assert tools.calls[0][2] == 1.0 and p.last_trace[0]["eps_requested"] == 0.5  # moved to the nearest accepted level
    assert v.probability > 0.2  # 'no' to dormant has LR 1.5 at eps 1: the question counted


def test_select_with_too_little_eps_gets_feedback_and_no_call():
    menu = [x for x in accepted_menu(table()) if x["eps"] == [2.0] and x["family"] == "new_account"]
    llm = MockLLM(script=[step("select", qid=menu[0]["id"], eps=1.0), STOP])
    tools = ScriptedToolBox([reply(True)])
    LLMPlanner(llm, questions=menu).investigate(alert(), tools, Limits(eps_per_alert=1.0))
    assert tools.calls == [] and "needs eps 2" in llm.requests[1]["messages"][1]["content"]


def test_question_filter_hides_inapplicable_questions():
    menu = accepted_menu(table())
    llm = MockLLM(script=[STOP])
    LLMPlanner(llm, questions=menu, question_filter=lambda q, c: applicable_to(q, ACCOUNT_SLOTS)).investigate(
        alert(), ScriptedToolBox(), Limits())  # fmt: skip
    user = llm.requests[0]["messages"][1]["content"]
    assert "dormant(ALERT.receiver" in user and "overlap(" not in user.split("## Question language")[0]


# -- retry feedback ---------------------------------------------------------------------------------------
def test_examples_by_function_and_retry_hint():
    ex = examples_by_function(DSL.reference_text())
    assert ex["n_counterparties"].startswith("n_counterparties(ALERT.") and "dormant" in ex
    h = retry_hint("n_counterparties", "syntax error", DSL.reference_text())
    assert "n_counterparties(ALERT.receiver, dir=in, window=before(72))" in h
    h2 = retry_hint("n_payees_pay_each_other(ALERT.sender, window=after(72))", "unknown", DSL.reference_text())
    assert "payees_pay_each_other(ALERT.receiver, window=after(72))" in h2


def test_planner_feedback_after_bare_function_name_has_an_example():
    llm = MockLLM(script=[step(program="n_counterparties"), STOP])
    LLMPlanner(llm).investigate(alert(), ScriptedToolBox(), Limits())
    fb = llm.requests[1]["messages"][1]["content"]
    assert "invalid question 'n_counterparties'" in fb and "n_counterparties(ALERT.receiver, dir=in" in fb


# -- refusals and the stop rule ---------------------------------------------------------------------------
def test_refusals_count_as_failures_with_reason_and_stop_the_loop():
    llm = MockLLM(script=[step(bank=220), step(bank=220), step(bank=220), step(bank=220)])
    p = LLMPlanner(llm)
    v = p.investigate(alert(), RefusingTools(), Limits())
    # first ask reaches the gate (refused), the repeats are caught before the gate, 3 failures in a row end the loop
    assert v.llm_calls == 3 and p.stats.refusals == 1 and p.stats.refused_repeats == 2
    assert "is not a partner bank" in llm.requests[1]["messages"][1]["content"]
    assert len(v.questions_log) == 1


def test_refusal_does_not_reset_the_failure_count_but_a_reply_does():
    class Mixed(ScriptedToolBox):
        def ask_bank(self, bank, program, eps):
            if str(bank) == "1588":
                raise GateBlocked("no")
            return super().ask_bank(bank, program, eps)

    llm = MockLLM(script=[step(bank=1588, eps=0.5), step(program="n_tx(ALERT.sender, dir=in, window=before(72))", bank=1588, eps=0.5),
                          step(program=Q_NEW, bank=220, eps=0.5), STOP])  # fmt: skip
    v = LLMPlanner(llm).investigate(alert(), Mixed([reply(True)]), Limits())
    assert len(v.questions_log) == 3 and v.llm_calls == 4


# -- ReAct ------------------------------------------------------------------------------------------------
def test_react_wrap_up_asks_for_finish_once_when_time_runs_out():
    llm = MockLLM(script=["Thought 1: ask\nAction 1: ask_bank[220 | 1.0 | " + Q_DORM + "]", "Thought 2: so\nAction 2: finish[0.4]"])
    v = React(llm).investigate(alert(), ScriptedToolBox([reply(True, days=6.0)]), Limits())  # 6 days > the 5-day clock
    assert v.probability == 0.4 and v.llm_calls == 2
    assert "No further question can be asked" in llm.requests[1]["messages"][-1]["content"]


def test_react_wrap_up_never_exceeds_the_call_cap_and_keeps_none_when_no_finish():
    llm = MockLLM(script=["Thought: hmm"] * 4)
    v = React(llm).investigate(alert(), ScriptedToolBox(), Limits(max_llm_calls=2))
    assert v.probability is None and v.llm_calls == 2
    v2 = React(MockLLM(script=["Thought 1: a\nAction 1: ask_bank[220 | 1.0 | " + Q_DORM + "]", "not a finish"])).investigate(
        alert(), ScriptedToolBox([reply(True, days=6.0)]), Limits())  # fmt: skip
    assert v2.probability is None and v2.llm_calls == 2


def test_react_invalid_question_observation_has_example_and_refusal_reason():
    llm = MockLLM(script=["Action 1: ask_bank[220 | 1.0 | n_tx]", "Action 2: ask_bank[220 | 1.0 | " + Q_DORM + "]",
                          "Action 3: finish[0.3]"])  # fmt: skip
    v = React(llm).investigate(alert(), RefusingTools(), Limits())
    convo = llm.requests[-1]["messages"]
    assert "n_tx(ALERT.receiver" in convo[3]["content"] and "Reason: bank 220 is not a partner bank" in convo[5]["content"]
    assert v.probability == 0.3


def test_investigation_keeps_the_refusal_message():
    from ftm.llm.investigation import Investigation

    s = Investigation(alert(), RefusingTools(), Limits())
    e = s.ask(220, DSL.validate(Q_DORM), 1.0, 1)
    assert e.gate_decision == "blocked" and s.last_refusal == "bank 220 is not a partner bank"


def test_modelled_banks_filters_partner_list_to_the_harness_world():
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_llm_agent.py"
    spec = importlib.util.spec_from_file_location("run_llm_agent", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tools = SimpleNamespace(_h=SimpleNamespace(banks={0: object(), 11: object(), 115: object()}))
    assert mod.modelled_banks(tools) == {"0", "11", "115"}
