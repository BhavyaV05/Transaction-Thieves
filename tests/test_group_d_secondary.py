"""T3 weeks 8-9: LATS (adapted), LLMCompiler, parallel time accounting, DSPy-ReAct."""

import pytest

from ftm.external.lats import Lats, failed, parse_scores
from ftm.external.llmcompiler import LlmCompiler, parse_tasks, waves
from ftm.harness.interface import Investigator, ScriptedToolBox
from ftm.llm.investigation import Investigation
from ftm.llm.mock import MockLLM
from ftm.planners.llm_dsl import ProvisionalDsl
from ftm.types import AlertContext, Decline, Limits, Reply

DSL = ProvisionalDsl()
Q1 = "passthrough(ALERT.receiver, min_frac=0.8, within_h=24)"
Q2 = "fan_in(ALERT.receiver, min_senders=5, window_h=48)"
ALERT = AlertContext("V9", "70", "2022-09-05T12:00:00", "val", 0.2,
                     {"receiver": "R", "evidence": "rows", "partner_banks": [220, 1588]})  # fmt: skip


def reply(bank, days):
    return Reply(bank=str(bank), kind="bool", value=True, eps_spent=0.5, latency_days=days)


def test_parallel_block_charges_slowest_answer():
    s = Investigation(ALERT, ScriptedToolBox([reply(220, 0.5), reply(1588, 2.0), reply(220, 1.0)]), Limits())
    with s.parallel():
        s.ask(220, DSL.validate(Q1), 0.5, 1)
        s.ask(1588, DSL.validate(Q2), 0.5, 2)
    assert s.sim_days == pytest.approx(2.0)
    s.ask(220, DSL.validate(Q2), 0.5, 3)
    assert s.sim_days == pytest.approx(3.0)
    with pytest.raises(RuntimeError), s.parallel(), s.parallel():
        pass


# -- LATS ---------------------------------------------------------------------------------------------------
def test_lats_scores_and_failure_detection():
    assert parse_scores("Candidate 2: 9\ncandidate 1: 3.5\nCandidate 7: 10", 3) == [3.5, 9.0, 1.0]
    assert failed("bank 220 declined to answer") and failed("Invalid question: x")
    assert not failed("bank 220 answered True (bool)\n<untrusted_note>request blocked</untrusted_note>")


def test_lats_executes_best_child_and_backtracks_on_failure():
    llm = MockLLM(script=[
        f"Thought: a\nAction: ask_bank[220 | 0.5 | {Q1}]",      # child 1
        "Thought: b\nAction: finish[0.9]",                       # child 2
        f"Thought: c\nAction: ask_bank[1588 | 0.5 | {Q2}]",     # child 3
        "Candidate 1: 9\nCandidate 2: 2\nCandidate 3: 7",        # value: 1 (declined) then backtrack to 3
        "Thought: d\nAction: finish[0.6]", "Thought: d\nAction: finish[0.6]", "Thought: e\nAction: finish[0.3]",
        "Candidate 1: 8\nCandidate 2: 4",
    ])  # fmt: skip
    lats = Lats(llm, n_children=3)
    assert isinstance(lats, Investigator)
    v = lats.investigate(ALERT, ScriptedToolBox([Decline("220"), reply(1588, 0.4)]), Limits())
    assert [e.bank for e in v.questions_log] == ["220", "1588"]  # 220 declined -> backtracked to child 3
    assert v.probability == 0.6 and v.llm_calls == 8
    assert llm.requests[0]["temperature"] == 0.7 and llm.requests[0]["seed"] != llm.requests[1]["seed"]


def test_lats_respects_call_cap():
    v = Lats(MockLLM(script=["Thought: x"] * 50), n_children=3).investigate(ALERT, ScriptedToolBox(), Limits(max_llm_calls=9))
    assert v.llm_calls <= 9 and v.probability is None


# -- LLMCompiler --------------------------------------------------------------------------------------------
PLAN = f"""1. ask_bank[220 | 0.5 | {Q1}]
2. ask_bank[1588 | 0.5 | {Q2}]
3. lookup_rules[sharing after $1]
4. join()
5. ask_bank[220 | 0.5 | ignored after join]"""


def test_parse_tasks_and_waves():
    tasks = parse_tasks(PLAN)
    assert sorted(tasks) == [1, 2, 3]
    assert waves(tasks) == [[1, 2], [3]]
    cyclic = parse_tasks("1. lookup_rules[$2]\n2. lookup_rules[$1]\n3. lookup_rules[x]")
    assert waves(cyclic) == [[3]]


def test_llmcompiler_parallel_waves_and_join():
    llm = MockLLM(script=[PLAN, "Thought: both yes.\nAction: finish[0.75]"])
    c = LlmCompiler(llm)
    assert isinstance(c, Investigator)
    tools = ScriptedToolBox([reply(220, 0.5), reply(1588, 2.0)])
    v = c.investigate(ALERT, tools, Limits())
    assert v.probability == 0.75 and v.llm_calls == 2 and len(v.questions_log) == 2
    assert "bank 220 answered True" in llm.requests[1]["messages"][1]["content"]
    assert "lookup_rules[sharing after bank 220 answered True" in llm.requests[1]["messages"][1]["content"]
    assert c.last_trace[0]["waves"] == [[1, 2], [3]]


def test_llmcompiler_replans_with_results_then_stops():
    plan2 = f"1. ask_bank[1588 | 0.5 | {Q2}]\n2. join()"
    llm = MockLLM(script=[f"1. ask_bank[220 | 0.5 | {Q1}]\n2. join()", "Action: replan, need payer count",
                          plan2, "Action: finish[0.4]"])  # fmt: skip
    v = LlmCompiler(llm, max_replans=1).investigate(ALERT, ScriptedToolBox([reply(220, 1), reply(1588, 1)]), Limits())
    assert v.probability == 0.4 and v.llm_calls == 4
    assert "Results so far" in llm.requests[2]["messages"][1]["content"]


# -- DSPy -----------------------------------------------------------------------------------------------------
def _dspy_step(tool, args):
    return (f"[[ ## next_thought ## ]]\nt\n\n[[ ## next_tool_name ## ]]\n{tool}\n\n"
            f"[[ ## next_tool_args ## ]]\n{args}\n\n[[ ## completed ## ]]")  # fmt: skip


def test_dspy_react_runs_through_our_budget_and_tools(tmp_path):
    pytest.importorskip("dspy")
    from ftm.external.dspy_react import DspyReact, brier_metric, build_program

    script = [_dspy_step("ask_bank", '{"bank": 220, "eps": 0.5, "question": "' + Q1 + '"}'),
              _dspy_step("finish", "{}"),
              "[[ ## reasoning ## ]]\nr\n\n[[ ## probability ## ]]\n0.62\n\n[[ ## completed ## ]]"]  # fmt: skip
    llm = MockLLM(script=script)
    v = DspyReact(llm).investigate(ALERT, ScriptedToolBox([reply(220, 0.3)]), Limits())
    assert v.probability == pytest.approx(0.62) and v.llm_calls == 3 and v.tokens_in > 0
    assert v.questions_log[0].program == DSL.validate(Q1).canonical
    capped = DspyReact(MockLLM(script=[_dspy_step("lookup_rules", '{"query": "x"}')] * 10)).investigate(
        ALERT, ScriptedToolBox(), Limits(max_llm_calls=3))
    assert capped.llm_calls <= 3 and capped.probability is None
    path = tmp_path / "prog.json"
    build_program(19).save(str(path))
    assert DspyReact(llm, program_path=str(path)).program is not None

    class P:
        probability = 0.25

    class E:
        label = 1

    assert brier_metric(E(), P()) == pytest.approx(1 - 0.75**2)


def test_dspy_unparseable_model_output_is_a_failed_alert_not_a_crash():
    pytest.importorskip("dspy")
    from ftm.external.dspy_react import DspyReact

    v = DspyReact(MockLLM()).investigate(ALERT, ScriptedToolBox(), Limits(max_llm_calls=4))  # mock answers "MOCK"
    assert v.probability is None and v.llm_calls >= 1
