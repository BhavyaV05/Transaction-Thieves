"""T3 weeks 8-9: LATS (adapted), LLMCompiler, parallel time accounting, DSPy-ReAct."""

import json

import pytest

from ftm.external.lats import Lats, failed, parse_scores
from ftm.external.llmcompiler import LlmCompiler, parse_tasks, waves
from ftm.harness.interface import Investigator, ScriptedToolBox
from ftm.llm.investigation import Investigation
from ftm.llm.mock import MockLLM
from ftm.planners.llm_dsl import V0Dsl
from ftm.types import AlertContext, Decline, Limits, Reply

DSL = V0Dsl()
Q1 = "passed_on(ALERT.receiver, window=after(24), min_frac=0.8)"
Q2 = "at_least(n_counterparties(ALERT.receiver, dir=in, window=before(48)), k=5)"
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


def test_lats_stale_sibling_is_dropped_after_a_successful_step():
    stale = "Thought: old\nAction: finish[0.99]"
    llm = MockLLM(script=[
        f"Thought: a\nAction: ask_bank[220 | 0.5 | {Q1}]", stale,      # step 1 children
        "Candidate 1: 9\nCandidate 2: 8",                              # the sibling outscores the later children
        "Thought: c\nAction: finish[0.3]", "Thought: d\nAction: finish[0.2]",  # step 2 children
        "Candidate 1: 1\nCandidate 2: 1",
    ])  # fmt: skip
    v = Lats(llm, n_children=2).investigate(ALERT, ScriptedToolBox([reply(220, 0.4)]), Limits())
    assert v.probability == 0.3 and v.llm_calls == 6  # the old sibling (0.99) never ran


def test_lats_reexpands_when_all_siblings_fail():
    llm = MockLLM(script=[
        f"Thought: a\nAction: ask_bank[220 | 0.5 | {Q1}]", f"Thought: b\nAction: ask_bank[220 | 0.5 | {Q2}]",
        "Candidate 1: 9\nCandidate 2: 8",
        "Thought: c\nAction: finish[0.4]", "Thought: d\nAction: finish[0.5]", "Candidate 1: 9\nCandidate 2: 1",
    ])  # fmt: skip
    v = Lats(llm, n_children=2).investigate(ALERT, ScriptedToolBox([Decline("220"), Decline("220")]), Limits())
    assert v.probability == 0.4 and v.llm_calls == 6


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


def _dspy_mipro_responder(messages, kw):
    last = str(messages[-1].get("content"))
    if "program_code" in last:  # MIPROv2's instruction proposer (prompt model)
        return json.dumps({k: "x" for k in ("proposed_instruction", "observations", "summary", "program_description",
                                            "module_description", "program_example", "task_demos", "basic_instruction")})
    if "next_tool_name" in last:
        return _dspy_step("finish", "{}")
    return "[[ ## reasoning ## ]]\nr\n\n[[ ## probability ## ]]\n0.4\n\n[[ ## completed ## ]]"


def test_dspy_mipro_optimize_saves_a_program_that_dspy_react_loads(tmp_path):
    """The optimiser path of scripts/optimize_dspy.py on the mock: val `fit` examples through a harness-style session
    (begin/end per alert), instruction search only, a compute log, and a saved program run_llm_agent --dspy-program loads."""
    pytest.importorskip("dspy")
    pytest.importorskip("optuna")
    from ftm.external.dspy_react import DspyReact, optimize

    class Session(ScriptedToolBox):
        began = ended = 0

        def begin(self, alert, limits):
            type(self).began += 1

        def end(self):
            type(self).ended += 1

    tools = Session()
    ex = [{"alert": AlertContext(f"V{i:06d}", "70", "2022-09-05T12:00:00", "val", 0.2,
                                 {"sender": "A1", "receiver": "A2", "amount": 1.0, "evidence": "r1\nr2", "partner_banks": [220]}),
           "tools": tools, "label": i % 2, "part": "fit"} for i in range(8)]  # fmt: skip
    log = tmp_path / "compute.jsonl"
    program = optimize(MockLLM(responder=_dspy_mipro_responder), ex, auto="light", compute_log=log)
    assert Session.began == Session.ended > 0
    path = tmp_path / "react_miprov2.json"
    program.save(str(path))
    rec = json.loads(log.read_text().splitlines()[-1])
    assert rec["part"] == "fit" and rec["llm_calls"] > 0 and rec["n_examples"] == 8
    v = DspyReact(MockLLM(script=[_dspy_step("finish", "{}"),
                                  "[[ ## reasoning ## ]]\nr\n\n[[ ## probability ## ]]\n0.3\n\n[[ ## completed ## ]]"]),
                  program_path=str(path)).investigate(ALERT, ScriptedToolBox(), Limits())
    assert v.probability == pytest.approx(0.3)


def test_run_llm_agent_dspy_program_flag_is_for_dspy_react_only(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("rla2", Path(__file__).resolve().parents[1] / "scripts" / "run_llm_agent.py")
    rla = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rla)
    f = tmp_path / "p.json"
    f.write_text("{}")
    with pytest.raises(SystemExit):
        rla.main(["--system", "react", "--alerts", "synthetic", "--n", "1", "--dspy-program", str(f), "--run-dir", str(tmp_path / "o")])
