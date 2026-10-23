"""Harness v1 is frozen (week 4). This file pins the contract; changing it needs all four owners (tracks.md section 8).

Pinned: the ToolBox and Investigator signatures, the shared types (ftm/types.py) with their defaults, the harness
version, the run-log schema, and the exceptions a refusal raises. Extensions go in as new optional protocols or
keyword arguments with defaults.
"""

import dataclasses
import inspect

from ftm import types as T
from ftm.compliance.gate import BudgetRefused, GateBlocked
from ftm.eval.runlog import EVENTS, RUN_LOG_COLUMNS, RUN_LOG_VERSION, V0_COLUMNS
from ftm.harness import interface as I
from ftm.harness.toolbox import HarnessToolBox, ToolUnavailable


def fields(cls):
    return [(f.name, str(f.type)) for f in dataclasses.fields(cls)]


def test_version():
    assert I.HARNESS_VERSION == "1.0"


def test_toolbox_and_investigator_signatures_are_frozen():
    sig = lambda f: str(inspect.signature(f))  # noqa: E731
    assert sig(I.ToolBox.local_graph) == "(self, query: 'str', as_of: 'str') -> 'list[dict]'"
    assert sig(I.ToolBox.ask_bank) == "(self, bank: 'str', program: 'DslProgram', eps: 'float') -> 'BankResponse'"
    assert sig(I.ToolBox.lookup_rules) == "(self, query: 'str') -> 'list[dict]'"
    assert sig(I.ToolBox.search_library) == "(self, shape: 'dict') -> 'list[dict]'"
    assert sig(I.Investigator.investigate) == "(self, alert: 'AlertContext', tools: 'ToolBox', limits: 'Limits') -> 'Verdict'"
    assert {n for n in vars(I.ToolBox) if not n.startswith("_")} >= {"local_graph", "ask_bank", "lookup_rules", "search_library"}


def test_the_real_toolbox_extends_but_does_not_change_the_contract():
    # extra optional keyword (rule_id) and session methods are allowed; the pinned positional parameters are the same
    params = list(inspect.signature(HarnessToolBox.ask_bank).parameters)
    assert params[:4] == ["self", "bank", "program", "eps"] and params[4:] == ["rule_id"]
    assert inspect.signature(HarnessToolBox.ask_bank).parameters["rule_id"].default is None
    for name in ("local_graph", "lookup_rules", "search_library"):
        assert list(inspect.signature(getattr(HarnessToolBox, name)).parameters) == list(
            inspect.signature(getattr(I.ToolBox, name)).parameters
        )


def test_shared_types_are_frozen():
    assert fields(T.AlertContext) == [("alert_id", "str"), ("home_bank", "str"), ("as_of", "str"),
                                      ("split", "Literal['train', 'val', 'test']"), ("p0", "float"),
                                      ("summary", "dict")]  # fmt: skip
    assert [f[0] for f in fields(T.Reply)] == ["bank", "kind", "value", "eps_spent", "latency_days", "note"]
    assert [f[0] for f in fields(T.Decline)] == ["bank", "reason"]
    assert [f[0] for f in fields(T.Timeout)] == ["bank", "after_days"]
    assert [f[0] for f in fields(T.Limits)] == ["eps_per_alert", "max_llm_calls", "max_tokens", "max_sim_time_days"]
    assert dataclasses.asdict(T.Limits()) == {"eps_per_alert": 2.0, "max_llm_calls": 20, "max_tokens": 6000,
                                              "max_sim_time_days": 5.0}  # fmt: skip
    assert [f[0] for f in fields(T.QuestionLogEntry)] == ["step", "bank", "program", "eps", "response", "gate_decision",
                                                          "posterior_after"]  # fmt: skip
    assert [f[0] for f in fields(T.Verdict)] == ["alert_id", "probability", "questions_log", "tokens_in", "tokens_out",
                                                 "llm_calls", "gpu_seconds", "trials"]  # fmt: skip
    # `trials` (Reflexion trials per alert, default None) was added by T3 after the freeze: additive and optional, accepted by T2
    assert T.Verdict("a", None).trials is None  # added after v1 as an optional field (Reflexion trials)
    assert T.AnswerKind.__args__ == ("bool", "count", "kary", "set_count")
    assert T.Reply.__dataclass_params__.frozen and T.Limits.__dataclass_params__.frozen


def test_run_log_schema_is_frozen():
    assert EVENTS == ("step", "verdict")
    assert RUN_LOG_VERSION == 1
    # v0 columns stay first and unchanged; run-log v1 (T4) only appends
    assert list(V0_COLUMNS) == [
        "run_id", "commit", "machine", "system", "variant", "seed", "split", "alert_id", "event", "step", "bank",
        "program_canonical", "eps", "gate", "response_kind", "answer_value", "latency_days", "posterior", "probability",
        "label", "tokens_in", "tokens_out", "llm_calls", "gpu_seconds",
    ]  # fmt: skip
    assert list(RUN_LOG_COLUMNS)[len(V0_COLUMNS):] == [
        "log_version", "model", "thinking", "mode", "home_bank", "val_part", "p0", "stop_reason", "eps_spent",
        "sim_days", "trials", "family", "kind", "rule_id", "gate_reason", "log_lr", "note_present", "injection_id",
    ]  # fmt: skip


def test_refusals_keep_the_classes_every_consumer_depends_on():
    assert issubclass(GateBlocked, PermissionError)
    assert "budget" in BudgetRefused.__name__.lower()
    assert issubclass(ToolUnavailable, RuntimeError)
