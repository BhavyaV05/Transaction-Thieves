import importlib
import pkgutil

import pytest

import ftm
from ftm.eval.runlog import EVENTS, RUN_LOG_COLUMNS
from ftm.harness.interface import Investigator, ScriptedToolBox, ToolBox
from ftm.types import AlertContext, Decline, Limits, Reply, Verdict

ALERT = AlertContext(alert_id="a1", home_bank="b1", as_of="2022-09-01T00:00", split="val", p0=0.4)


def test_scripted_toolbox_satisfies_protocol():
    assert isinstance(ScriptedToolBox(), ToolBox)


def test_scripted_toolbox_returns_in_order_and_records_calls():
    reply = Reply(bank="b2", kind="bool", value=True, eps_spent=1.0, latency_days=0.5)
    tools = ScriptedToolBox([reply, Decline(bank="b3")])
    assert tools.ask_bank("b2", "q1", 1.0) is reply
    assert isinstance(tools.ask_bank("b3", "q2", 1.0), Decline)
    assert [c[1] for c in tools.calls] == ["q1", "q2"]
    with pytest.raises(RuntimeError):
        tools.ask_bank("b4", "q3", 1.0)


def test_dummy_investigator_satisfies_protocol():
    class Dummy:
        def investigate(self, alert, tools, limits):
            return Verdict(alert_id=alert.alert_id, probability=alert.p0)

    assert isinstance(Dummy(), Investigator)
    assert Dummy().investigate(ALERT, ScriptedToolBox(), Limits()).probability == 0.4


def test_default_limits_match_plan():
    assert Limits().eps_per_alert == 2.0


def test_runlog_schema_has_core_columns():
    for col in ("run_id", "system", "seed", "alert_id", "event", "eps", "probability", "machine"):
        assert col in RUN_LOG_COLUMNS
    assert EVENTS == ("step", "verdict")


def _all_modules():
    for m in pkgutil.walk_packages(ftm.__path__, "ftm."):
        yield m.name


@pytest.mark.parametrize("name", sorted(_all_modules()))
def test_every_module_imports_and_names_an_owner(name):
    mod = importlib.import_module(name)
    if name in {"ftm.types", "ftm.guards", "ftm.harness.interface", "ftm.eval.runlog"}:
        return  # real code, not stubs
    doc = mod.__doc__ or ""
    assert "Owner: T" in doc or name.startswith("ftm.baselines.") or name == "ftm.planners.llm", name
