import pytest


@pytest.mark.skip(reason="T2, week 2: static check that no code path in ftm/banks returns raw rows")
def test_no_code_path_returns_raw_rows():
    ...


@pytest.mark.skip(reason="T1, week 1: every tool that reads transactions applies the as_of time guard")
def test_every_tool_applies_time_guard():
    ...


@pytest.mark.skip(reason="T2, week 4: budgets and the gate cannot be bypassed through the harness")
def test_harness_cannot_bypass_gate_or_accountant():
    ...
