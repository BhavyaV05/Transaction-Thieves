"""Every system registered in scripts/run_llm_agent.py builds with the arguments the script passes it.

Regression guard: a registry argument added for one system (e.g. `pool` for T2's Group C ports) must not break the
others (it broke the T4 Sharma/Torres entries once).
"""

import importlib.util

import pytest

from ftm.llm.mock import MockLLM

spec = importlib.util.spec_from_file_location("run_llm_agent", "scripts/run_llm_agent.py")
rla = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rla)


@pytest.mark.parametrize("name", sorted(rla.GROUP_C))
def test_group_c_entries_accept_the_script_kwargs(name):
    system = rla.GROUP_C[name](MockLLM(), seed=0, extra=None, fusion=None, pool=None)
    assert hasattr(system, "investigate")


@pytest.mark.parametrize("name", sorted(rla.GROUP_D))
def test_group_d_entries_build(name):
    if name == "dspy_react":
        pytest.importorskip("dspy")
    system = rla.GROUP_D[name](MockLLM(), seed=0, extra=None, fusion=None)
    assert hasattr(system, "investigate")
