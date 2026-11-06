"""R2: the ported agent graphs reproduce today's loops exactly (same verdict, probability, question log, trace, LLM requests)."""

import pytest

pytest.importorskip("langgraph")

from graph_agents_support import same, text_responder  # noqa: E402

from ftm.external import lats as old_lats  # noqa: E402
from ftm.external import llmcompiler as old_llmc  # noqa: E402
from ftm.external import react as old_react  # noqa: E402
from ftm.external import reflexion as old_reflexion  # noqa: E402
from ftm.external import rewoo as old_rewoo  # noqa: E402
from ftm.graph.agents import lats, llmcompiler, react, reflexion, rewoo  # noqa: E402


def test_react_parity():
    assert same(old_react.React, react.React, text_responder) > 5


def test_reflexion_parity():
    assert same(old_reflexion.Reflexion, reflexion.Reflexion, text_responder) > 5


def test_rewoo_parity():
    assert same(old_rewoo.Rewoo, rewoo.Rewoo, text_responder) > 5


def test_lats_parity():
    assert same(old_lats.Lats, lats.Lats, text_responder) > 5


def test_llmcompiler_parity():
    assert same(old_llmc.LlmCompiler, llmcompiler.LlmCompiler, text_responder) > 5


from graph_agents_support import json_responder  # noqa: E402

from ftm.external import bed_llm as old_bed  # noqa: E402
from ftm.external import debate_autogen as old_debate  # noqa: E402
from ftm.external import mediq as old_mediq  # noqa: E402
from ftm.external import uot as old_uot  # noqa: E402
from ftm.graph.agents import bed_llm, debate_autogen, mediq, uot  # noqa: E402


def test_debate_parity():
    assert same(old_debate.Debate, debate_autogen.Debate, json_responder) > 5


def test_uot_parity():
    assert same(old_uot.Uot, uot.Uot, json_responder, seeds=range(60), extra_attrs=("last_belief", "last_stop")) > 5


def test_bed_llm_parity():
    assert same(old_bed.BedLlm, bed_llm.BedLlm, json_responder, seeds=range(60), extra_attrs=("last_hypotheses", "last_stop")) > 3


def test_mediq_parity():
    assert same(old_mediq.MediQ, mediq.MediQ, json_responder, extra_attrs=("last_stop",)) > 5


def test_dspy_react_parity():
    pytest.importorskip("dspy")
    from graph_agents_support import dspy_responder

    from ftm.external import dspy_react as old_dspy
    from ftm.graph.agents import dspy_react

    assert same(old_dspy.DspyReact, dspy_react.DspyReact, dspy_responder, seeds=range(15)) > 3
