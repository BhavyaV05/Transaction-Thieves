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
