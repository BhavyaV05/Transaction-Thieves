"""Shared prompt refit in BudgetedClient (Gemma counts ~5% more than the Qwen-fitted estimate): the Group C/D ports trim the
evidence block and resend on a server-counted PromptTooLong, like the planner. Mock LLM only."""

import math
from dataclasses import replace

import pytest

from ftm.external.react import React
from ftm.external.text_agent import case_text
from ftm.harness.interface import ScriptedToolBox
from ftm.llm.budget import BudgetedClient, PromptTooLong, estimate_messages
from ftm.llm.mock import MockLLM
from ftm.llm.trim import OMITTED, trim_evidence
from ftm.types import AlertContext, Limits

ROWS = "\n".join(f"2022-09-0{i % 9} 10:{i % 60:02d} | A{i:07x} -> A01bc22d | {i},000.00 US Dollar | ACH" for i in range(120))
ALERT = AlertContext("V000001", "70", "2022-09-05T12:00:00", "val", 0.2,
                     {"sender": "A0000002", "receiver": "A0000001", "amount": 9500.0, "evidence": ROWS, "partner_banks": [220, 1588]})


class Dense(MockLLM):
    """A tokenizer that counts `ratio` times the estimate."""

    def __init__(self, ratio, **kw):
        super().__init__(**kw)
        self.ratio = ratio

    def chat(self, messages, **kw):
        return replace(super().chat(messages, **kw), tokens_in=math.ceil(self.ratio * estimate_messages(messages)))


def msgs():
    return [{"role": "system", "content": "sys"}, {"role": "user", "content": case_text(ALERT, ROWS) + "\n\nThought 1:"}]


def test_trim_evidence_drops_oldest_rows_and_marks_the_cut():
    out = trim_evidence(msgs(), 0.5)
    text = out[1]["content"]
    assert OMITTED in text and ROWS.splitlines()[-1] in text and ROWS.splitlines()[0] not in text
    assert text.endswith("Thought 1:") and len(text) < len(msgs()[1]["content"])
    assert trim_evidence([{"role": "user", "content": "no evidence block here"}], 0.5) is None


def test_overflow_by_the_servers_count_is_trimmed_and_resent_then_learned():
    limits = Limits(max_tokens=3000, max_llm_calls=10)
    b = BudgetedClient(Dense(1.25), limits, tag="a")
    assert estimate_messages(msgs()) * 1.25 > 3000 and estimate_messages(msgs()) < 3000 * 1.15  # passes the pre-send check
    r = b.chat(msgs())
    assert r.tokens_in <= 3000 and b.usage.llm_calls == 2 and b.n_trims >= 1  # one over-long call, one resend
    b.chat(msgs())  # the measured ratio is kept: trimmed before sending, no wasted call
    assert b.usage.llm_calls == 3


def test_without_an_evidence_block_the_overflow_still_raises_and_is_counted():
    b = BudgetedClient(Dense(1.25), Limits(max_tokens=3000), tag="a")
    with pytest.raises(PromptTooLong) as e:
        b.chat([{"role": "user", "content": "x" * 8800}])
    assert e.value.server_tokens and b.usage.llm_calls == 1


def test_trim_false_keeps_the_old_behaviour():
    b = BudgetedClient(Dense(1.25), Limits(max_tokens=3000), tag="a", trim=False)
    with pytest.raises(PromptTooLong):
        b.chat(msgs())


def test_react_on_a_dense_tokenizer_finishes_instead_of_ending_at_p0():
    llm = Dense(1.25, script=["Thought 1: nothing to ask.\nAction 1: finish[0.4]"] * 2)  # the over-long call uses up one reply
    v = React(llm).investigate(ALERT, ScriptedToolBox([]), Limits(max_tokens=4000))
    assert v.probability == pytest.approx(0.4) and v.llm_calls == 2  # first call over the cap by the server's count, resent trimmed
