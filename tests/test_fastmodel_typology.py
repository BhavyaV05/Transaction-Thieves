"""T3: the typology prompt defines its labels (zero-shot 8B answered 'A' = fan_in for every row without them)."""

import numpy as np

from ftm.fastmodel.decisions import DECISIONS, DecisionExample, messages, to_sft_example
from ftm.fastmodel.model import label_logps
from ftm.planners.llm_prompt import TYPOLOGIES


def test_typology_prompt_defines_every_label_and_offers_none():
    user = messages(DecisionExample("typology", "Alert: x"))[-1]["content"]
    for t in TYPOLOGIES:
        assert f"{t} =" in user  # each pattern has a one-line definition
    assert "Choose none when the text shows no clear pattern" in user and "I = none" in user
    assert user.index("Patterns:") < user.index("Answer with exactly one letter")


def test_binary_decisions_have_no_hint_and_prompt_is_unchanged():
    for name in ("worth_asking", "rule_applies", "injection_present", "stop"):
        assert DECISIONS[name].hint == ""
        user = messages(DecisionExample(name, "ctx"))[-1]["content"]
        assert user == f"ctx\n\nQuestion: {DECISIONS[name].question}\nAnswer with exactly one word: yes or no."


def test_label_tokens_are_distinct_single_letters_and_read_back():
    d = DECISIONS["typology"]
    assert len(set(d.tokens)) == len(d.labels) == 9 and all(len(t) == 1 for t in d.tokens)
    top = [{"token": " I", "logprob": -0.1}, {"token": "A", "logprob": -3.0}, {"token": "the", "logprob": -2.0}]
    lp = label_logps(top, "typology")
    assert int(np.argmax(lp)) == d.labels.index("none") and lp[0] == -3.0


def test_sft_example_uses_the_same_prompt():
    ex = DecisionExample("typology", "Alert: x", label="cycle")
    sft = to_sft_example(ex)
    assert sft["messages"][:-1] == messages(ex) and sft["messages"][-1]["content"] == "C"
