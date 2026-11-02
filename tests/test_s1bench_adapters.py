"""Adapters: the model-native input of one shared typed question, and the mapping of native outputs back to the question's option order."""

import json

import pytest

from ftm.fastmodel.s1bench_adapters import (
    TEV_SYSTEM,
    TypedQuestion,
    bekko_input,
    gliner_input,
    jevk5_question,
    laya_questions,
    normalise,
    tev_messages,
)

YN = TypedQuestion({"text": "ctx"}, "Is it so?", (("yes", "It is so."), ("no", "It is not so.")))
CH = TypedQuestion({"text": "ctx"}, "Which?", (("fan", "many to one"), ("hub", "scatter"), ("none", "nothing")))


def test_yes_no_becomes_noul_with_false_true_definitions():
    assert YN.kind == "noul" and CH.kind == "choice"
    assert [i for i, _ in YN.ordered()] == ["false", "true"] and dict(YN.ordered())["true"] == "It is so."
    assert YN.to_option_order({"true": 0.8, "false": 0.2}) == [0.8, 0.2]  # options order is (yes, no)
    assert CH.to_option_order({"fan": 0.1, "hub": 0.2, "none": 0.7}) == [0.1, 0.2, 0.7]


def test_bekko_input_follows_the_card_format():
    b = bekko_input(YN)
    assert json.loads(b["state_json"]) == {"text": "ctx"}
    d = b["decisions"][0]
    assert d["type"] == "noul" and json.loads(d["instructions_json"]) == "Is it so?" and d["documents"] == [] and d["scoring"] is None
    assert [c["id"] for c in d["criteria"]] == ["false", "true"] and all(c["value"] is None for c in d["criteria"])
    assert [c["id"] for c in bekko_input(CH)["decisions"][0]["criteria"]] == ["fan", "hub", "none"]


def test_gliner_input_uses_anonymous_choice_keys_and_noul_keys():
    text, labels = gliner_input(CH)
    assert text.startswith("Question: Which?") and "ctx" in text and list(labels) == ["option_0", "option_1", "option_2"] and labels["option_1"] == "scatter"
    assert list(gliner_input(YN)[1]) == ["false", "true"]


def test_jevk5_and_laya_questions():
    q = jevk5_question(CH)
    assert q == {"type": "choice", "instructions": "Which?", "criteria": {"option_0": "many to one", "option_1": "scatter", "option_2": "nothing"}}
    assert jevk5_question(YN)["type"] == "noul" and list(jevk5_question(YN)["criteria"]) == ["false", "true"]
    assert laya_questions(CH) == {"decision": q}


def test_tev_messages_follow_the_card():
    m = tev_messages(CH)
    assert m[0] == {"role": "system", "content": TEV_SYSTEM} and m[1]["role"] == "user"
    body = json.loads(m[1]["content"])
    assert body["state"] == {"text": "ctx"} and body["question"] == "Which?"
    assert [o["label"] for o in body["options"]] == ["A", "B", "C"] and body["options"][0] == {"label": "A", "key": "option_0", "description": "many to one"}
    yn = json.loads(tev_messages(YN)[1]["content"])
    assert [(o["label"], o["key"]) for o in yn["options"]] == [("A", "false"), ("B", "true")]


def test_normalise_handles_degenerate_vectors():
    assert normalise([2, 2]) == [0.5, 0.5] and normalise([0, 0, 0]) == pytest.approx([1 / 3] * 3) and normalise([-1, 1]) == [0.0, 1.0]
