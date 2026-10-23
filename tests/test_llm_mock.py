"""T3 week 2: mock LLM (in-process + HTTP), record/replay, budgets, JSON output checks."""

import json

import pytest

from ftm.llm.budget import BudgetedClient, BudgetExceeded, PromptTooLong
from ftm.llm.client import ChatResponse, HttpChatClient, LLMError
from ftm.llm.jsonout import (
    JsonOutputError,
    example_instance,
    extract_json,
    parse_and_validate,
    schema_errors,
)
from ftm.llm.mock import MockLLM, RecordingClient, request_key, start_server
from ftm.types import Limits

MSG = [{"role": "user", "content": "hello"}]
SCHEMA = {
    "title": "t",
    "type": "object",
    "properties": {
        "action": {"enum": ["select", "stop"]},
        "eps": {"enum": [0.0, 0.5]},
        "bank": {"anyOf": [{"enum": [70, 220]}, {"type": "null"}]},
        "belief": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["action", "eps", "bank", "belief"],
    "additionalProperties": False,
}


def test_extract_json_handles_think_fences_and_prose():
    assert extract_json('<think>{"no": 1}</think>{"a": 1}') == {"a": 1}
    assert extract_json('Sure:\n```json\n{"a": "x}"}\n```') == {"a": "x}"}
    assert extract_json('blah {"a": {"b": "q\\"}"}} tail') == {"a": {"b": 'q"}'}}
    with pytest.raises(JsonOutputError):
        extract_json("no json here")


def test_schema_subset_validator():
    good = {"action": "stop", "eps": 0.0, "bank": None, "belief": 0.2}
    assert schema_errors(good, SCHEMA) == []
    assert any("missing" in e for e in schema_errors({"action": "stop"}, SCHEMA))
    assert any("unexpected" in e for e in schema_errors({**good, "x": 1}, SCHEMA))
    assert schema_errors({**good, "bank": 5}, SCHEMA)
    assert schema_errors({**good, "belief": 1.5}, SCHEMA)
    assert schema_errors({**good, "belief": True}, SCHEMA)  # bool is not a number
    assert schema_errors(example_instance(SCHEMA), SCHEMA) == []


def test_mock_order_script_then_default():
    llm = MockLLM(script=['{"action": "select"}', lambda m, kw: "from callable"])
    assert llm.chat(MSG).text == '{"action": "select"}'
    assert llm.chat(MSG).text == "from callable"
    out = llm.chat(MSG, json_schema=SCHEMA)
    assert parse_and_validate(out.text, SCHEMA)["action"] == "select"
    assert llm.chat(MSG).text == "MOCK"
    assert len(llm.requests) == 4


def test_record_then_strict_replay(tmp_path):
    cas = tmp_path / "c.jsonl"

    class Real:
        def chat(self, messages, **kw):
            return ChatResponse(text=f"real:{messages[0]['content']}", tokens_in=11, tokens_out=3, latency_s=0.7,
                                model="qwen3-8b")  # fmt: skip

    rec = RecordingClient(Real(), cas)
    rec.chat(MSG, max_tokens=64)
    rec.chat([{"role": "user", "content": "bye"}], json_schema=SCHEMA)
    replay = MockLLM(cassette=cas, strict=True)
    r = replay.chat(MSG, max_tokens=64)
    assert (r.text, r.tokens_in, r.tokens_out, r.latency_s, r.model) == ("real:hello", 11, 3, 0.7, "qwen3-8b")
    assert replay.chat([{"role": "user", "content": "bye"}], json_schema=SCHEMA).text == "real:bye"
    with pytest.raises(KeyError):
        replay.chat(MSG, max_tokens=65)


def test_request_key_defaults_match_explicit():
    assert request_key(MSG) == request_key(MSG, max_tokens=512, temperature=0.0)
    assert request_key(MSG) != request_key(MSG, seed=1)


def test_http_server_round_trip_with_auth_and_schema():
    mock = MockLLM(script=['{"action": "stop", "eps": 0.0, "bank": null, "belief": 0.1}'])
    server, url = start_server(mock, api_key="k1")
    try:
        client = HttpChatClient(url, api_key="k1", retries=0)
        assert client.model == "mock"
        r = client.chat(MSG, json_schema=SCHEMA, max_tokens=50)
        assert parse_and_validate(r.text, SCHEMA)["belief"] == 0.1
        r2 = client.chat(MSG, json_schema=SCHEMA)  # unscripted: schema-shaped default
        assert schema_errors(json.loads(r2.text), SCHEMA) == []
        assert r2.tokens_in > 0
        assert mock.requests[0]["json_schema"] == SCHEMA and mock.requests[0]["max_tokens"] == 50
        with pytest.raises(LLMError, match="401"):
            HttpChatClient(url, api_key="wrong", model="mock", retries=0).chat(MSG)
    finally:
        server.shutdown()


def test_http_replays_cassette_recorded_in_process(tmp_path):
    cas = tmp_path / "c.jsonl"

    class Real:
        def chat(self, messages, **kw):
            return ChatResponse(text="recorded", tokens_in=5, tokens_out=1, latency_s=0.1, model="qwen3-8b")

    RecordingClient(Real(), cas).chat(MSG, json_schema=SCHEMA, max_tokens=99)
    server, url = start_server(MockLLM(cassette=cas, strict=True))
    try:
        r = HttpChatClient(url, model="mock", retries=0).chat(MSG, json_schema=SCHEMA, max_tokens=99)
        assert (r.text, r.tokens_in) == ("recorded", 5)
    finally:
        server.shutdown()


def test_budget_caps_calls_and_prompt_and_counts_cost():
    limits = Limits(max_llm_calls=2, max_tokens=50)
    b = BudgetedClient(MockLLM(), limits, tag="a1")
    b.chat(MSG)
    b.chat(MSG)
    assert b.usage.llm_calls == 2 and b.calls_left == 0
    with pytest.raises(BudgetExceeded):
        b.chat(MSG)
    with pytest.raises(PromptTooLong):  # pre-send estimate
        BudgetedClient(MockLLM(), limits).chat([{"role": "user", "content": "x" * 1000}])

    class Liar:  # server counts more tokens than our estimate: charged, then refused
        def chat(self, messages, **kw):
            return ChatResponse(text="", tokens_in=51, tokens_out=0, latency_s=0.2, model="m")

    b2 = BudgetedClient(Liar(), limits)
    with pytest.raises(PromptTooLong):
        b2.chat(MSG)
    assert b2.usage.llm_calls == 1 and b2.usage.tokens_in == 51 and b2.usage.gpu_seconds == pytest.approx(0.2)
