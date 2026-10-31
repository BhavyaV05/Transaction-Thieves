

def test_server_schema_drops_grammar_unsupported_keys():
    from ftm.llm.client import server_schema

    schema = {"type": "object", "properties": {"agents": {"type": "array", "items": {"enum": ["a", "b"]}, "uniqueItems": True}},
              "anyOf": [{"uniqueItems": True, "type": "array"}]}  # fmt: skip
    out = server_schema(schema)
    assert "uniqueItems" not in out["properties"]["agents"] and "uniqueItems" not in out["anyOf"][0]
    assert out["properties"]["agents"]["items"] == {"enum": ["a", "b"]}
    assert schema["properties"]["agents"]["uniqueItems"] is True  # the caller's schema (used to validate) is untouched
