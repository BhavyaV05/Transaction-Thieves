"""Structured-output helpers: pull JSON out of model text, check it against a JSON-schema subset, make examples.

Owner: T3 (Akshat).  Spec: tracks.md 5.4 (planner outputs are schema-validated JSON; >=95% must parse).

vLLM's `response_format=json_schema` constrains decoding on the server, but every output is still checked
here: other servers (llama.cpp, ollama, the mock) may not constrain, and the parse rate is a reported metric.
Only the schema keywords our own schemas use are supported (type, properties, required,
additionalProperties, enum, const, minimum, maximum, minLength, maxLength, items, maxItems, anyOf).
"""

from __future__ import annotations

import json
import re
from typing import Any

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
    "null": type(None),
}


class JsonOutputError(ValueError):
    """The text holds no JSON object, or the object breaks the schema."""


def extract_json(text: str) -> Any:
    """First JSON object in `text`, ignoring <think> blocks and code fences."""
    text = _THINK.sub("", text or "").strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise JsonOutputError(f"no JSON object in model output: {text[:200]!r}")


def _is_type(value: Any, t: str) -> bool:
    if t in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, _TYPES[t])


def schema_errors(value: Any, schema: dict, path: str = "$") -> list[str]:
    """All violations of `schema` by `value` (empty list = valid)."""
    if "anyOf" in schema:
        options = [schema_errors(value, s, path) for s in schema["anyOf"]]
        return [] if any(not o for o in options) else [f"{path}: matches no anyOf option ({options[0][0]})"]
    errs: list[str] = []
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_is_type(value, x) for x in types):
            return [f"{path}: expected {t}, got {type(value).__name__}"]
    if "const" in schema and value != schema["const"]:
        errs.append(f"{path}: must be {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} not in {schema['enum'][:12]}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errs.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append(f"{path}: longer than {schema['maxLength']}")
    if isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        if "items" in schema:
            for i, v in enumerate(value):
                errs += schema_errors(v, schema["items"], f"{path}[{i}]")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{path}: missing {k!r}")
        if schema.get("additionalProperties") is False:
            errs += [f"{path}: unexpected key {k!r}" for k in value if k not in props]
        for k, v in value.items():
            if k in props:
                errs += schema_errors(v, props[k], f"{path}.{k}")
    return errs


def parse_and_validate(text: str, schema: dict) -> dict:
    obj = extract_json(text)
    errs = schema_errors(obj, schema)
    if errs:
        raise JsonOutputError("; ".join(errs[:5]))
    return obj


def example_instance(schema: dict) -> Any:
    """A minimal value that satisfies `schema` (used by the mock server when no answer is scripted)."""
    if "anyOf" in schema:
        return example_instance(schema["anyOf"][0])
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    t = schema.get("type", "object")
    t = t[0] if isinstance(t, list) else t
    if t == "object":
        props = schema.get("properties", {})
        return {k: example_instance(props.get(k, {})) for k in schema.get("required", list(props))}
    if t == "array":
        return []
    if t == "string":
        return "x" * schema.get("minLength", 0)
    if t in ("number", "integer"):
        lo = schema.get("minimum", 0)
        hi = schema.get("maximum", lo)
        return lo if lo == hi or lo > 0 or hi < 0 else max(lo, min(0, hi))
    if t == "boolean":
        return False
    return None
