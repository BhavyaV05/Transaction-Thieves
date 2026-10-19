"""Mock LLM for offline development: scripted answers, record/replay cassettes, and an HTTP server.

Owner: T3 (Akshat).  Spec: tracks.md 1.2 rule 2 and hand-off #10 ("a mock for offline development"),
tracks.md 8 ("mocks first").

In-process (unit tests, CI):
    llm = MockLLM(script=['{"action": "stop", ...}'])          # answers in order
    llm = MockLLM(cassette="outputs/cassettes/react_val20.jsonl")  # replay a recorded run
    rec = RecordingClient(HttpChatClient.from_env(), "outputs/cassettes/x.jsonl")  # record a real run

As a server, so code that speaks HTTP (AG2, DSPy, the harness) needs no GPU:
    python -m ftm.llm.mock --port 8001 [--cassette FILE] [--api-key KEY]
    FTM_LLM_ENDPOINT=http://127.0.0.1:8001/v1 FTM_LLM_MODEL=mock ...

When nothing is scripted or recorded for a request, the mock answers with the smallest JSON that satisfies the
request's json_schema (so structured callers always parse), or with the text "MOCK".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ftm.llm.budget import estimate_messages, estimate_tokens
from ftm.llm.client import ChatClient, ChatResponse
from ftm.llm.jsonout import example_instance

Responder = Callable[[list[dict], dict], str]


def request_key(messages: list[dict], **kw: Any) -> str:
    """Stable key of a request. `extra` (server-specific knobs) is left out on purpose: a cassette is recorded
    under one serving config, and the same calls must replay in-process and over HTTP."""
    body = {
        "messages": [{"role": m.get("role"), "content": m.get("content")} for m in messages],
        "max_tokens": kw.get("max_tokens") or 512,  # HttpChatClient.chat defaults, so omitted == default
        "temperature": kw.get("temperature") or 0.0,
        "json_schema": kw.get("json_schema"),
        "stop": kw.get("stop"),
        "seed": kw.get("seed"),
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:24]


def load_cassette(path: str | Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["key"]] = rec["response"]
    return out


class MockLLM:
    """In-process ChatClient. Order: script, then cassette, then responder, then the schema-shaped default.

    strict=True raises on a request that is neither scripted nor recorded (use it to prove a replay is complete).
    """

    def __init__(
        self,
        script: list[str | Responder] | None = None,
        cassette: str | Path | None = None,
        responder: Responder | None = None,
        strict: bool = False,
        model: str = "mock",
        latency_s: float = 0.0,
    ):
        self.script = list(script or [])
        self.recorded = load_cassette(cassette) if cassette else {}
        self.responder = responder
        self.strict = strict
        self.model = model
        self.latency_s = latency_s
        self.requests: list[dict] = []
        self._lock = threading.Lock()

    def _answer(self, messages: list[dict], kw: dict) -> tuple[str, dict | None]:
        with self._lock:
            if self.script:
                item = self.script.pop(0)
                return (item(messages, kw) if callable(item) else item), None
        rec = self.recorded.get(request_key(messages, **kw))
        if rec is not None:
            return rec["text"], rec
        if self.strict:
            raise KeyError(f"MockLLM(strict): no scripted or recorded answer for {request_key(messages, **kw)}")
        if self.responder is not None:
            return self.responder(messages, kw), None
        if kw.get("json_schema"):
            return json.dumps(example_instance(kw["json_schema"])), None
        return "MOCK", None

    def chat(self, messages: list[dict], **kw: Any) -> ChatResponse:
        self.requests.append({"messages": messages, **kw})
        text, rec = self._answer(messages, kw)
        if self.latency_s:
            time.sleep(self.latency_s)
        if rec is not None:  # replay the recorded costs too, so token accounting matches the real run
            return ChatResponse(text=text, tokens_in=rec["tokens_in"], tokens_out=rec["tokens_out"],
                                latency_s=rec["latency_s"], model=rec.get("model", self.model),
                                reasoning=rec.get("reasoning"))  # fmt: skip
        return ChatResponse(text=text, tokens_in=estimate_messages(messages), tokens_out=estimate_tokens(text),
                            latency_s=self.latency_s, model=self.model, finish_reason="stop")  # fmt: skip


class RecordingClient:
    """Wraps a real client and appends every exchange to a JSONL cassette that MockLLM can replay."""

    def __init__(self, client: ChatClient, path: str | Path):
        self.client = client
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def chat(self, messages: list[dict], **kw: Any) -> ChatResponse:
        resp = self.client.chat(messages, **kw)
        rec = {
            "key": request_key(messages, **kw),
            "response": {"text": resp.text, "tokens_in": resp.tokens_in, "tokens_out": resp.tokens_out,
                         "latency_s": resp.latency_s, "model": resp.model, "reasoning": resp.reasoning},
        }  # fmt: skip
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return resp


# -- HTTP server (OpenAI-compatible subset: /v1/models, /v1/chat/completions, /version) -----------------------
_KNOWN = {"model", "messages", "max_tokens", "temperature", "response_format", "stop", "seed"}


def _handler(mock: MockLLM, api_key: str | None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep test output quiet
            pass

        def _send(self, code: int, obj: dict):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorised(self) -> bool:
            if api_key and self.headers.get("Authorization") != f"Bearer {api_key}":
                self._send(401, {"error": "invalid API key"})
                return False
            return True

        def do_GET(self):
            if self.path == "/version":
                return self._send(200, {"version": "mock"})
            if not self._authorised():
                return
            if self.path.rstrip("/") == "/v1/models":
                return self._send(200, {"object": "list", "data": [{"id": mock.model, "object": "model"}]})
            self._send(404, {"error": f"no route {self.path}"})

        def do_POST(self):
            if not self._authorised():
                return
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._send(404, {"error": f"no route {self.path}"})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            rf = body.get("response_format") or {}
            schema = (rf.get("json_schema") or {}).get("schema") if rf.get("type") == "json_schema" else None
            try:
                r = mock.chat(
                    body.get("messages", []), max_tokens=body.get("max_tokens"), temperature=body.get("temperature"),
                    json_schema=schema, stop=body.get("stop"), seed=body.get("seed"),
                    extra={k: v for k, v in body.items() if k not in _KNOWN},
                )  # fmt: skip
            except KeyError as e:
                return self._send(400, {"error": str(e)})
            self._send(200, {
                "id": "mock-1", "object": "chat.completion", "model": r.model,
                "choices": [{"index": 0, "finish_reason": r.finish_reason or "stop",
                             "message": {"role": "assistant", "content": r.text, "reasoning_content": r.reasoning}}],
                "usage": {"prompt_tokens": r.tokens_in, "completion_tokens": r.tokens_out,
                          "total_tokens": r.tokens_in + r.tokens_out},
            })  # fmt: skip

    return Handler


def start_server(mock: MockLLM, host: str = "127.0.0.1", port: int = 0, api_key: str | None = None):
    """Start in a background thread. Returns (server, base_url); call server.shutdown() to stop."""
    server = ThreadingHTTPServer((host, port), _handler(mock, api_key))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://{host}:{server.server_address[1]}/v1"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="OpenAI-compatible mock LLM server (offline development)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--cassette", default=None, help="JSONL recorded with RecordingClient")
    ap.add_argument("--strict", action="store_true", help="400 on requests that are not in the cassette")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--model", default="mock")
    a = ap.parse_args(argv)
    mock = MockLLM(cassette=a.cassette, strict=a.strict, model=a.model)
    server = ThreadingHTTPServer((a.host, a.port), _handler(mock, a.api_key))
    print(f"mock LLM on http://{a.host}:{a.port}/v1 (model {a.model!r}, {len(mock.recorded)} recorded answers)")
    server.serve_forever()


if __name__ == "__main__":
    main()
