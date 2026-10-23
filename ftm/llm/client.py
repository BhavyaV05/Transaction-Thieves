"""Minimal OpenAI-compatible chat client over the standard library.

Owner: T3 (Akshat).  Spec: tracks.md 5.3 (shared endpoint, hand-off #10), docs/hardware.md section 5.

Talks to vLLM (`vllm serve`), llama.cpp's server, ollama's /v1 endpoint, or the mock server in
ftm/llm/mock.py. No `openai` package needed, so every machine (and CI) can import it.

Configuration comes from the environment, matching configs/llm/default.yaml:
  FTM_LLM_ENDPOINT  base URL ending in /v1 (default http://localhost:8000/v1)
  FTM_LLM_API_KEY   bearer token (never commit it; `scripts/serve_llm.py --gen-key` writes it to .env)
  FTM_LLM_MODEL     served model name (default: the first model the server lists)
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_ENDPOINT = "http://localhost:8000/v1"


class LLMError(RuntimeError):
    """The endpoint failed or returned something that is not a chat completion."""


@dataclass(frozen=True)
class ChatResponse:
    text: str
    tokens_in: int
    tokens_out: int
    latency_s: float
    model: str
    finish_reason: str | None = None
    reasoning: str | None = None  # thinking text, if the server splits it out (vLLM --reasoning-parser)
    raw: dict = field(default_factory=dict, repr=False, compare=False)


class ChatClient(Protocol):
    def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        json_schema: dict | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
        extra: dict | None = None,
    ) -> ChatResponse: ...


def parse_completion(body: dict, latency_s: float) -> ChatResponse:
    """Turn an OpenAI chat-completion JSON body into a ChatResponse."""
    try:
        choice = body["choices"][0]
        msg = choice["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"not a chat completion: {str(body)[:300]}") from e
    usage = body.get("usage") or {}
    return ChatResponse(
        text=msg.get("content") or "",
        tokens_in=int(usage.get("prompt_tokens") or 0),
        tokens_out=int(usage.get("completion_tokens") or 0),
        latency_s=latency_s,
        model=body.get("model", ""),
        finish_reason=choice.get("finish_reason"),
        reasoning=msg.get("reasoning_content") or msg.get("reasoning"),
        raw=body,
    )


class HttpChatClient:
    """Blocking chat client for one OpenAI-compatible endpoint.

    `default_extra` is merged into every request body (e.g. `{"chat_template_kwargs": {"enable_thinking": False}}`
    for Qwen3 on vLLM). Connection errors and 5xx answers are retried `retries` times with backoff; 4xx are not.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_ENDPOINT,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 180.0,
        retries: int = 2,
        default_extra: dict | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key not in (None, "", "none") else None
        self._model = model
        self.timeout_s = timeout_s
        self.retries = retries
        self.default_extra = dict(default_extra or {})

    @classmethod
    def from_env(cls, endpoint: str | None = None, model: str | None = None, **kwargs: Any) -> HttpChatClient:
        """Environment first; `endpoint` / `model` override it (E18 serves one candidate per endpoint)."""
        return cls(
            base_url=endpoint or os.environ.get("FTM_LLM_ENDPOINT", DEFAULT_ENDPOINT),
            api_key=os.environ.get("FTM_LLM_API_KEY"),
            model=model or os.environ.get("FTM_LLM_MODEL"),
            **kwargs,
        )

    # -- HTTP ---------------------------------------------------------------------------------------------
    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        data = None if payload is None else json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:500]
                if e.code < 500:
                    raise LLMError(f"{method} {url} -> HTTP {e.code}: {body}") from e
                last = LLMError(f"{method} {url} -> HTTP {e.code}: {body}")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = LLMError(f"{method} {url} failed: {e}")
            if attempt < self.retries:
                time.sleep(min(2.0**attempt, 8.0))
        raise last  # type: ignore[misc]

    def get(self, path: str) -> dict:
        return self._request("GET", path)

    @property
    def model(self) -> str:
        if self._model is None:
            models = self.get("/models").get("data") or []
            if not models:
                raise LLMError("the endpoint lists no models; set FTM_LLM_MODEL")
            self._model = models[0]["id"]
        return self._model

    def server_version(self) -> str | None:
        """vLLM exposes /version at the server root, not under /v1."""
        root = self.base_url[: -len("/v1")] if self.base_url.endswith("/v1") else self.base_url
        try:
            req = urllib.request.Request(f"{root}/version")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode()).get("version")
        except Exception:
            return None

    # -- chat ---------------------------------------------------------------------------------------------
    def build_body(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        json_schema: dict | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
        extra: dict | None = None,
    ) -> dict:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": json_schema.get("title", "output"), "schema": json_schema, "strict": True},
            }
        if stop:
            body["stop"] = stop
        if seed is not None:
            body["seed"] = seed
        body.update(self.default_extra)
        body.update(extra or {})
        return body

    def chat(self, messages: list[dict], **kwargs: Any) -> ChatResponse:
        body = self.build_body(messages, **kwargs)
        t0 = time.perf_counter()
        out = self._request("POST", "/chat/completions", body)
        return parse_completion(out, time.perf_counter() - t0)
