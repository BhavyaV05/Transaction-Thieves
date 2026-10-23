"""Per-investigation LLM budget: call cap, prompt-token cap and cost accounting (fairness protocol, plan.md 7.2).

Owner: T3 (Akshat).  Spec: tracks.md 1.2 rule 4 (fixed prompt cap) and rule 5 (tokens and calls are the cost).

Every LLM system (our planner and every Group D port) wraps its client in one `BudgetedClient` per alert, so
`Limits.max_llm_calls` and `Limits.max_tokens` bind the same way for all of them and the Verdict cost fields
come from one place. `gpu_seconds` is wall-clock LLM latency; it is a GPU-time number only on the reference
machine (tracks.md 1.2 rule 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ftm.llm.client import ChatClient, ChatResponse
from ftm.types import Limits

CHARS_PER_TOKEN = 3.2  # conservative for Qwen3 / gpt-oss tokenizers on our English + alias text
ESTIMATE_SLACK = 1.15  # pre-send check only rejects prompts clearly over the cap; the server count is final


class BudgetExceeded(RuntimeError):
    """No LLM calls left for this alert."""


class PromptTooLong(ValueError):
    """The prompt is over Limits.max_tokens."""


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def estimate_messages(messages: list[dict]) -> int:
    return sum(estimate_tokens(m.get("content") or "") + 4 for m in messages)


@dataclass
class Usage:
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    gpu_seconds: float = 0.0
    calls: list[dict] = field(default_factory=list)  # per-call record for the run log


class BudgetedClient:
    """Wraps a ChatClient for ONE investigation. Raises instead of exceeding the shared limits."""

    def __init__(self, client: ChatClient, limits: Limits, tag: str = ""):
        self.client = client
        self.limits = limits
        self.tag = tag
        self.usage = Usage()

    @property
    def calls_left(self) -> int:
        return self.limits.max_llm_calls - self.usage.llm_calls

    def chat(self, messages: list[dict], **kwargs) -> ChatResponse:
        if self.calls_left <= 0:
            raise BudgetExceeded(f"{self.tag}: max_llm_calls={self.limits.max_llm_calls} used")
        est = estimate_messages(messages)
        if est > self.limits.max_tokens * ESTIMATE_SLACK:
            raise PromptTooLong(f"{self.tag}: prompt ~{est} tokens > cap {self.limits.max_tokens}")
        resp = self.client.chat(messages, **kwargs)
        u = self.usage
        u.llm_calls += 1
        u.tokens_in += resp.tokens_in
        u.tokens_out += resp.tokens_out
        u.gpu_seconds += resp.latency_s
        u.calls.append({"tokens_in": resp.tokens_in, "tokens_out": resp.tokens_out, "latency_s": resp.latency_s})
        if resp.tokens_in > self.limits.max_tokens:
            # counted (the call happened) but refused, so no system profits from an over-long prompt
            raise PromptTooLong(f"{self.tag}: server counted {resp.tokens_in} prompt tokens > cap {self.limits.max_tokens}")
        return resp
