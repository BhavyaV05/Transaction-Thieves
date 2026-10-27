"""Per-investigation LLM budget: call cap, prompt-token cap and cost accounting (fairness protocol, plan.md 7.2).

Owner: T3 (Akshat).  Spec: tracks.md 1.2 rule 4 (fixed prompt cap) and rule 5 (tokens and calls are the cost).

Every LLM system (our planner and every Group D port) wraps its client in one `BudgetedClient` per alert, so
`Limits.max_llm_calls` and `Limits.max_tokens` bind the same way for all of them and the Verdict cost fields
come from one place. `gpu_seconds` is wall-clock LLM latency; it is a GPU-time number only on the reference
machine (tracks.md 1.2 rule 1).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field

from ftm.llm.client import ChatClient, ChatResponse
from ftm.llm.trim import trim_evidence
from ftm.types import Limits

CHARS_PER_TOKEN = 3.2  # conservative for Qwen3 / gpt-oss tokenizers on our English + alias text
ESTIMATE_SLACK = 1.15  # pre-send check only rejects prompts clearly over the cap; the server count is final
MAX_PROMPT_REFITS = 2  # evidence trims after a server-counted overflow, per call (same rule as the planner, ftm/planners/llm.py)
REFIT_MARGIN = 0.97  # aim a little under the cap: the trimmed prompt's count is not exactly proportional
MAX_SCALE = 1.5  # a server/estimate ratio beyond this is not a tokenizer difference (mock clients, broken counts): ignore it


class BudgetExceeded(RuntimeError):
    """No LLM calls left for this alert."""


class PromptTooLong(ValueError):
    """The prompt is over Limits.max_tokens. `server_tokens`: the server's own count, when the call was made."""

    def __init__(self, msg: str, server_tokens: int | None = None):
        super().__init__(msg)
        self.server_tokens = server_tokens


def _note(kind: str, tag: str, msg: str) -> None:
    """One stderr line per refit / refusal (scripts/val_runs_table.py counts them per alert from the run logs)."""
    print(f"{kind} {tag}: {msg}", file=sys.stderr)  # kind: PromptRefit | PromptTooLong (PromptTooLong[planner]: its own refit follows)


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

    def __init__(self, client: ChatClient, limits: Limits, tag: str = "", trim: bool = True):
        self.client = client
        self.limits = limits
        self.tag = tag
        self.trim = trim  # the planner trims itself (build_messages_with_estimate) and sets False
        self.usage = Usage()
        self.token_scale = 1.0  # server count / estimate, learned per alert from the server's own counts (Gemma: about 1.05)
        self.n_trims = 0

    @property
    def calls_left(self) -> int:
        return self.limits.max_llm_calls - self.usage.llm_calls

    def _fit(self, messages: list[dict], est: int, cap: int) -> list[dict] | None:
        """Trim the evidence until `est` * scale fits `cap`; None if the evidence cannot be cut any further."""
        msgs = messages
        while est * self.token_scale > cap:
            nxt = trim_evidence(msgs, max(0.125, 1 - cap / (est * self.token_scale)))
            if nxt is None:
                return None
            msgs, est = nxt, estimate_messages(nxt)
            self.n_trims += 1
        return msgs

    def chat(self, messages: list[dict], **kwargs) -> ChatResponse:
        if self.calls_left <= 0:
            raise BudgetExceeded(f"{self.tag}: max_llm_calls={self.limits.max_llm_calls} used")
        cap = self.limits.max_tokens
        est = estimate_messages(messages)
        if self.trim and est * self.token_scale > cap:  # known (or likely) overflow: trim before the call
            fitted = self._fit(messages, est, int(cap * REFIT_MARGIN))
            if fitted is not None:
                messages, est = fitted, estimate_messages(fitted)
        if est > cap * ESTIMATE_SLACK * (self.token_scale if self.trim else 1.0):
            _note("PromptTooLong" if self.trim else "PromptTooLong[planner]", self.tag, f"estimate ~{est} > cap {cap}")
            raise PromptTooLong(f"{self.tag}: prompt ~{est} tokens > cap {cap}")
        refits = 0
        while True:
            resp = self.client.chat(messages, **kwargs)
            u = self.usage
            u.llm_calls += 1
            u.tokens_in += resp.tokens_in
            u.tokens_out += resp.tokens_out
            u.gpu_seconds += resp.latency_s
            u.calls.append({"tokens_in": resp.tokens_in, "tokens_out": resp.tokens_out, "latency_s": resp.latency_s})
            if est > 0 and resp.tokens_in > 0 and resp.tokens_in / est <= MAX_SCALE:
                self.token_scale = max(self.token_scale, resp.tokens_in / est)
            if resp.tokens_in <= cap:
                return resp
            # counted (the call happened) but refused, so no system profits from an over-long prompt
            over = PromptTooLong(f"{self.tag}: server counted {resp.tokens_in} prompt tokens > cap {cap}", server_tokens=resp.tokens_in)
            if not self.trim or refits >= MAX_PROMPT_REFITS or self.calls_left <= 0:
                _note("PromptTooLong" if self.trim else "PromptTooLong[planner]", self.tag, f"server counted {resp.tokens_in} > cap {cap}")
                raise over
            fitted = self._fit(messages, est, int(cap * REFIT_MARGIN))  # token_scale is now the measured ratio
            if fitted is None or fitted is messages:  # nothing left to cut, or the estimate says it already fits (a miscount)
                _note("PromptTooLong", self.tag, f"server counted {resp.tokens_in} > cap {cap}, no evidence left to trim")
                raise over
            _note("PromptRefit", self.tag, f"server counted {resp.tokens_in} > cap {cap}; evidence trimmed, resent (scale {self.token_scale:.3f})")
            messages, est, refits = fitted, estimate_messages(fitted), refits + 1
