"""llm: client side of the shared LLM endpoint (OpenAI-compatible), used by every LLM system in the harness.

Owner: T3 (Akshat).  Component: Serving (hand-off #10).  Tier: Core support.
Spec: tracks.md 1.2 and 5.3; docs/hardware.md sections 4-5.

client.py: stdlib-only chat client (no `openai` dependency, so CPU-only machines and CI can import it).
"""

from ftm.llm.client import ChatClient, ChatResponse, HttpChatClient, LLMError

__all__ = ["ChatClient", "ChatResponse", "HttpChatClient", "LLMError"]
