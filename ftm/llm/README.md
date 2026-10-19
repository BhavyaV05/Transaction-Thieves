# ftm/llm: the shared LLM endpoint, client side (hand-off #10)

Owner: T3 (Akshat). Spec: `tracks.md` §1.2 rules 2–5 and §5.3. How to *start* a server is in `docs/hardware.md` §5. This page is for everyone who *calls* one.

## Connect

Every LLM system reads three environment variables:

| Variable | Shared server (8B on Akshat's PC) | Local mock (no GPU) |
|---|---|---|
| `FTM_LLM_ENDPOINT` | `http://<serving-pc-tailnet-name>:8000/v1` | `http://127.0.0.1:8001/v1` |
| `FTM_LLM_API_KEY` | sent to you privately; **never commit it** | unset (or what you passed to `--api-key`) |
| `FTM_LLM_MODEL` | `qwen3-8b` (also `qwen3-4b`, `gpt-oss-20b` when those are served) | `mock` |

```python
from ftm.llm import HttpChatClient
llm = HttpChatClient.from_env(default_extra={"chat_template_kwargs": {"enable_thinking": False}})
r = llm.chat([{"role": "user", "content": "hi"}], max_tokens=32)
r.text, r.tokens_in, r.tokens_out, r.latency_s
```

The client uses only the standard library (no `openai` package). AG2 and DSPy can point their own OpenAI clients at the same URL and key.

## Budgets (same for every system)

Wrap the client in one `BudgetedClient` per alert. It enforces `Limits.max_llm_calls` and the prompt cap `Limits.max_tokens`, and it accumulates the cost fields that go into the `Verdict`:

```python
from ftm.llm.budget import BudgetedClient, BudgetExceeded, PromptTooLong
b = BudgetedClient(llm, limits, tag=alert.alert_id)
...                                   # b.chat(...) raises instead of going over a limit
verdict.tokens_in, verdict.tokens_out = b.usage.tokens_in, b.usage.tokens_out
verdict.llm_calls, verdict.gpu_seconds = b.usage.llm_calls, b.usage.gpu_seconds
```

- The prompt check before sending uses a conservative estimate: 3.2 characters per token, plus 15% slack. The server's own count is final. A prompt the server counts as over the cap is charged to the budget and still refused, so no system gains from a long prompt.
- `gpu_seconds` is LLM wall-clock latency. It counts as GPU time only on the reference machine (`tracks.md` §1.2 rule 1).

## Structured output

Pass `json_schema=` to `chat()`. vLLM then constrains decoding (`response_format: json_schema`, strict). Always check the result anyway, because the parse rate is a reported metric and other servers may not constrain:

```python
from ftm.llm.jsonout import parse_and_validate, JsonOutputError
obj = parse_and_validate(r.text, schema)   # strips <think> blocks and code fences, then checks the schema subset
```

## Mock LLM (offline development, CI)

| Use | How |
|---|---|
| Unit tests: answers in order | `MockLLM(script=['{"action": "stop", ...}', callable, ...])` |
| Anything structured, no script | With no scripted answer, the mock returns the smallest JSON that satisfies the request's `json_schema` |
| Record a real run | `RecordingClient(HttpChatClient.from_env(), "outputs/cassettes/<run>.jsonl")` |
| Replay it without a GPU | `MockLLM(cassette=".../<run>.jsonl", strict=True)`. Recorded token counts and latencies replay too, so cost accounting matches |
| HTTP, for code that makes its own requests | `python -m ftm.llm.mock --port 8001 [--cassette FILE] [--strict] [--api-key KEY]` |

A cassette entry is keyed by messages, `max_tokens`, `temperature`, `json_schema`, `stop` and `seed`. Server-specific `extra` knobs are not part of the key, so record and replay under the same serving config. Cassettes live in `outputs/`, which is gitignored. They contain aliased prompts only, but share them through the agreed artifact folder, not git.
