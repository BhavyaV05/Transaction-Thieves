"""A deterministic stand-in for a vLLM logprobs endpoint, for smoke-testing S1BENCH plumbing without a GPU.  Owner: T3 (Akshat).

TEST-ONLY helper (lives in tests/, nothing under scripts/ or ftm/ may use it, and no result may be produced against it).  It is NOT a model: it reads the allowed answer tokens from the prompt ("Answer with exactly one letter: A = .., B = .." or "yes or no")
and returns first-token top_logprobs that favour one allowed token chosen by a hash of the prompt.  Numbers produced against it mean nothing.

    python tests/s1bench_fake_server.py --port 8099     # FTM_LLM_ENDPOINT=http://127.0.0.1:8099/v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def allowed_tokens(prompt: str) -> list[str]:
    m = re.search(r"Answer with exactly one letter: (.*?)\.(?:\n|$)", prompt, re.S)
    if m:
        return re.findall(r"(?:^|, )([A-Z]) = ", m.group(1))
    return ["yes", "no"] if "yes or no" in prompt else ["A", "B"]


def fake_top(prompt: str, k: int = 20) -> list[dict]:
    toks = allowed_tokens(prompt)[:k] or ["A"]
    h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
    w = [1.0 + ((h >> (3 * i)) % 7) for i in range(len(toks))]
    w[h % len(toks)] += 6.0
    z = sum(w)
    return [{"token": t, "logprob": math.log(x / z)} for t, x in zip(toks, w, strict=True)]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": "fake"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = req["messages"][-1]["content"]
        top = fake_top(prompt)
        out = {"model": "fake", "choices": [{"message": {"content": top[0]["token"]}, "finish_reason": "length",
                                              "logprobs": {"content": [{"token": top[0]["token"], "logprob": top[0]["logprob"], "top_logprobs": top}]}}],
               "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": 1}}  # fmt: skip
        body = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    a = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
