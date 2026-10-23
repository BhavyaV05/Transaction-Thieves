"""Fast-model interface and the two LLM implementations compared in E10 (C13).

Owner: T3 (Akshat).  Spec: plan.md 6.14, tracks.md 5.6.

  LogprobFastModel   our fine-tuned Qwen3-4B (or any served model): one forward pass, max_tokens=1, read the top
                     logprobs of the first token, keep the label tokens, renormalise, apply temperature T.
  JsonDecisionModel  the baseline: the 8B writes {"label": ..., "confidence": ...}; the stated confidence is the
                     probability of that label, the rest is spread evenly over the other labels.
Both return label -> probability. Latency is recorded per call (timing counts only on the reference machine).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ftm.fastmodel.calibration import softmax_t
from ftm.fastmodel.decisions import DECISIONS, DecisionExample, messages
from ftm.llm.client import ChatClient
from ftm.llm.jsonout import JsonOutputError, parse_and_validate

FLOOR_LOGP = math.log(1e-6)  # a label token missing from the top logprobs


class FastModel(Protocol):
    def predict_proba(self, ex: DecisionExample) -> dict[str, float]: ...


def label_logps(top_logprobs: list[dict], decision: str) -> np.ndarray:
    """Label log-probs (in DECISIONS order) from one token's top logprobs [{token, logprob}, ...]."""
    d = DECISIONS[decision]
    t2l = {t.lower(): i for i, t in enumerate(d.tokens)}
    out = np.full(len(d.labels), FLOOR_LOGP)
    for item in top_logprobs:
        i = t2l.get(item["token"].strip().lower())
        if i is not None:
            out[i] = max(out[i], item["logprob"])  # " yes" and "yes" both count; keep the larger
    return out


@dataclass
class LogprobFastModel:
    client: ChatClient
    temperature: float = 1.0  # fitted by ftm.fastmodel.calibration.fit_temperature on validation
    top_logprobs: int = 20
    extra: dict = field(default_factory=lambda: {"chat_template_kwargs": {"enable_thinking": False}})
    latencies: list[float] = field(default_factory=list)

    def raw_logps(self, ex: DecisionExample) -> np.ndarray:
        r = self.client.chat(messages(ex), max_tokens=1, temperature=0.0,
                             extra={**self.extra, "logprobs": True, "top_logprobs": self.top_logprobs})  # fmt: skip
        self.latencies.append(r.latency_s)
        try:
            top = r.raw["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError):
            top = []  # server without logprobs: every label at the floor -> uniform
        return label_logps(top, ex.decision)

    def predict_proba(self, ex: DecisionExample) -> dict[str, float]:
        p = softmax_t(self.raw_logps(ex)[None, :], self.temperature)[0]
        return dict(zip(DECISIONS[ex.decision].labels, p.tolist(), strict=True))


@dataclass
class JsonDecisionModel:
    client: ChatClient
    extra: dict = field(default_factory=lambda: {"chat_template_kwargs": {"enable_thinking": False}})
    latencies: list[float] = field(default_factory=list)

    def predict_proba(self, ex: DecisionExample) -> dict[str, float]:
        d = DECISIONS[ex.decision]
        schema = {"title": "decision", "type": "object", "additionalProperties": False, "required": ["label", "confidence"],
                  "properties": {"label": {"enum": list(d.labels)}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}}}  # fmt: skip
        msgs = messages(ex)
        msgs[-1] = {**msgs[-1], "content": msgs[-1]["content"].rsplit("\n", 1)[0]
                    + f"\nReply with JSON: label (one of {list(d.labels)}) and confidence (0-1)."}  # fmt: skip
        r = self.client.chat(msgs, max_tokens=40, temperature=0.0, json_schema=schema, extra=self.extra)
        self.latencies.append(r.latency_s)
        try:
            obj = parse_and_validate(r.text, schema)
        except JsonOutputError:
            return {lbl: 1 / len(d.labels) for lbl in d.labels}
        conf = min(max(float(obj["confidence"]), 1e-6), 1 - 1e-6)
        rest = (1 - conf) / (len(d.labels) - 1)
        return {lbl: conf if lbl == obj["label"] else rest for lbl in d.labels}


def to_json_line(ex: DecisionExample, probs: dict[str, float]) -> str:
    return json.dumps({"alert_id": ex.alert_id, "decision": ex.decision, "label": ex.label, "probs": probs})
