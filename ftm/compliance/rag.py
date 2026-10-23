"""Regulatory RAG (C6, secondary): retrieval over the legal-basis rules and the red-flag corpus.

Owner: T4 (Shreyas).  Spec: plan.md 6.7, tracks.md 6.3.

BM25 (Robertson-Sparck Jones idf, k1 = 1.5, b = 0.75) in plain Python: the corpus is ~50 short documents, so CI needs
no extra package. A dense retriever plugs in as `embedder(texts) -> list[vector]` (e.g. bge-small on the serving
machine, plan.md 6.7) and is mixed in as hybrid = (1 - alpha) * bm25 / max_bm25 + alpha * cosine. Without one the
retriever is BM25 only. `whole_corpus()` is the "whole corpus in the prompt" ablation.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

STOP = frozenset("""a an and are as at be been by for from has have in into is it its of on or that the their this to
was were which with within any all each other than then there these those such not no only also may can""".split())
_TOKEN = re.compile(r"[a-z0-9]+")


def _stem(w: str) -> str:
    for suf in ("ings", "ing", "ies", "ied", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf in ("ies", "ied") else "")
    return w


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower().replace("_", " ")) if t not in STOP and len(t) > 1]


@dataclass(frozen=True)
class Doc:
    doc_id: str
    kind: str  # rule | red_flag
    jurisdiction: str  # US | SG | ANY
    text: str
    source: str
    extra: str = ""  # searchable but not shown: typologies, signal

    def searchable(self) -> str:
        return f"{self.doc_id} {self.text} {self.source} {self.extra}"


class BM25:
    def __init__(self, docs: Sequence[Doc], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = list(docs), k1, b
        self.tf = [Counter(tokenize(d.searchable())) for d in self.docs]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter(w for t in self.tf for w in t)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, dl in zip(self.tf, self.len, strict=True):
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            out.append(s)
        return out


class Retriever:
    def __init__(self, docs: Sequence[Doc], embedder: Callable[[list[str]], list] | None = None, alpha: float = 0.5):
        self.docs = list(docs)
        self.bm25 = BM25(self.docs)
        self.embedder, self.alpha = embedder, alpha
        self._emb = embedder([d.text for d in self.docs]) if embedder else None

    def search(self, query: str, jurisdiction: str | None = None, k: int = 5, kinds: Sequence[str] | None = None) -> list[dict]:
        bm = self.bm25.scores(query)
        top = max(bm) if bm and max(bm) > 0 else 1.0
        dense = [0.0] * len(self.docs)
        if self._emb is not None:
            qv = self.embedder([query])[0]
            dense = [_cos(qv, v) for v in self._emb]
        hits = []
        for d, s, c in zip(self.docs, bm, dense, strict=True):
            if jurisdiction and d.jurisdiction not in (jurisdiction, "ANY"):
                continue
            if kinds and d.kind not in kinds:
                continue
            score = (1 - self.alpha) * s / top + self.alpha * c if self._emb is not None else s
            if score > 0:
                hits.append((score, d))
        hits.sort(key=lambda t: (-t[0], t[1].doc_id))
        return [{"rule_id": d.doc_id, "kind": d.kind, "jurisdiction": d.jurisdiction, "text": d.text, "source": d.source,
                 "score": round(s, 4)} for s, d in hits[:k]]  # fmt: skip

    def whole_corpus(self, jurisdiction: str | None = None) -> list[dict]:
        """Ablation: every document (for the prompt), no retrieval."""
        return [{"rule_id": d.doc_id, "kind": d.kind, "jurisdiction": d.jurisdiction, "text": d.text, "source": d.source}
                for d in self.docs if not jurisdiction or d.jurisdiction in (jurisdiction, "ANY")]  # fmt: skip


def _cos(a, b) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb) if na and nb else 0.0


def docs_from_gate(gate) -> list[Doc]:
    docs = [Doc(r.rule_id, "rule", r.jurisdiction, r.text, r.source) for r in gate.rules.values()]
    docs += [Doc(f.flag_id, "red_flag", f.jurisdiction, f.text, f.source, " ".join(f.typologies) + " " + f.signal)
             for f in gate.red_flags.values()]  # fmt: skip
    return docs
