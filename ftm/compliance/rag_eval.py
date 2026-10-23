"""Evaluation of the regulatory RAG (tracks.md 6.3: "citation accuracy is measured on 100 labelled cases";
plan.md 6.7 ablation: whole corpus vs retrieved rules). Owner: T4 (Shreyas).

Two measurements:
  retrieval     every expert question spec (configs/questions/expert_specs.yaml) cites the red flags it comes from;
                those are its relevant documents. recall@k and hit@k of a retriever given the question text.
  citation      100 labelled cases (question, partner jurisdiction) -> the legal-basis rule that permits it. A
                system's cited rule_id is correct if it exists, applies to that jurisdiction and permits ask_bank
                for the question's answer kind (the gate's own test); exact match with the expected id is also
                reported.
"""

from __future__ import annotations

import random

from ftm.compliance import expert_specs as es


def retrieval_eval(search, k: int = 5, questions=None) -> dict:
    """`search(query, k) -> list of hits with rule_id`. Relevant = the spec's red flags."""
    qs = questions if questions is not None else es.load_questions(askable_only=True)
    recalls, hits = [], []
    for q in qs:
        got = {h["rule_id"] for h in search(f"{q.family.replace('_', ' ')}: {q.text}", k)}
        rel = set(q.red_flags)
        recalls.append(len(got & rel) / len(rel))
        hits.append(bool(got & rel))
    n = len(qs)
    return {"k": k, "n": n, "recall_at_k": sum(recalls) / n, "hit_at_k": sum(hits) / n}


def citation_cases(n: int = 100, seed: int = 0, gate=None) -> list[dict]:
    """(question, jurisdiction) -> expected legal-basis rule, seeded. Every askable expert question in US and SG."""
    from ftm.compliance.gate import Gate

    gate = gate or Gate.from_config()
    rng = random.Random(seed)
    pool = []
    for q in es.load_questions(askable_only=True):
        for jur in ("US", "SG"):
            r = gate._rule_for("ask_bank", jur, q.kind, None)
            pool.append({"case_id": f"{q.qid}-{jur}", "question": q.text, "kind": q.kind, "jurisdiction": jur,
                         "expected_rule_id": r.rule_id if r else None})  # fmt: skip
    rng.shuffle(pool)
    return pool[:n]


def citation_accuracy(cited: dict, cases: list[dict], gate=None) -> dict:
    """`cited`: case_id -> rule_id the system cited (missing = no citation)."""
    from ftm.compliance.gate import Gate

    gate = gate or Gate.from_config()
    valid = exact = missing = 0
    for c in cases:
        rid = cited.get(c["case_id"])
        if rid is None:
            missing += 1
            continue
        r = gate._rule_for("ask_bank", c["jurisdiction"], c["kind"], rid)
        valid += r is not None
        exact += rid == c["expected_rule_id"]
    n = len(cases)
    return {"n": n, "valid": valid / n, "exact": exact / n, "missing": missing / n}
