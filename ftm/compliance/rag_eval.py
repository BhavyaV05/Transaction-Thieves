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


def citation_by_retrieval(lookup, cases: list[dict], query_of, jurisdiction: str = "partner", k: int = 5, gate=None) -> dict:
    """No-LLM citation baseline: for each case, retrieve with `query_of(case)` and cite the top-ranked hit of kind "rule".
    `lookup(query, jurisdiction, k) -> hits`. `jurisdiction`: "partner" (the case's partner bank, an oracle), "none" (the query has
    to decide) or "US" (the home bank's, what the harness's lookup_rules passes today). Scored by `citation_accuracy`; also
    `rule_in_topk`: the share of cases whose top k holds any citable rule at all."""
    cited, has_rule = {}, 0
    for c in cases:
        jur = {"partner": c["jurisdiction"], "none": None}.get(jurisdiction, jurisdiction)
        hits = lookup(query_of(c), jur, k)
        rules = [h for h in hits if h["kind"] == "rule"]
        has_rule += bool(rules)
        cited[c["case_id"]] = rules[0]["rule_id"] if rules else None
    return {**citation_accuracy(cited, cases, gate), "rule_in_topk": has_rule / len(cases)}


def citation_whole_corpus(cases: list[dict], gate=None) -> dict:
    """Ablation 'whole corpus in the prompt', LLM-free floor: every rule is in the prompt and the citer picks one uniformly among
    the rules of the corpus (all jurisdictions). Expected valid / exact share, computed exactly. recall of every document is 1 by
    construction; the cost is the prompt (n_docs documents instead of k)."""
    from ftm.compliance.gate import Gate

    gate = gate or Gate.from_config()
    rules = list(gate.rules)
    valid = exact = 0.0
    for c in cases:
        ok = [r for r in rules if gate._rule_for("ask_bank", c["jurisdiction"], c["kind"], r) is not None]  # noqa: SLF001
        valid += len(ok) / len(rules)
        exact += (c["expected_rule_id"] in rules) / len(rules)
    n = len(cases)
    return {"n": n, "valid": valid / n, "exact": exact / n, "missing": 0.0, "n_docs_in_prompt": len(gate.rules) + len(gate.red_flags)}
