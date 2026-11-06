"""Regulatory RAG evaluation (T4). CPU only, seconds (hybrid: bge-small on CPU, ~1 min the first time).

    python scripts/rag_eval.py                        # retrieval recall@k and citation accuracy: bm25 / hybrid / keyword / whole corpus
    python scripts/rag_eval.py --no-hybrid            # skip the dense retriever (no sentence-transformers)
    python scripts/rag_eval.py --cases-out data/runs/citation_cases.jsonl   # also write the 100 citation cases

Writes data/runs/rag_eval.json. Two measurements (ftm.compliance.rag_eval):
  retrieval  recall@k of the red flags each askable expert question cites (pure ranking, `rules_first` off)
  citation   the 100 labelled cases: the top-ranked citable rule of the retrieval, scored valid (passes the gate's own check) and
             exact (the expected id), for the fixed planner query, the question alone, the fixed query plus the question, and that plus the
             partner's jurisdiction words (what ftm.compliance.query.rules_query adds), each with the partner's
             jurisdiction, no jurisdiction, or the home bank's (US), with and without the citable rules pinned to the head.
             Whole corpus: the LLM-free floor (a uniform pick among all rules); an LLM-in-the-loop citation run needs the server.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.compliance import rag_eval  # noqa: E402
from ftm.compliance.gate import Gate  # noqa: E402
from ftm.compliance.query import JURISDICTION_WORDS  # noqa: E402

FIXED_QUERY = "sharing information with another financial institution about suspected money laundering"  # RULES_QUERY of the planner (v0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/runs/rag_eval.json")
    ap.add_argument("--cases-out", default=None)
    ap.add_argument("--no-hybrid", action="store_true")
    a = ap.parse_args()
    modes = ["bm25", "keyword"] + ([] if a.no_hybrid else ["hybrid"])
    cases = rag_eval.citation_cases()
    res: dict = {"retrieval": {}, "citation": {}}
    for mode in modes:
        g = Gate.from_config(retrieval=mode)
        res["retrieval"][mode] = [rag_eval.retrieval_eval(lambda q, k, g=g: g.lookup_rules(q, k=k, rules_first=False), k=k) for k in (3, 5, 10)]
        cit = {}
        for qname, qf in (("fixed_query", lambda c: FIXED_QUERY), ("question_only", lambda c: c["question"]),
                          ("case_query", lambda c: f"{FIXED_QUERY} {c['question']}"),
                          ("aware_query", lambda c: f"{FIXED_QUERY} {JURISDICTION_WORDS[c['jurisdiction']]} {c['question']}")):
            for jur in ("partner", "none", "US"):
                for pin in (False, True):
                    cit[f"{qname}/jur={jur}/rules_first={pin}"] = rag_eval.citation_by_retrieval(
                        lambda q, j, k, g=g, pin=pin: g.lookup_rules(q, j, k, rules_first=pin), cases, qf, jur, gate=g)
        res["citation"][mode] = cit
    res["citation"]["whole_corpus"] = rag_eval.citation_whole_corpus(cases)
    res["citation_cases"] = {"n": len(cases), "expected_rules": sorted({c["expected_rule_id"] for c in cases})}
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    if a.cases_out:
        Path(a.cases_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.cases_out).write_text("".join(json.dumps(c) + "\n" for c in cases))
    for mode in modes:
        print(mode, " ".join(f"recall@{r['k']}={r['recall_at_k']:.3f}" for r in res["retrieval"][mode]))
        for name, r in res["citation"][mode].items():
            print(f"  {name:48s} valid={r['valid']:.2f} exact={r['exact']:.2f} missing={r['missing']:.2f} rule_in_top5={r['rule_in_topk']:.2f}")
    w = res["citation"]["whole_corpus"]
    print(f"whole_corpus (uniform pick, {w['n_docs_in_prompt']} docs in the prompt) valid={w['valid']:.2f} exact={w['exact']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
