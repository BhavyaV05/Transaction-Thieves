"""T4: regulatory RAG (BM25 + optional dense), retrieval and citation evaluation."""

import pytest

from ftm.compliance import rag, rag_eval
from ftm.compliance.gate import Gate


def test_tokenize_stems_and_drops_stopwords():
    assert rag.tokenize("The funds were passed on to accounts") == ["fund", "pass", "account"]


def test_bm25_ranks_the_matching_document_first():
    docs = [rag.Doc("A", "red_flag", "US", "funds moved back to the first bank", "s"),
            rag.Doc("B", "red_flag", "US", "round dollar amounts", "s"),
            rag.Doc("C", "rule", "SG", "request to another participant", "s")]  # fmt: skip
    r = rag.Retriever(docs)
    assert r.search("money moved back to first bank")[0]["rule_id"] == "A"
    assert [h["rule_id"] for h in r.search("request", jurisdiction="US")] == []
    assert len(r.whole_corpus("SG")) == 1


def test_dense_hook_is_mixed_in():
    docs = [rag.Doc("A", "red_flag", "US", "alpha", "s"), rag.Doc("B", "red_flag", "US", "beta", "s")]
    emb = {"alpha": [1.0, 0.0], "beta": [0.0, 1.0], "query": [0.0, 1.0]}
    r = rag.Retriever(docs, embedder=lambda ts: [emb.get(t, [0.0, 1.0]) for t in ts], alpha=1.0)
    assert r.search("query")[0]["rule_id"] == "B"  # no word overlap: the dense score decides


def test_gate_uses_bm25_and_keeps_keyword_ablation():
    g = Gate.from_config()
    assert g.retrieval == "bm25"
    hits = g.lookup_rules("money moved from one bank to another and then back to the first bank", k=3)
    assert "RF-US-14" in [h["rule_id"] for h in hits]
    assert set(hits[0]) == {"rule_id", "kind", "jurisdiction", "text", "source"}
    assert Gate.from_config(retrieval="keyword").lookup_rules("cash", k=2)


def test_retrieval_eval_bm25_not_worse_than_keyword_at_10():
    gates = {m: Gate.from_config(retrieval=m) for m in ("bm25", "keyword")}
    res = {m: rag_eval.retrieval_eval(lambda q, k, g=g: g.lookup_rules(q, k=k), k=10) for m, g in gates.items()}
    assert res["bm25"]["n"] >= 50 and 0 < res["bm25"]["recall_at_k"] <= 1
    assert res["bm25"]["recall_at_k"] >= res["keyword"]["recall_at_k"]


def test_citation_cases_and_accuracy():
    cases = rag_eval.citation_cases()
    assert len(cases) == 100 and {c["jurisdiction"] for c in cases} == {"US", "SG"}
    assert {c["expected_rule_id"] for c in cases} == {"US-314B-SHARE", "SG-COSMIC-REQUEST"}
    perfect = {c["case_id"]: c["expected_rule_id"] for c in cases}
    assert rag_eval.citation_accuracy(perfect, cases) == {"n": 100, "valid": 1.0, "exact": 1.0, "missing": 0.0}
    wrong = {c["case_id"]: "US-314B-SHARE" for c in cases}  # right for US, wrong jurisdiction for SG
    acc = rag_eval.citation_accuracy(wrong, cases)
    n_us = sum(c["jurisdiction"] == "US" for c in cases)
    assert acc["valid"] == pytest.approx(n_us / 100) and acc["missing"] == 0
    assert rag_eval.citation_accuracy({}, cases)["missing"] == 1.0
