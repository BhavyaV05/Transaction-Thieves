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


def test_citable_rules_head_the_list_and_never_fall_out_of_the_top_k():
    g = Gate.from_config()
    for jur, rid in (("US", "US-314B-SHARE"), ("SG", "SG-COSMIC-REQUEST")):
        hits = g.lookup_rules("round dollar amounts structuring cash", jurisdiction=jur, k=3)
        assert hits[0]["rule_id"] == rid and hits[0]["kind"] == "rule" and len(hits) == 3
    both = g.lookup_rules("cash", k=5)
    assert {"US-314B-SHARE", "SG-COSMIC-REQUEST"} <= {h["rule_id"] for h in both[:2]}
    # the old order stays available for the ablation
    assert Gate.from_config().lookup_rules("round dollar amounts structuring cash", jurisdiction="US", k=3, rules_first=False)[0]["kind"] == "red_flag"


def test_hybrid_gate_uses_the_embedder_and_keeps_the_pinned_rules():
    calls = []

    def stub(texts):
        calls.append(len(texts))
        return [[1.0 if "cash" in t else 0.0, 1.0 if "layer" in t else 0.0, 0.1] for t in texts]

    g = Gate.from_config(retrieval="hybrid", embedder=stub, retrieval_alpha=0.7)
    hits = g.lookup_rules("layering through intermediaries", jurisdiction="US", k=4)
    assert calls and hits[0]["rule_id"] == "US-314B-SHARE" and len(hits) == 4
    assert any(h["kind"] == "red_flag" for h in hits)


def test_bge_embedder_is_pinned_and_loads_when_installed():
    from ftm.compliance import dense

    assert len(dense.BGE_REVISION) == 40
    pytest.importorskip("sentence_transformers")
    try:
        emb = dense.bge_embedder()
    except Exception as e:  # noqa: BLE001 - weights not in the cache of this machine
        pytest.skip(f"bge-small not available: {e}")
    v = emb(["money moved back to the first bank", "round dollar amounts"])
    assert len(v) == 2 and len(v[0]) == 384 and abs(sum(x * x for x in v[0]) - 1) < 1e-3


def test_rules_query_depends_on_the_alert():
    from ftm.compliance.query import BASE, rules_query
    from ftm.types import AlertContext

    def alert(**s):
        return AlertContext("A", "70", "2022-09-05T12:00:00", "val", 0.2, {"partner_banks": [220, 1588], **s})

    g = Gate.from_config(jurisdiction_by_bank={"1588": "SG"})
    q1 = rules_query(alert(amount=950.0, format="ACH", currency="US Dollar"), g)
    q2 = rules_query(alert(amount=250_000.0, format="Wire", currency="Euro", typology="fan-in"), g)
    assert q1 != q2 and q1.startswith(BASE) and "Singapore" in q1 and "cross-border" in q1
    assert "fan-in" not in q1 and "many senders" in q2 and "ACH" in q1 and "Euro" in q2
    assert "A0000" not in q1 and rules_query(alert()) .startswith(BASE)


def test_citation_by_retrieval_and_whole_corpus_floor():
    g = Gate.from_config()
    cases = rag_eval.citation_cases()
    lk = lambda q, j, k: g.lookup_rules(q, j, k)  # noqa: E731
    oracle = rag_eval.citation_by_retrieval(lk, cases, lambda c: "sharing information", "partner", gate=g)
    assert oracle["exact"] == 1.0 and oracle["rule_in_topk"] == 1.0
    blind = rag_eval.citation_by_retrieval(lk, cases, lambda c: "sharing information", "none", gate=g)
    assert 0.4 < blind["valid"] < 0.7  # without the jurisdiction the query cannot tell US from SG
    aware = rag_eval.citation_by_retrieval(lk, cases, lambda c: "sharing information " + ("Singapore COSMIC" if c["jurisdiction"] == "SG" else "United States 314(b)"), "none", gate=g)
    assert aware["exact"] > blind["exact"]
    floor = rag_eval.citation_whole_corpus(cases, g)
    assert floor["valid"] == pytest.approx(0.25) and floor["n_docs_in_prompt"] > 40
