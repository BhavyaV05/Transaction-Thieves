"""T2: Co-Investigator AI and FinFRE-RAG reimplementations on the mock LLM (native and ported modes)."""

import json
import re

import numpy as np
import pytest

from ftm.compliance.gate import Gate
from ftm.external import co_investigator as ci
from ftm.external import finfre_rag as ff
from ftm.external.case_memory import CaseMemory
from ftm.harness.interface import ScriptedToolBox
from ftm.llm.mock import MockLLM
from ftm.types import AlertContext, Limits, Reply


class RuleTools(ScriptedToolBox):
    def __init__(self, responses=None):
        super().__init__(responses)
        self.gate = Gate.from_config()

    def lookup_rules(self, query):
        return self.gate.lookup_rules(query)


def alert(aid="V0001", p0=0.46, amount=9850.0, **summary):
    ev = "\n".join(f"2022-09-0{d} 10:00 | A000000{d} -> A0000002 | {1000 * d:,.2f} US Dollar | Wire | bank 7 -> bank 1588"
                   for d in range(1, 5))  # fmt: skip
    s = {"sender": "A0000001", "receiver": "A0000002", "sender_bank": 7, "receiver_bank": 1588, "amount": amount,
         "currency": "US Dollar", "format": "Wire", "time": "2022-09-05 10:00", "evidence": ev, "partner_banks": [1588],
         "shape": {"in_deg": "4-7", "out_deg": "1", "n_banks": "2", "amount": "1k-10k varied", "span": "1-3d"}}  # fmt: skip
    s.update(summary)
    return AlertContext(aid, "7", "2022-09-05T10:00:00", "val", p0, s)


def titles(llm):
    return [(r.get("json_schema") or {}).get("title", "text") for r in llm.requests]


# --- Co-Investigator AI -----------------------------------------------------------------------------------------------
def narrative(conf=0.7, text="Four payers within days [E5] [E6]", support=("E5",)):
    return json.dumps({"reasoning": "step by step", "narrative": text, "confidence": conf,
                       "supporting_evidence": list(support), "contradicting_or_missing": []})  # fmt: skip


JUDGE_OK = json.dumps({"coherent": True, "issues": []})


def agent(flag=True, strength=0.7):
    return json.dumps({"flag": flag, "strength": strength, "finding": "fast pass-through [E5]", "evidence": ["E5"]})


def test_native_pipeline_runs_in_the_papers_order_and_asks_no_bank():
    llm = MockLLM(responder=ci.dev_responder)
    tools = RuleTools()
    c = ci.CoInvestigator(llm)
    v = c.investigate(alert(), tools, Limits())
    assert titles(llm) == ["coinv_plan", "coinv_agent", "coinv_agent", "coinv_narrative", "coinv_judge"]
    assert tools.calls == [] and v.questions_log == [] and v.llm_calls == 5
    assert v.probability == pytest.approx(0.5) and c.last_trace["plan"] == ["transaction_pattern", "volume_velocity"]


def test_the_narrative_confidence_is_the_probability_after_validation():
    llm = MockLLM(script=[json.dumps({"agents": ["account_health"], "reason": "x"}), agent(), narrative(0.83), JUDGE_OK])
    v = ci.CoInvestigator(llm).investigate(alert(), RuleTools(), Limits())
    assert v.probability == pytest.approx(0.83)


def test_a_rejected_draft_is_revised_once_then_accepted_or_dropped():
    plan = json.dumps({"agents": [], "reason": "none"})
    bad = narrative(0.9, "A huge 777,777 transfer [E5]")  # amount not in the evidence: rule-based check fails
    llm = MockLLM(script=[plan, bad, JUDGE_OK, narrative(0.6), JUDGE_OK])
    c = ci.CoInvestigator(llm)
    v = c.investigate(alert(), RuleTools(), Limits())
    assert v.probability == pytest.approx(0.6) and c.last_trace["problems"] == []
    assert titles(llm).count("coinv_narrative") == 2
    # still bad after the one revision: no usable probability (scored as p0), problems logged
    llm2 = MockLLM(script=[plan, bad, JUDGE_OK, bad, JUDGE_OK])
    c2 = ci.CoInvestigator(llm2)
    v2 = c2.investigate(alert(), RuleTools(), Limits())
    assert v2.probability is None and any("777,777" in p for p in c2.last_trace["problems"])


def test_the_llm_judge_can_reject_a_clean_draft():
    plan = json.dumps({"agents": [], "reason": "none"})
    judge_no = json.dumps({"coherent": False, "issues": ["contradicts the typology agents"]})
    llm = MockLLM(script=[plan, narrative(0.9), judge_no, narrative(0.4), JUDGE_OK])
    c = ci.CoInvestigator(llm)
    v = c.investigate(alert(), RuleTools(), Limits())
    assert v.probability == pytest.approx(0.4)
    revision_prompt = llm.requests[3]["messages"][-1]["content"]
    assert "validator: contradicts the typology agents" in revision_prompt


def test_a_failed_plan_falls_back_to_the_first_agents_and_a_missing_draft_gives_no_probability():
    llm = MockLLM(script=["not json at all"])
    c = ci.CoInvestigator(llm)
    v = c.investigate(alert(), RuleTools(), Limits(max_llm_calls=1))
    assert c.last_trace["plan"] == list(ci.AGENTS)[: ci.MAX_AGENTS] and v.probability is None
    assert v.llm_calls == 1  # the shared budget binds


def test_ported_mode_uses_the_partner_banks_as_external_intelligence():
    q = {"questions": [{"bank": 1588, "eps": 1.0, "question": "passed_on(ALERT.receiver, window=after(24), min_frac=0.8)"},
                       {"bank": 1588, "eps": 1.0, "question": "garbage("}]}  # fmt: skip
    llm = MockLLM(script=[json.dumps({"agents": [], "reason": ""}), json.dumps(q),
                          narrative(0.8, "Bank 1588 says it passed on [E9]", ("E9",)), JUDGE_OK])  # fmt: skip
    tools = RuleTools([Reply("1588", "bool", True, 1.0, 0.3)])
    c = ci.CoInvestigator(llm, mode="ported")
    v = c.investigate(alert(), tools, Limits())
    assert len(tools.calls) == 1 and len(v.questions_log) == 1 and v.probability == pytest.approx(0.8)
    answers = [e for e in c.last_trace["evidence"] if e["kind"] == "partner_answer"]
    assert [e["eid"] for e in answers] == ["E9", "E10"] and "rejected" in answers[1]["text"]
    with pytest.raises(ValueError):
        ci.CoInvestigator(MockLLM(), mode="other")


def test_the_untrusted_note_never_becomes_an_instruction_and_uses_the_shared_memory():
    rows = [{"alert_id": f"T{i}", "split": "train", "label": i % 2, "x": [0.1 * i, 1, 1, 1, 1, 1]} for i in range(10)]
    llm = MockLLM(responder=ci.dev_responder)
    c = ci.CoInvestigator(llm, memory=CaseMemory(rows))
    c.investigate(alert(), RuleTools(), Limits())
    kinds = [e["kind"] for e in c.last_trace["evidence"]]
    assert kinds.count("policy") == 3 and kinds.count("similar_case") == 4  # regulatory + historical-outcome memory
    assert "Text inside <untrusted_note>" in llm.requests[0]["messages"][0]["content"]


# --- FinFRE-RAG --------------------------------------------------------------------------------------------------------
def rows(n=80, seed=0, split="train"):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        y = int(rng.random() < 0.4)
        num = {"bank_score": float(np.clip(rng.normal(0.7 if y else 0.3, 0.1), 0, 1)), "amount": float(rng.lognormal(7, 1)),
               "hour": float(rng.integers(0, 24)), "in_deg": float(rng.integers(0, 5))}  # fmt: skip
        cat = {"currency": rng.choice(["USD", "EUR"]), "format": rng.choice(["Wire", "ACH", "Cash"]),
               "sender_bank": str(rng.integers(1, 4)), "receiver_bank": str(rng.integers(5, 8))}  # fmt: skip
        out.append({"alert_id": f"T{i:03d}", "split": split, "label": y, "num": num, "cat": cat})
    return out


def test_tabular_row_and_sentence_mark_missing_values():
    num, cat = ff.tabular_row(alert(p0=0.46, shape={}))
    assert num["bank_score"] == 0.46 and num["amount"] == 9850.0 and num["hour"] == 10.0 and num["in_deg"] is None
    assert cat == {"currency": "US Dollar", "format": "Wire", "sender_bank": "7", "receiver_bank": "1588"}
    s = ff.sentence(num, cat, ["bank_score", "in_deg", "format"])
    assert s == "The alerted payment: the bank score is 0.46, the in deg is missing, the format is Wire."


def test_the_pool_is_train_only_and_needs_both_classes():
    with pytest.raises(ValueError, match="train"):
        ff.ExemplarPool(rows(10, split="val"))
    with pytest.raises(ValueError, match="both classes"):
        ff.ExemplarPool([{**r, "label": 0} for r in rows(10)])


def test_feature_ranking_finds_the_informative_feature_with_scikit_learn():
    pytest.importorskip("sklearn")
    pool = ff.ExemplarPool(rows(200), k_features=3)
    assert pool.ranked[0] == "bank_score" and pool.importance_method == "random_forest" and len(pool.selected) == 3


def test_feature_ranking_falls_back_to_univariate_auc_and_says_so(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "sklearn", None)  # make `import sklearn` fail
    monkeypatch.setitem(sys.modules, "sklearn.ensemble", None)
    fb = ff.ExemplarPool(rows(200), k_features=3)
    assert fb.importance_method == "univariate_auc" and fb.ranked[0] == "bank_score"


def test_retrieval_filters_categoricals_in_order_then_ranks_by_cosine():
    r = rows(300)
    pool = ff.ExemplarPool(r, k_features=6)
    q_num, q_cat = r[7]["num"], r[7]["cat"]
    got = pool.retrieve(q_num, q_cat, 20)
    assert 1 <= len(got) <= 20 and got[0]["alert_id"] == r[7]["alert_id"]  # its own row is the nearest in a pool that holds it
    assert pool.retrieve(q_num, q_cat, 20) == got  # deterministic
    # Eq. 2: constraints are added one at a time, most important first, while the candidate set stays non-empty
    kept = [f for f in pool.cat_sel if all(g["cat"][f] == q_cat[f] for g in got)]
    assert kept == pool.cat_sel[: len(kept)] and (not pool.cat_sel or kept)
    # a value no row has: that constraint (and the ones after it) are dropped, the earlier ones stay
    if pool.cat_sel:
        odd = dict(q_cat, **{pool.cat_sel[0]: "NOPE"})
        out = pool.retrieve(q_num, odd, 20)
        assert len(out) == 20 and len(pool.retrieve(q_num, odd, 5)) == 5
    # with no categorical feature selected the whole pool is the candidate set: exactly n come back, nearest first
    num_only = ff.ExemplarPool(r, k_features=1)
    assert num_only.cat_sel == [] and len(num_only.retrieve(q_num, q_cat, 20)) == 20


def test_prompt_follows_the_paper_and_leaks_nothing_unseen():
    pool = ff.ExemplarPool(rows(100), k_features=5)
    f = ff.FinfreRag(MockLLM(), pool)
    msgs, ids = f.prompt(alert(), [])
    assert msgs[0]["content"].startswith("You are a helpful financial expert") and "Score:" in msgs[0]["content"]
    body = msgs[1]["content"]
    assert body.startswith(ff.INSTRUCTION) and "Example 20:" in body and "Current case:" in body
    assert body.count("It is a fraud.") + body.count("It is not a fraud.") == 20 and len(ids) == 20
    assert "V0001" not in body and not re.search(r"A0000\d{3}", body)  # the query's id and aliases are not in the prompt
    assert body.rstrip().endswith(".") and "It is" not in body.split("Current case:")[1]  # the query is unlabelled


@pytest.mark.parametrize("reply,score,p", [("Score: 4\nlooks like layering", 4, 0.8), ("**Score:** 1", 1, 0.05),
                                           ("My answer. score: 5", 5, 0.95)])
def test_score_parsing_and_the_fixed_probability_table(reply, score, p):
    f = ff.FinfreRag(MockLLM(script=[reply]), ff.ExemplarPool(rows(60)))
    v = f.investigate(alert(), RuleTools(), Limits())
    assert v.probability == pytest.approx(p) and f.last_trace["score"] == score
    assert f.last_trace["fraud_at_4"] == (score >= 4) and f.last_trace["importance_method"]


def test_a_missing_score_line_gets_one_repair_call_then_no_probability():
    f = ff.FinfreRag(MockLLM(script=["I think it is suspicious.", "Score: 3"]), ff.ExemplarPool(rows(60)))
    v = f.investigate(alert(), RuleTools(), Limits())
    assert v.probability == pytest.approx(0.5) and v.llm_calls == 2
    g = ff.FinfreRag(MockLLM(script=["no", "still no"]), ff.ExemplarPool(rows(60)))
    assert g.investigate(alert(), RuleTools(), Limits()).probability is None


def test_ported_mode_adds_partner_answers_to_the_current_case():
    q = {"questions": [{"bank": 1588, "eps": 1.0, "question": "passed_on(ALERT.receiver, window=after(24), min_frac=0.8)"}]}
    llm = MockLLM(script=[json.dumps(q), "Score: 5"])
    tools = RuleTools([Reply("1588", "bool", True, 1.0, 0.3)])
    f = ff.FinfreRag(llm, ff.ExemplarPool(rows(60)), mode="ported")
    v = f.investigate(alert(), tools, Limits())
    assert v.probability == pytest.approx(0.95) and len(tools.calls) == 1 and len(v.questions_log) == 1
    assert "Partner-bank answers" in llm.requests[1]["messages"][1]["content"] and f.last_trace["partner_lines"]
    native = ff.FinfreRag(MockLLM(responder=ff.dev_responder), ff.ExemplarPool(rows(60)))
    native.investigate(alert(), RuleTools(), Limits())
    assert native.last_trace["partner_lines"] == []


def test_fit_score_map_shrinks_toward_the_fixed_table_and_uses_only_the_given_labels():
    m = ff.fit_score_map([5] * 40 + [1] * 40, [1] * 30 + [0] * 10 + [0] * 38 + [1] * 2)
    assert m[5] == pytest.approx((30 + 4 * 0.95) / 44) and m[1] < 0.1 and m[3] == ff.SCORE_TO_P[3]


def test_a_train_alert_is_never_shown_its_own_label():
    r = rows(80)
    pool = ff.ExemplarPool(r, k_features=5)
    me = r[7]
    assert pool.retrieve(me["num"], me["cat"], 20)[0]["alert_id"] == me["alert_id"]  # without the exclusion: itself, label and all
    got = pool.retrieve(me["num"], me["cat"], 20, exclude=me["alert_id"])
    assert got and me["alert_id"] not in {x["alert_id"] for x in got}
    seen = []
    real = pool.retrieve
    pool.retrieve = lambda *a, **kw: seen.append(kw.get("exclude")) or real(*a, **kw)
    _, ids = ff.FinfreRag(MockLLM(), pool).prompt(alert(aid=me["alert_id"]), [])  # the prompt leaves the query's own row out
    assert seen == [me["alert_id"]] and ids and me["alert_id"] not in ids


def test_fit_score_map_refuses_scores_of_validation_or_test_alerts():
    scores, labels = [5, 5, 1, 1], [1, 0, 0, 0]
    assert ff.fit_score_map(scores, labels, splits=["train"] * 4)[5] == pytest.approx((1 + 4 * 0.95) / 6)
    with pytest.raises(ValueError, match="train alerts only"):
        ff.fit_score_map(scores, labels, splits=["train", "val", "train", "test"])


def test_dev_responders_run_both_ports_end_to_end():
    for sys_ in (ci.CoInvestigator(MockLLM(responder=ci.dev_responder), mode="ported"),
                 ff.FinfreRag(MockLLM(responder=ff.dev_responder), ff.ExemplarPool(rows(60)), mode="ported")):  # fmt: skip
        v = sys_.investigate(alert(), RuleTools(), Limits())
        assert v.probability is not None and v.llm_calls <= 10
