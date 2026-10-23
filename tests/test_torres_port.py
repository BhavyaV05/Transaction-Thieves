"""T4: Torres et al. [6] reimplementation on the mock LLM."""

import json

import pytest

from ftm.compliance.gate import Gate
from ftm.external import torres as tr
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


def _alert():
    ev = "\n".join(f"2022-09-0{d} 10:00 | A000000{d} -> A0000002 | {1000 * d:,.2f} US Dollar | Wire | bank 7 -> bank 1588"
                   for d in range(1, 5))  # fmt: skip
    s = {"sender": "A0000001", "receiver": "A0000002", "sender_bank": 7, "receiver_bank": 1588, "amount": 9850.0,
         "currency": "US Dollar", "format": "Wire", "time": "2022-09-05 10:00", "evidence": ev, "partner_banks": [1588],
         "shape": {"in_deg": "4-7", "out_deg": "1", "n_banks": "2", "amount": "1k-10k varied", "span": "1-3d"}}  # fmt: skip
    return AlertContext("V0001", "7", "2022-09-05T10:00:00", "val", 0.46, s)


def out(decision="escalate", conf=0.8, support=("E5",), rationale="Four payers within days [E5] [E6]", **kw):
    o = {"decision": decision, "confidence": conf, "supporting_evidence": list(support), "contradicting_or_missing": [],
         "rationale": rationale, "next_actions": ["request statements"]}  # fmt: skip
    o.update(kw)
    return json.dumps(o)


def test_disposition_probability_mapping():
    assert tr.disposition_probability("escalate", 0.8) == pytest.approx(0.8 + 0.5 * 0.1)
    assert tr.disposition_probability("dismiss", 0.8) == pytest.approx(0.1 + 0.5 * 0.1)
    assert tr.disposition_probability("monitor", 1.0) == pytest.approx(0.5)


def test_bundle_has_quotas_and_ids():
    rows = [{"alert_id": f"T{i}", "split": "train", "label": i % 2, "x": [0.1 * i, 1, 1, 1, 1, 1]} for i in range(10)]
    t = tr.Torres(MockLLM(), memory=CaseMemory(rows))
    items = t.bundle(_alert(), RuleTools())
    kinds = [i.kind for i in items]
    assert kinds.count("policy") == 3 and kinds.count("context") == 1 and kinds.count("transaction") == 4
    assert kinds.count("similar_case") == 4
    assert [i.eid for i in items] == [f"E{n}" for n in range(1, len(items) + 1)]
    assert "0.460" in items[3].text  # the context item carries the bank's score


def test_verifier_catches_fake_citations_amounts_and_dates():
    items = tr.Torres(MockLLM()).bundle(_alert(), RuleTools())
    good = json.loads(out(rationale="Payments of 4,000.00 on 2022-09-04 [E5]"))
    assert tr.verify(good, items) == []
    bad = json.loads(out(support=("E99",), rationale="A 123,456 wire on 2021-01-01, no citation"))
    probs = " ".join(tr.verify(bad, items))
    assert "E99" in probs and "cites no evidence" in probs and "123,456" in probs and "2021-01-01" in probs


def test_repair_step_then_counterfactuals():
    llm = MockLLM(script=[
        out(rationale="huge 777,777 transfer [E5]"),            # rejected: amount not in evidence
        out(rationale="Four payers [E5] [E6]"),                  # repaired
        out(decision="monitor", conf=0.6, support=(), rationale="less evidence [E7]"),  # CF 1: decision moves
        out(rationale="still [E5]", support=()),                 # CF 2: same decision, cites dropped E5 -> decorative
    ])  # fmt: skip
    t = tr.Torres(llm)
    v = t.investigate(_alert(), RuleTools(), Limits())
    assert v.llm_calls == 4 and v.probability == pytest.approx(0.85)
    assert t.last_trace["verifier_problems"] == []
    cf = t.last_trace["counterfactuals"]
    assert cf[0]["changed"] and not cf[0]["decorative"]
    assert cf[1]["decorative"] and cf[1]["cites_dropped"] == ["E5"]


def test_failed_repair_keeps_output_and_logs_problems():
    llm = MockLLM(script=[out(rationale="no citation at all"), out(rationale="still none")])
    t = tr.Torres(llm)
    v = t.investigate(_alert(), RuleTools(), Limits(max_llm_calls=2))
    assert v.probability is not None and t.last_trace["verifier_problems"]


def test_ported_question_round_adds_partner_evidence():
    q = {"questions": [{"bank": 1588, "eps": 1.0, "question": "passthrough(ALERT.receiver, min_frac=0.8, within_h=24)"},
                       {"bank": 1588, "eps": 1.0, "question": "not a question"}]}  # fmt: skip
    llm = MockLLM(script=[json.dumps(q), out(rationale="bank 1588 says it passed on [E9]", support=("E9",))])
    tools = RuleTools([Reply("1588", "bool", True, 1.0, 0.3)])
    t = tr.Torres(llm, mode="ported")
    v = t.investigate(_alert(), tools, Limits(max_llm_calls=2))
    answers = [e for e in t.last_trace["evidence"] if e["kind"] == "partner_answer"]
    assert len(tools.calls) == 1 and len(v.questions_log) == 1
    assert len(answers) == 2 and "rejected" in answers[1]["text"]
    assert [e["eid"] for e in answers] == ["E9", "E10"]  # 3 policy + 1 context + 4 transactions, then the answers
    assert v.probability == pytest.approx(0.85)


def test_native_mode_never_asks_banks():
    tools = RuleTools()
    tr.Torres(MockLLM(script=[out()])).investigate(_alert(), tools, Limits(max_llm_calls=1))
    assert tools.calls == []
