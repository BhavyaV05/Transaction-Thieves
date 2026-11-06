"""Group C: the ported StateGraph agents (Sharma, Torres, Co-Investigator, FinFRE-RAG) reproduce the old loops exactly.

Each test runs the legacy loop and the graph on the same deterministic fuzzing mock LLM + toolbox over many seeds, in native and ported
modes, and compares verdict, probability, question log, trace, last_* attributes and every LLM request. A branch counter on the legacy run
asserts the fuzzing really reaches memory hits, verifier repair, validator reject/accept, the governed fallback, and so on.
"""

import json
from collections import Counter

import numpy as np
import pytest

pytest.importorskip("langgraph")

from graph_agents_support import (  # noqa: E402
    ALL_QS,
    alert,
    rng_for,
    same,
)

from ftm.compliance.gate import Gate  # noqa: E402
from ftm.external import co_investigator as old_coinv  # noqa: E402
from ftm.external import finfre_rag as old_ff  # noqa: E402
from ftm.external import sharma as old_sharma  # noqa: E402
from ftm.external import torres as old_torres  # noqa: E402
from ftm.external.case_memory import CaseMemory  # noqa: E402
from ftm.graph.agents import co_investigator, finfre_rag, sharma, torres  # noqa: E402
from ftm.graph.agents.registry import ported_group_c  # noqa: E402
from ftm.llm.devtools import DevRandomToolBox  # noqa: E402
from ftm.types import Limits  # noqa: E402

SHAPE = {"in_deg": "4-7", "out_deg": "1", "n_banks": "2", "amount": "1k-10k varied", "span": "1-3d"}


def g_alert(p0=0.4):
    a = alert(p0)
    a.summary["shape"] = SHAPE
    a.summary["top_features"] = [{"feature": "n_in", "shap": 0.2}]
    return a


def memory():
    return CaseMemory([{"alert_id": f"T{i}", "split": "train", "label": i % 2, "x": [0.1 * i, 1, 1, 1, 1, 1]} for i in range(10)])


def question_obj(r, schema):
    props = schema["properties"]["questions"]["items"]["properties"]
    eps = props["eps"].get("enum") or [0.5]
    return {"bank": r.choice(props["bank"]["enum"]), "eps": r.choice(eps + [7.0]), "question": r.choice(ALL_QS)}


def questions_reply(r, schema):
    return json.dumps({"questions": [question_obj(r, schema) for _ in range(r.randint(0, 4))]})


def cited(r, ids, n=3):
    ids = ids or ["E1"]
    cites = " ".join(f"[{r.choice(ids + ['E99'] * (r.random() < 0.12))}]" for _ in range(r.randint(0, n)))
    extra = r.choice(["", "", " amount 4,000.00", " amount 12,345.67", " on 2022-09-01", " on 2031-01-01"])
    return f"The flow {cites}{extra}"


# -- responders ----------------------------------------------------------------------------------------------------------
def sharma_responder(salt):
    def f(messages, kw):
        r = rng_for(messages, kw, salt)
        schema = kw["json_schema"]
        if r.random() < 0.07:
            return "no json at all"
        p = schema["properties"]
        tool = r.choice(p["tool"]["enum"] if r.random() < 0.85 else ["submit_verdict"])
        bank = eps = q = verdict = rationale = None
        if tool == "ask_bank":
            if "anyOf" in p["bank"] and r.random() < 0.9:
                bank = r.choice(p["bank"]["anyOf"][0]["enum"])
            if "anyOf" in p["eps"] and r.random() < 0.9:
                eps = r.choice(p["eps"]["anyOf"][0]["enum"])
            q = r.choice(ALL_QS) if r.random() < 0.9 else None
        if tool == "submit_verdict":
            verdict = r.choice(["fraud", "not_fraud", "escalate", None] if r.random() < 0.9 else ["fraud"])
            rationale = r.choice([None, "because [x]"])
        return json.dumps({"tool": tool, "bank": bank, "eps": eps, "question": q, "verdict": verdict, "rationale": rationale})

    return f


def torres_responder(salt):
    def f(messages, kw):
        r = rng_for(messages, kw, salt)
        schema = kw["json_schema"]
        if schema["title"] == "torres_questions":
            return questions_reply(r, schema) if r.random() < 0.9 else "oops"
        if r.random() < 0.08:
            return "garbled"
        ids = schema["properties"]["supporting_evidence"]["items"].get("enum") or []
        out = {"decision": r.choice(["dismiss", "monitor", "escalate"]), "confidence": r.choice([0.1, 0.5, 0.8, 0.95]),
               "supporting_evidence": r.sample(ids, min(len(ids), r.randint(0, 3))) if ids else [],
               "contradicting_or_missing": r.sample(ids, min(len(ids), r.randint(0, 2))) if ids else [],
               "rationale": cited(r, ids if r.random() < 0.5 else [f"E{i}" for i in range(1, 12)]), "next_actions": ["a"]}  # fmt: skip
        return json.dumps(out)

    return f


def coinv_responder(salt):
    def f(messages, kw):
        r = rng_for(messages, kw, salt)
        schema = kw["json_schema"]
        title, p = schema["title"], schema["properties"]
        if r.random() < 0.06:
            return "garbled"
        if title == "coinv_plan":
            from ftm.external.co_investigator import AGENTS

            return json.dumps({"agents": r.sample(list(AGENTS), r.randint(0, 4)), "reason": "r"})
        if title == "coinv_questions":
            return questions_reply(r, schema)
        if title == "coinv_judge":
            return json.dumps({"coherent": r.random() < 0.5, "issues": ["thin"] * r.randint(0, 2)})
        ids_of = lambda key: p[key]["items"].get("enum") or []  # noqa: E731
        if title == "coinv_agent":
            ids = ids_of("evidence")
            return json.dumps({"flag": r.random() < 0.5, "strength": r.random(), "finding": "f", "evidence": r.sample(ids, min(len(ids), 2))})
        ids = ids_of("supporting_evidence")
        return json.dumps({"reasoning": "r", "narrative": cited(r, ids if r.random() < 0.7 else ["E1", "E2", "E40"]),
                           "confidence": r.choice([0.1, 0.6, 0.9]), "supporting_evidence": r.sample(ids, min(len(ids), 2)),
                           "contradicting_or_missing": []})  # fmt: skip

    return f


def ff_responder(salt):
    def f(messages, kw):
        r = rng_for(messages, kw, salt)
        schema = kw.get("json_schema")
        if schema:
            return questions_reply(r, schema) if r.random() < 0.9 else "oops"
        return r.choice(["Score: 4 plausible", "Score: **2**", "no score line", "Score: 7", "Score: 5"])

    return f


def ff_pool():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(80):
        y = int(rng.random() < 0.4)
        rows.append({"alert_id": f"T{i:03d}", "split": "train", "label": y,
                     "num": {"bank_score": float(np.clip(rng.normal(0.7 if y else 0.3, 0.1), 0, 1)), "amount": float(rng.lognormal(7, 1)),
                             "hour": float(rng.integers(0, 24)), "in_deg": float(rng.integers(0, 5))},
                     "cat": {"currency": str(rng.choice(["US Dollar", "EUR"])), "format": str(rng.choice(["Wire", "ACH"])),
                             "sender_bank": str(rng.integers(5, 9)), "receiver_bank": str(rng.integers(1, 3))}})  # fmt: skip
    return old_ff.ExemplarPool(rows, k_features=4)


class Tools(DevRandomToolBox):
    """The fuzzing toolbox plus the real rule lookup (Torres / Co-Investigator bundle the policy items)."""

    def __init__(self, seed):
        super().__init__(seed=seed, p_decline=0.15, p_timeout=0.05)
        self.gate = Gate.from_config()

    def lookup_rules(self, query):
        return self.gate.lookup_rules(query)


@pytest.fixture(autouse=True)
def _real_tools(monkeypatch):
    import graph_agents_support as sup

    monkeypatch.setattr(sup, "toolbox", Tools)
    return None


def run_cases(old, new, responder, observe, *, mode, seeds, **kw):
    """Parity over `seeds` for each limits / alert variant; returns the total of runs that asked a question."""
    total = 0
    for lim in (Limits(), Limits(max_llm_calls=3), Limits(eps_per_alert=0.5)):
        for p0 in (0.2, 0.5, 0.85):
            total += same(lambda llm: old(llm, mode=mode, **kw()), lambda llm: new(llm, mode=mode, **kw()), responder, seeds=seeds,
                          limits=lim, alert_fn=lambda p0=p0: g_alert(p0), extra_attrs=("last_verdict", "last_views"), observe=observe)  # fmt: skip
    return total


# -- Sharma --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["native", "ported"])
@pytest.mark.parametrize("variant", ["agent", "governed"])
def test_sharma_parity(mode, variant):
    seen = Counter()

    def observe(snap, agent):
        for t in agent.last_trace:
            seen[t["type"]] += 1
            if t["type"] == "tool_call" and t["name"] == "get_similar_past_cases" and "closest_known_laundering" in str(t["result"]):
                seen["memory_hit"] += 1
            if t["type"] == "tool_call" and t["name"] == "ask_bank" and str(t["result"]).startswith("Invalid"):
                seen["invalid_question"] += 1
        v = agent.last_verdict
        if v:
            seen["final"] += 1
            seen["fallback"] += v["final"] != v["verdict"]
        else:
            seen["no_verdict"] += 1

    mk = lambda: {"variant": variant, "memory": memory()}  # noqa: E731
    n = 0
    for lim in (Limits(), Limits(max_llm_calls=3)):
        for p0 in (0.2, 0.5, 0.85):
            n += same(lambda llm: old_sharma.Sharma(llm, mode=mode, **mk()), lambda llm: sharma.Sharma(llm, mode=mode, **mk()),
                      sharma_responder, seeds=range(25), limits=lim, alert_fn=lambda p0=p0: g_alert(p0),
                      extra_attrs=("last_verdict",), observe=observe)  # fmt: skip
    assert seen["memory_hit"] and seen["text"] and seen["verdict"] and seen["no_verdict"] and seen["tool_call"]
    if variant == "governed":
        assert seen["fallback"], "the governed escalation fallback was never reached"
    if mode == "ported":
        assert n > 5 and seen["invalid_question"]


# -- Torres --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["native", "ported"])
def test_torres_parity(mode):
    seen = Counter()

    def observe(snap, agent):
        t = agent.last_trace
        kinds = {i["kind"] for i in t["evidence"]}
        seen["memory_hit"] += "similar_case" in kinds
        seen["partner_answer"] += "partner_answer" in kinds
        seen["output"] += t["output"] is not None
        seen["no_output"] += t["output"] is None
        seen["problems"] += bool(t["verifier_problems"])  # the verifier rejected, repair did not (or could not) fix it
        seen["accepted"] += not t["verifier_problems"]
        seen["cf"] += bool(t.get("counterfactuals"))
        seen["cf_decorative"] += any(c["decorative"] for c in t.get("counterfactuals", []))
        seen["cf_changed"] += any(c["changed"] for c in t.get("counterfactuals", []))
        seen["repair_call"] += any(m["role"] == "user" and "rejected by the verifier" in m["content"]
                                   for req in snap["_requests"] for m in req[0])  # fmt: skip
        seen["repaired"] += any("rejected by the verifier" in m["content"] for req in snap["_requests"] for m in req[0]) and not t["verifier_problems"]

    mk = lambda: {"memory": memory()}  # noqa: E731
    n = 0
    for lim in (Limits(), Limits(max_llm_calls=3), Limits(max_llm_calls=1)):
        for p0 in (0.2, 0.85):
            n += same(lambda llm: old_torres.Torres(llm, mode=mode, **mk()), lambda llm: torres.Torres(llm, mode=mode, **mk()),
                      torres_responder, seeds=range(25), limits=lim, alert_fn=lambda p0=p0: g_alert(p0), observe=observe)  # fmt: skip
    for k in ("memory_hit", "output", "no_output", "problems", "accepted", "cf", "cf_decorative", "repair_call", "repaired"):
        assert seen[k], f"branch never reached: {k} ({dict(seen)})"
    if mode == "ported":
        assert seen["partner_answer"] and n > 5


# -- Co-Investigator -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["native", "ported"])
@pytest.mark.parametrize("emit", [False, True])
def test_co_investigator_parity(mode, emit):
    seen = Counter()

    def observe(snap, agent):
        t, v = agent.last_trace, agent.last_views
        seen["memory_hit"] += any(i["kind"] == "similar_case" for i in t["evidence"])
        seen["partner_answer"] += any(i["kind"] == "partner_answer" for i in t["evidence"])
        seen["no_draft"] += t["draft"] is None or "_invalid" in t["draft"]
        seen["accepted"] += bool(v["validated"])
        seen["rejected"] += t["draft"] is not None and "_invalid" not in t["draft"] and not v["validated"]
        seen["emitted_unvalidated"] += (not v["validated"]) and v["p_draft"] is not None and snap["probability"] is not None
        seen["held_back"] += (not v["validated"]) and v["p_draft"] is not None and snap["probability"] is None
        seen["revised"] += sum(1 for req in snap["_requests"] if "compliance validator flagged" in json.dumps(req[0])) > 0
        seen["findings"] += bool(t["findings"])

    mk = lambda: {"memory": memory(), "emit_unvalidated": emit}  # noqa: E731
    n = 0
    for lim in (Limits(), Limits(max_llm_calls=4), Limits(max_llm_calls=8)):
        for p0 in (0.2, 0.85):
            n += same(lambda llm: old_coinv.CoInvestigator(llm, mode=mode, **mk()), lambda llm: co_investigator.CoInvestigator(llm, mode=mode, **mk()),
                      coinv_responder, seeds=range(25), limits=lim, alert_fn=lambda p0=p0: g_alert(p0),
                      extra_attrs=("last_views",), observe=observe)  # fmt: skip
    for k in ("memory_hit", "no_draft", "accepted", "rejected", "revised", "findings"):
        assert seen[k], f"branch never reached: {k} ({dict(seen)})"
    assert seen["emitted_unvalidated" if emit else "held_back"]
    if mode == "ported":
        assert seen["partner_answer"] and n > 5


# -- FinFRE-RAG ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["native", "ported"])
def test_finfre_rag_parity(mode):
    seen = Counter()
    pool = ff_pool()

    def observe(snap, agent):
        t = agent.last_trace
        seen["scored"] += t["score"] is not None
        seen["unscored"] += t["score"] is None
        seen["repair"] += sum(1 for r in snap["_requests"] if "You must include 'Score:'" in json.dumps(r[0])) > 0
        seen["exemplars"] += len(t["exemplars"]) == 20
        seen["partner"] += bool(t["partner_lines"])
        seen[f"score{t['score']}"] += 1

    n = 0
    for lim in (Limits(), Limits(max_llm_calls=1), Limits(eps_per_alert=0.5)):
        for p0 in (0.2, 0.85):
            n += same(lambda llm: old_ff.FinfreRag(llm, pool, mode=mode), lambda llm: finfre_rag.FinfreRag(llm, pool, mode=mode),
                      ff_responder, seeds=range(25), limits=lim, alert_fn=lambda p0=p0: g_alert(p0), observe=observe)  # fmt: skip
    for k in ("scored", "unscored", "repair", "exemplars"):
        assert seen[k], f"branch never reached: {k} ({dict(seen)})"
    if mode == "ported":
        assert seen["partner"] and n > 5


def test_registry_default_is_graph_and_legacy_switch(monkeypatch):
    c = ported_group_c()
    assert c["sharma"] is sharma.Sharma and c["torres"] is torres.Torres
    assert c["co_investigator"] is co_investigator.CoInvestigator and c["finfre_rag"] is finfre_rag.FinfreRag
    monkeypatch.setenv("FTM_LEGACY_AGENT_LOOPS", "1")
    assert ported_group_c()["sharma"] is old_sharma.Sharma
