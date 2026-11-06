"""Parity: our planner as a LangGraph policy (ftm/graph/ours.py) == the frozen pre-graph loop (tests/legacy_llm_planner.py) on the
same scripted LLM answers: same verdict probability, question logs, tokens, traces, planner stats and bank calls, in every mode,
with retries (bad JSON, bad DSL, repeats, refusals, routing), batched waves and a prompt refit."""

import json
import math
from dataclasses import replace

import pytest
from legacy_llm_planner import LLMPlanner as Legacy

from ftm.graph.ours import OursPolicy
from ftm.graph.pipeline import GraphInvestigator
from ftm.llm.budget import estimate_messages
from ftm.llm.investigation import LREntry, LRTable, NaiveLRFusion
from ftm.llm.mock import MockLLM
from ftm.planners.llm import LLMPlanner
from ftm.planners.llm_dsl import V0Dsl
from ftm.types import AlertContext, Limits, Reply

DSL = V0Dsl()
Q_PASS = "passed_on(ALERT.receiver, window=after(24), min_frac=0.8)"
Q_FAN = "at_least(n_counterparties(ALERT.receiver, dir=in, window=before(48)), k=5)"
Q_FAN2 = "at_least(n_counterparties(ALERT.receiver, dir=in, window=before(24)), k=3)"
Q_HOME = "passed_on(ALERT.sender, window=after(24), min_frac=0.8)"  # the sender is at the home bank in ROUTED
BAD = "passed_on(ALERT.receiver, window="
PLAIN = AlertContext("V000001", "70", "2022-09-05T12:00:00", "val", 0.2,
                     {"sender": "A1", "receiver": "A2", "amount": 9500.0, "evidence": "row1\nrow2", "partner_banks": [220, 1588]})  # fmt: skip
ROUTED = replace(PLAIN, summary={**PLAIN.summary, "sender_bank": 70, "receiver_bank": 220})
TABLE = LRTable([LREntry(DSL.validate(Q_PASS).canonical, e, True, {"true": 4.0, "false": 0.5}, "passed_on") for e in (0.5, 1.0)]
                + [LREntry(DSL.validate(Q_FAN).canonical, 0.5, True, {"true": 3.0, "false": 0.4}, "at_least:n_counterparties:in"),
                   LREntry(DSL.validate(Q_FAN2).canonical, 0.5, False, {"true": 9.0, "false": 0.1}, "at_least:n_counterparties:in")])  # fmt: skip
MENU = [{"id": "B1", "program": DSL.validate(Q_PASS).canonical, "family": "passed_on", "eps": [0.5, 1.0], "slots": ["receiver"], "lr_text": "x0.5 to x4"},
        {"id": "B2", "program": DSL.validate(Q_FAN).canonical, "family": "at_least", "eps": [0.5], "slots": ["receiver"], "lr_text": "x0.4 to x3"}]  # fmt: skip


class BudgetRefused(Exception):  # classified by name as an accountant refusal
    pass


class Box:
    """Deterministic harness stand-in: answers by hashing the question, blocks / refuses chosen programs, logs every call."""

    def __init__(self, blocked=(), used_up=(), waves=True):
        self.blocked, self.used_up, self.calls = set(blocked), set(used_up), []
        if not waves:
            self.__class__ = SingleBox

    def lookup_rules(self, q):
        return [{"rule_id": "R1", "text": "rule one"}, {"rule_id": "R2", "snippet": "rule two"}]

    def search_library(self, shape):
        return [{"text": f"strategy {shape.get('typology')}"}]

    def local_graph(self, query, as_of):
        return []

    def _one(self, bank, program, eps):
        self.calls.append((str(bank), program, eps))
        if any(b in program for b in self.blocked):
            raise PermissionError("blocked: not permitted")
        if any(b in program for b in self.used_up):
            raise BudgetRefused("per-alert eps budget of this run is used up")
        h = sum(map(ord, f"{bank}{program}{eps}"))
        if "n_counterparties" in program:
            return Reply(str(bank), "count", h % 9, eps, 0.2 + (h % 7) / 10)
        return Reply(str(bank), "bool", h % 3 != 0, eps, 0.2 + (h % 7) / 10)

    def ask_bank(self, bank, program, eps):
        return self._one(bank, program, eps)

    def ask_banks(self, requests):
        out = []
        for b, p, e in requests:
            try:
                out.append(self._one(b, p, e))
            except (PermissionError, BudgetRefused) as exc:
                out.append(exc)
        return out


class SingleBox(Box):  # a toolbox without the wave extension (Investigation falls back to ask_bank one by one)
    def __getattribute__(self, name):
        if name == "ask_banks":
            raise AttributeError(name)
        return super().__getattribute__(name)


def one(action="compose", program=Q_PASS, bank=220, eps=0.5, qid=None, belief=None, typology="fan_in"):
    d = {"hypothesis": {"typology": typology, "rationale": "r"}, "action": action, "question_id": qid,
         "program": program if action == "compose" else None, "bank": bank if action != "stop" else None,
         "eps": eps if action != "stop" else 0.0, "rule_id": None}  # fmt: skip
    if belief is not None:
        d["belief"] = belief
    return json.dumps(d)


def wave(*qs):
    return json.dumps({"hypothesis": {"typology": "fan_in", "rationale": "r"}, "action": "ask" if qs else "stop", "questions": list(qs)})


def wq(program, bank=220, eps=0.5, action="compose", qid=None):
    return {"action": action, "question_id": qid, "program": program if action == "compose" else None, "bank": bank, "eps": eps, "rule_id": None}


STOP = one("stop")
JUNK = "this is not json"

SCENARIOS = {  # name: (mode kwargs, alert, box kwargs, script, limits kwargs)
    "compose_retries_then_stop": (dict(), ROUTED, {}, [JUNK, one(program=BAD), one(), one(), one(program=Q_HOME), one(program=Q_FAN, bank=1588), STOP], {}),
    "compose_unrouted_alert_keeps_model_bank": (dict(), PLAIN, {}, [one(bank=1588), one(program=Q_FAN, bank=220), one(program=Q_FAN2, bank=220), STOP], {}),
    "compose_blocked_then_repeat": (dict(), ROUTED, dict(blocked=["window=after(24)"]), [one(), one(), one(program=Q_FAN), STOP], {}),
    "compose_budget_used_up_ends": (dict(), ROUTED, dict(used_up=["window=after(24)"]), [one(), one(program=Q_FAN), STOP], {}),
    "compose_call_cap": (dict(), ROUTED, {}, [JUNK, one(), one(program=Q_FAN)], dict(max_llm_calls=2)),
    "compose_three_failures": (dict(), ROUTED, {}, [JUNK, JUNK, JUNK, one()], {}),
    "compose_eps_runs_out": (dict(), ROUTED, {}, [one(eps=1.0), one(program=Q_FAN, eps=1.0), one(program=Q_FAN2, eps=0.5)], {}),
    "compose_with_menu_select_snaps_eps": (dict(questions=MENU, fusion=NaiveLRFusion(TABLE)), ROUTED, {},
                                            [one("select", qid="B1", eps=0.25), one("select", qid="B1", eps=1.0), one("select", qid="B2", eps=1.0),
                                             one("select", qid="B1", eps=0.5), STOP], {}),
    "compose_library": (dict(use_library=True, fusion=NaiveLRFusion(TABLE)), ROUTED, {},
                        [one(typology="fan_in"), one(program=Q_FAN, typology="mule"), one(program=Q_FAN2, typology="mule"), STOP], {}),
    "no_rules": (dict(use_rules=False, threshold=0.4), ROUTED, {}, [one(), STOP], {}),
    "fixed_list": (dict(mode="fixed_list", questions=MENU, fusion=NaiveLRFusion(TABLE)), ROUTED, {},
                   [one("select", qid="B2"), one("select", qid="B1", eps=1.0), STOP], {}),
    "act_only": (dict(mode="act_only", fusion=NaiveLRFusion(TABLE)), ROUTED, {}, [one(), one(program=Q_FAN), STOP], {}),
    "llm_decides_belief": (dict(mode="llm_decides", fusion=NaiveLRFusion(TABLE)), ROUTED, {}, [one(belief=0.3), one(program=Q_FAN, belief=0.7), STOP], {}),
    "llm_decides_no_belief": (dict(mode="llm_decides"), ROUTED, {}, [STOP], {}),
    "no_backtest": (dict(mode="no_backtest", questions=MENU, fusion=NaiveLRFusion(TABLE)), ROUTED, {},
                    [one(belief=0.6), one(program=Q_FAN, belief=0.2), one("stop", belief=0.9)], {}),
    "batched_waves": (dict(batch_schedule=(2, 1), fusion=NaiveLRFusion(TABLE)), ROUTED, {},
                      [wave(wq(Q_PASS), wq(Q_FAN, bank=1588)), one(program=Q_FAN2)], {}),
    "batched_wave_rules": (dict(batch_schedule=(4, 3, 2), fusion=NaiveLRFusion(TABLE)), ROUTED, {},
                           [wave(wq(Q_PASS), wq(Q_PASS), wq(BAD), wq(Q_HOME)), wave(wq(Q_FAN, bank=1588, eps=1.0), wq(Q_FAN2, eps=1.0), wq(Q_PASS, eps=0.5)),
                            wave(wq(Q_PASS, eps=0.5), wq(Q_FAN, eps=0.5)), wave(wq(Q_FAN2, bank=1588))], {}),
    "batched_nothing_valid_then_blocked": (dict(batch_schedule=(2, 2)), ROUTED, dict(blocked=["window=after(24)", "window=before(48)"]),
                                           [wave(wq(BAD)), wave(wq(Q_PASS), wq(Q_FAN)), wave(wq(Q_PASS), wq(Q_FAN)), wave(wq(Q_FAN2))], {}),
    "batched_select_menu": (dict(batch_schedule=(2,), questions=MENU, fusion=NaiveLRFusion(TABLE)), ROUTED, {},
                            [wave(wq(None, qid="B1", action="select", eps=0.25), wq(None, qid="B2", action="select"))], {}),
    "batched_no_ask_banks": (dict(batch_schedule=(2, 1)), ROUTED, dict(waves=False), [wave(wq(Q_PASS), wq(Q_FAN)), STOP], {}),
    "batched_used_up": (dict(batch_schedule=(2, 1)), ROUTED, dict(used_up=["window=after(24)", "window=before(48)"]),
                        [wave(wq(Q_PASS), wq(Q_FAN)), one()], {}),
    "batched_stop": (dict(batch_schedule=(2, 1)), ROUTED, {}, [wave()], {}),
}


def build(cls, kw, script):
    return cls(MockLLM(script=list(script)), **kw)


def run(cls, name):
    kw, alert, boxkw, script, lim = SCENARIOS[name]
    box = Box(**boxkw)
    p = build(cls, kw, script)
    v = p.investigate(alert, box, Limits(**lim))
    return v, p, box


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_graph_planner_matches_the_frozen_loop(name):
    v0, p0, b0 = run(Legacy, name)
    v1, p1, b1 = run(LLMPlanner, name)
    assert v1.probability == v0.probability
    assert v1.questions_log == v0.questions_log
    assert (v1.llm_calls, v1.tokens_in, v1.tokens_out) == (v0.llm_calls, v0.tokens_in, v0.tokens_out)
    assert p1.last_trace == p0.last_trace
    assert p1.stats.as_counts() == p0.stats.as_counts() and p1.stats.usable == p0.stats.usable and p1.stats.alerts == p0.stats.alerts == 1
    assert b1.calls == b0.calls


def test_scenarios_exercise_the_branches():
    """The scenarios are only evidence if they reach the interesting branches."""
    seen = {}
    for name in SCENARIOS:
        _, p, _ = run(LLMPlanner, name)
        for rec in p.last_trace:
            for key in ("error", "refit_token_scale", "bank_requested", "eps_requested"):
                if key in rec:
                    seen.setdefault(key, set()).add(str(rec[key]) if key == "error" else key)
            for q in rec.get("wave", []):
                if "error" in q:
                    seen.setdefault("error", set()).add(q["error"])
    errors = seen["error"]
    for want in ("parse", "dsl", "no_partner_holds_subject", "duplicate", "refused", "wave: no valid question", "wave: nothing allowed",
                 "same_family_in_wave", "duplicate_in_wave", "wave_over_eps", "duplicate_refused"):
        assert any(e.startswith(want) for e in errors), (want, errors)
    assert "bank_requested" in seen and "eps_requested" in seen


def test_prompt_refit_matches():
    class Dense(MockLLM):  # a tokenizer that counts 40% more than estimate_messages
        def chat(self, messages, **kw):
            return replace(super().chat(messages, **kw), tokens_in=math.ceil(1.4 * estimate_messages(messages)))

    big = "\n".join(f"2022-09-0{i % 9} 10:{i % 60:02d} | A{i:07x} -> A01bc22d | {i},000.00 US Dollar | ACH" for i in range(300))
    a = replace(ROUTED, summary={**ROUTED.summary, "evidence": big})
    outs = []
    for cls in (Legacy, LLMPlanner):
        p = cls(Dense(script=[STOP, one(), STOP]))
        v = p.investigate(a, Box(), Limits(max_tokens=6000))
        outs.append((v.probability, v.questions_log, v.llm_calls, v.tokens_in, p.last_trace, p.stats.as_counts()))
    assert outs[0] == outs[1]
    assert "refit_token_scale" in outs[1][4][0]


def test_graph_shape_and_hooks():
    p = LLMPlanner(MockLLM(script=[STOP]))
    assert isinstance(p.graph, GraphInvestigator) and isinstance(p.policy, OursPolicy)
    assert {"prepare", "check", "plan", "act", "fuse", "decide"} <= set(p.graph.graph.get_graph().nodes)
    p.investigate(ROUTED, Box(), Limits())
    st = p.graph.last_state
    assert st["stop_reason"] == "model_stop" and st["verdict"].probability == pytest.approx(ROUTED.p0)
    assert {m.qid for m in p.graph.last_state["menu"]} == set()


def test_menu_items_carry_the_bin_to_lr_map_and_the_routed_bank():
    p = LLMPlanner(MockLLM(script=[STOP]), questions=MENU, fusion=NaiveLRFusion(TABLE))
    p.investigate(ROUTED, Box(), Limits())
    menu = p.graph.last_state["menu"]
    assert [(m.qid, m.eps, m.bank, m.family) for m in menu] == [("B1", 0.5, "220", "passed_on"), ("B1", 1.0, "220", "passed_on"),
                                                               ("B2", 0.5, "220", "at_least")]  # fmt: skip
    assert dict(menu[0].lr_bins) == {"true": 4.0, "false": 0.5}


def test_rules_use_the_alert_specific_query():
    seen = []

    class Q(Box):
        def lookup_rules(self, q):
            seen.append(q)
            return super().lookup_rules(q)

    LLMPlanner(MockLLM(script=[STOP])).investigate(ROUTED, Q(), Limits())
    LLMPlanner(MockLLM(script=[STOP])).investigate(replace(ROUTED, summary={**ROUTED.summary, "amount": 5_000_000.0}), Q(), Limits())
    assert len(seen) == 2 and seen[0] != seen[1] and seen[0].startswith("sharing information")
