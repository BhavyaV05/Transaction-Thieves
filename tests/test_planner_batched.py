"""E13 / H8: the LLM planner's batched option (batch_schedule): waves of k questions, then one more."""

import json
from dataclasses import replace

import pytest

from ftm.llm.mock import MockLLM
from ftm.planners.llm import LLMPlanner
from ftm.planners.llm_prompt import (
    Mode,
    PlannerParseError,
    PlannerState,
    parse_action,
    planner_schema,
)
from ftm.types import AlertContext, Limits, Reply

Q_PASS = "passed_on(ALERT.receiver, window=after(24), min_frac=0.8)"
Q_FAN = "at_least(n_counterparties(ALERT.receiver, dir=in, window=before(48)), k=5)"
Q_FAN2 = "at_least(n_counterparties(ALERT.receiver, dir=in, window=before(24)), k=3)"
ALERT = AlertContext("W1", "70", "2022-09-05T12:00:00", "val", 0.2,
                     {"receiver": "R", "evidence": "rows", "partner_banks": [1, 2, 3]})  # fmt: skip
LIMITS = Limits(max_llm_calls=10)
STOP = json.dumps({"hypothesis": {"typology": "none", "rationale": "r"}, "action": "stop", "question_id": None,
                   "program": None, "bank": None, "eps": 0.0, "rule_id": None})


class ClockBox:
    """Mock harness: a clock, ask_banks (a wave costs its slowest answer) and ask_bank (one answer)."""

    def __init__(self, days):
        self.days, self.clock, self.waves, self.singles = days, 0.0, [], 0

    def lookup_rules(self, q):
        return []

    def _reply(self, bank, eps):
        return Reply(bank=str(bank), kind="bool", value=True, eps_spent=eps, latency_days=self.days[bank])

    def ask_bank(self, bank, program, eps):
        self.singles += 1
        r = self._reply(bank, eps)
        self.clock += r.latency_days
        return r

    def ask_banks(self, requests):
        out = [self._reply(b, e) for b, _p, e in requests]
        self.waves.append(len(requests))
        self.clock += max(r.latency_days for r in out)
        return out


def q(program, bank, eps=0.5):
    return {"action": "compose", "question_id": None, "program": program, "bank": bank, "eps": eps, "rule_id": None}


def wave_step(*qs):
    return json.dumps({"hypothesis": {"typology": "fan_in", "rationale": "r"}, "action": "ask" if qs else "stop",
                       "questions": list(qs)})


def one_step(program, bank, eps=0.5):
    return json.dumps({"hypothesis": {"typology": "fan_in", "rationale": "r"}, "action": "compose", "question_id": None,
                       "program": program, "bank": bank, "eps": eps, "rule_id": None})


def planner(script, schedule):
    return LLMPlanner(MockLLM(script=script), mode="compose", batch_schedule=schedule)


def test_wave_advances_clock_by_max_then_single_then_ends():
    box = ClockBox({1: 0.5, 2: 2.0, 3: 1.0})
    p = planner([wave_step(q(Q_PASS, 1), q(Q_FAN, 2)), one_step(Q_FAN2, 3)], (2, 1))
    v = p.investigate(ALERT, box, LIMITS)
    assert box.waves == [2] and box.singles == 1
    assert box.clock == pytest.approx(2.0 + 1.0)  # max(0.5, 2.0), then 1.0; not 0.5 + 2.0 + 1.0
    assert len(v.questions_log) == 3
    assert v.llm_calls == 2  # the schedule is used up: no third call, no stop needed
    assert [r["step"] for r in p.last_trace] == [1, 2]


def test_failed_step_does_not_use_a_slot_and_stop_ends():
    box = ClockBox({1: 0.5, 2: 0.5, 3: 0.5})
    p = planner([wave_step(q("n_tx", 1)), wave_step(q(Q_PASS, 1), q(Q_FAN, 2)), STOP], (2, 1))
    v = p.investigate(ALERT, box, LIMITS)
    assert box.waves == [2] and box.singles == 0 and len(v.questions_log) == 2
    assert v.llm_calls == 3 and p.stats.stops == 1


def test_same_family_in_a_wave_is_dropped_and_wave_eps_must_fit():
    box = ClockBox({1: 0.5, 2: 0.5, 3: 0.5})
    p = planner([wave_step(q(Q_FAN, 1), q(Q_FAN2, 2))], (2,))  # same family twice
    v = p.investigate(ALERT, box, LIMITS)
    assert box.waves == [1] and len(v.questions_log) == 1
    assert p.last_trace[0]["wave"][1]["error"] == "same_family_in_wave"
    lim = replace(LIMITS, max_llm_calls=1)
    box = ClockBox({1: 0.5, 2: 0.5})
    p = planner([wave_step(q(Q_PASS, 1, 2.0), q(Q_FAN, 2, 2.0))], (2,))
    v = p.investigate(ALERT, box, lim)
    assert sum(e.eps for e in v.questions_log) <= 2.0 + 1e-9  # both asks were 2.0 each: they cannot both go out
    assert box.waves in ([1], [2])


def state(k=1):
    return PlannerState(alert_summary={}, evidence_text="", partner_banks=[1, 2], eps_left=2.0, calls_left=5,
                        questions=[{"id": "B01", "program": Q_PASS}], wave_k=k)


def test_schema_accepts_lists_only_at_batched_steps():
    seq, wave = planner_schema(Mode.COMPOSE, state(1)), planner_schema(Mode.COMPOSE, state(2))
    assert "questions" not in seq["properties"] and "question_id" in seq["properties"]
    assert wave["properties"]["questions"]["type"] == "array" and wave["properties"]["questions"]["maxItems"] == 2
    assert set(wave["properties"]["action"]["enum"]) == {"ask", "stop"}
    item = wave["properties"]["questions"]["items"]["properties"]
    assert item["eps"]["enum"] == [0.25, 0.5, 1.0, 2.0] and item["bank"]["enum"] == [1, 2] and "question_id" in item
    sel = {"action": "select", "question_id": "B01", "program": None, "bank": 1, "eps": 0.5, "rule_id": None}
    ok = wave_step(sel, q(Q_FAN, 2))
    a = parse_action(ok, Mode.COMPOSE, state(2))
    assert a.action == "wave" and len(a.wave) == 2 and a.wave[0].question_id == "B01" and a.wave[1].program == Q_FAN
    with pytest.raises(PlannerParseError):  # a list at a sequential step is not valid
        parse_action(ok, Mode.COMPOSE, state(1))
    with pytest.raises(PlannerParseError):  # more than k
        parse_action(wave_step(q(Q_PASS, 1), q(Q_FAN, 2), q(Q_FAN2, 1)), Mode.COMPOSE, state(2))
    with pytest.raises(PlannerParseError):  # ask with no question
        parse_action(wave_step() .replace('"stop"', '"ask"'), Mode.COMPOSE, state(2))


def test_sequential_path_unchanged_when_option_off():
    box = ClockBox({1: 0.5, 2: 2.0})
    llm = MockLLM(script=[one_step(Q_PASS, 1), one_step(Q_FAN, 2), STOP])
    p = LLMPlanner(llm, mode="compose")
    v = p.investigate(ALERT, box, LIMITS)
    assert box.waves == [] and box.singles == 2 and box.clock == pytest.approx(2.5)  # one by one: the sum
    assert v.llm_calls == 3
    assert all("questions" not in r["json_schema"]["properties"] for r in llm.requests)


def test_bad_schedule_rejected():
    with pytest.raises(ValueError):
        LLMPlanner(MockLLM(), batch_schedule=(0, 1))
