"""LLMCompiler waves are charged the slowest answer on the toolbox clock, not the sum (RQ5 / E13)."""

import pytest

from ftm.external.llmcompiler import LlmCompiler
from ftm.harness import make_toolbox
from ftm.llm.investigation import DuplicateQuestion, Investigation
from ftm.llm.mock import MockLLM
from ftm.planners.llm_dsl import V0Dsl
from ftm.types import AlertContext, Limits, Reply

DSL = V0Dsl()
Q1 = "passed_on(ALERT.receiver, window=after(24), min_frac=0.8)"
Q2 = "at_least(n_counterparties(ALERT.receiver, dir=in, window=before(48)), k=5)"
ALERT = AlertContext("W1", "70", "2022-09-05T12:00:00", "val", 0.2,
                     {"receiver": "R", "evidence": "rows", "partner_banks": [1, 2, 3]})  # fmt: skip


class GateBlocked(Exception):
    pass


def reply(bank, days, eps=0.5):
    return Reply(bank=str(bank), kind="bool", value=True, eps_spent=eps, latency_days=days)


class WaveBox:
    """Fake toolbox with a clock and the ask_banks extension (a wave costs its slowest answer)."""

    def __init__(self, answers, blocked=()):
        self.answers, self.blocked, self.clock, self.wave_calls = dict(answers), set(blocked), 0.0, 0

    def lookup_rules(self, q):
        return []

    def ask_banks(self, requests):
        self.wave_calls += 1
        out, waits = [], [0.0]
        for bank, _program, _eps in requests:
            if bank in self.blocked:
                out.append(GateBlocked("no"))
            else:
                r = self.answers[bank]
                out.append(r)
                waits.append(r.latency_days)
        self.clock += max(waits)
        return out


class PlainBox:
    """No ask_banks: one ask_bank at a time."""

    def __init__(self, answers, blocked=()):
        self.answers, self.blocked, self.calls = dict(answers), set(blocked), 0

    def ask_bank(self, bank, program, eps):
        self.calls += 1
        if bank in self.blocked:
            raise GateBlocked("no")
        return self.answers[bank]


def wave(s, *reqs):
    return s.ask_wave([(b, DSL.validate(q), e, i + 1) for i, (b, q, e) in enumerate(reqs)])


def test_wave_costs_max_on_toolbox_clock_and_sim_days():
    box = WaveBox({1: reply(1, 0.5), 2: reply(2, 2.0), 3: reply(3, 1.0)})
    s = Investigation(ALERT, box, Limits())
    out = wave(s, (1, Q1, 0.5), (2, Q1, 0.5), (3, Q1, 0.5))
    assert box.wave_calls == 1 and box.clock == pytest.approx(2.0) and s.sim_days == pytest.approx(2.0)
    assert [e.gate_decision for e in out] == ["allowed"] * 3 and len(s.log) == 3
    assert s.eps_left == pytest.approx(Limits().eps_per_alert - 1.5)


def test_refusal_in_wave_keeps_the_others():
    box = WaveBox({1: reply(1, 0.5), 3: reply(3, 1.0)}, blocked={2})
    s = Investigation(ALERT, box, Limits())
    out = wave(s, (1, Q1, 0.5), (2, Q1, 0.5), (3, Q1, 0.5))
    assert [e.gate_decision for e in out] == ["allowed", "blocked", "allowed"]
    assert s.last_refusal == "no" and s.sim_days == pytest.approx(1.0) and len(s.log) == 3
    canon = DSL.validate(Q1).canonical
    assert not s.already_asked(canon, 2) and s.already_asked(canon, 1)


def test_wave_without_ask_banks_still_charges_max():
    box = PlainBox({1: reply(1, 0.5), 2: reply(2, 2.0), 3: reply(3, 1.0)}, blocked={3})
    s = Investigation(ALERT, box, Limits())
    out = wave(s, (1, Q1, 0.5), (2, Q1, 0.5), (3, Q1, 0.5))
    assert box.calls == 3 and s.sim_days == pytest.approx(2.0)
    assert [e.gate_decision for e in out] == ["allowed", "allowed", "blocked"]


def test_wave_duplicates_and_over_budget():
    box = WaveBox({1: reply(1, 1.0, 0.5), 2: reply(2, 1.0, 0.5), 3: reply(3, 1.0, 0.5)})
    s = Investigation(ALERT, box, Limits(eps_per_alert=1.0))
    s.ask_wave([(1, DSL.validate(Q1), 0.5, 1)])  # earlier question
    out = wave(s, (1, Q1, 0.5), (2, Q2, 0.5), (2, Q2, 0.5), (3, Q2, 0.5))
    assert isinstance(out[0], DuplicateQuestion) and isinstance(out[2], DuplicateQuestion)
    assert out[1].gate_decision == "allowed" and out[3].gate_decision == "over_budget"  # 0.5 + 0.5 already used
    assert len(s.log) == 3 and s.eps_left == pytest.approx(0.0)


def test_parallel_block_still_works_with_waves_and_does_not_nest():
    box = WaveBox({1: reply(1, 0.5), 2: reply(2, 2.0)})
    s = Investigation(ALERT, box, Limits())
    with s.parallel():
        wave(s, (1, Q1, 0.5), (2, Q1, 0.5))
    assert s.sim_days == pytest.approx(2.0)
    with pytest.raises(RuntimeError), s.parallel(), s.parallel():
        pass


def test_llmcompiler_end_to_end_on_harness_mock():
    plan = (f"1. ask_bank[1 | 0.5 | {Q1}]\n2. ask_bank[2 | 0.5 | {Q1}]\n3. ask_bank[3 | 0.5 | {Q1}]\n"
            f"4. ask_bank[1 | 0.5 | {Q2}]\n5. join()")  # one wave: tasks 1-4, no dependencies
    tb = make_toolbox(seed=1, mode="mock", n_banks=5)
    llm = MockLLM(script=[plan, "Action: finish[0.5]"])
    tb.begin(ALERT, Limits())
    v = LlmCompiler(llm).investigate(ALERT, tb, Limits())
    log = tb.end()
    answered = [e for e in log if hasattr(e.response, "latency_days")]
    assert len(answered) == 4 and v.probability == 0.5
    lat = [e.response.latency_days for e in answered]
    assert max(lat) < sum(lat)
    assert tb.last_elapsed_days == pytest.approx(max(lat))
