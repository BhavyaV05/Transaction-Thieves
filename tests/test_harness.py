"""C16 harness v0: gate + accountant + clock in front of ask_bank; sessions; no way around them."""

import ast
import math
import random
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ftm.backtest.synth import make_world
from ftm.banks.partner import PartnerBank, Question
from ftm.compliance.gate import BudgetRefused, Gate, GateBlocked
from ftm.dsl.parser import canonical, gate_validator, parse
from ftm.guards import TimeGuardError
from ftm.harness import Harness, MockBank, SessionError, ToolUnavailable, make_toolbox
from ftm.harness.interface import Investigator, SessionToolBox, ToolBox
from ftm.privacy.accountant import PrivacyAccountant
from ftm.types import AlertContext, Decline, Limits, Reply, Timeout, Verdict

T0 = datetime(2022, 9, 5, 12, 0)
PROG = "n_counterparties(ALERT.receiver, dir=in, window=before(72))"


def alert(aid="A1", home="1"):
    return AlertContext(alert_id=aid, home_bank=home, as_of=T0.isoformat(), split="val", p0=0.1)


class Counting:
    """Wraps a bank and counts how often it was actually asked."""

    needs_binding = False

    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    def respond(self, question, as_of, purpose="live"):
        self.calls += 1
        return self.inner.respond(question, as_of, purpose)


class Fixed:
    needs_binding = False

    def __init__(self, response, wait):
        self.response, self.wait = response, wait

    def respond(self, question, as_of, purpose="live"):
        return self.response, self.wait


def harness(banks=None, acc=None, **kw):
    acc = acc or PrivacyAccountant(eps_per_alert=2.0)
    gate = Gate.from_config(validator=gate_validator, accountant=acc)
    banks = banks or {b: Counting(MockBank(b, 0, p_decline=0.0)) for b in (2, 3)}
    return Harness(banks, gate, acc, **kw), banks, acc


def session(h, a=None, limits=None):
    t = h.toolbox()
    t.begin(a or alert(), limits or Limits())
    return t


def test_a_session_is_needed_and_the_toolbox_satisfies_the_contracts():
    t = make_toolbox(seed=1)
    with pytest.raises(SessionError):
        t.ask_bank(2, PROG, 1.0)
    assert isinstance(t, ToolBox) and isinstance(t, SessionToolBox)


def test_a_valid_question_goes_through_and_is_logged():
    h, banks, acc = harness()
    t = session(h)
    r = t.ask_bank(2, PROG, 1.0)
    assert isinstance(r, Reply) and r.kind == "count" and banks[2].calls == 1
    (e,) = t.questions_log
    assert (e.step, e.bank, e.program, e.eps, e.gate_decision) == (1, "2", canonical(parse(PROG)), 1.0, "allowed")
    assert e.response == r and t.eps_spent == 1.0 and t.elapsed_days == pytest.approx(r.latency_days)
    assert acc.alert_spent["A1"] == 1.0


@pytest.mark.parametrize(
    "bank, program, eps",
    [
        (2, "nope(ALERT.receiver)", 1.0),  # does not parse
        (2, "n_tx(ALERT.receiver, dir=in, window=before(72), account=A1b2c3d4)", 1.0),  # names an account
        (2, "n_counterparties(A1b2c3d4, dir=in, window=before(72))", 1.0),  # alias as subject
        (2, "entity_accounts(ALERT.receiver)", 1.0),  # not askable
        (9, PROG, 1.0),  # not a partner bank
        (1, PROG, 1.0),  # the home bank cannot ask itself
        (2, PROG, 0.7),  # not an allowed eps level
        (2, PROG, 3.0),
        (2, PROG, math.inf),
        (2, PROG, -1.0),
        (2, PROG, 0.0),
    ],
)
def test_the_gate_blocks_and_the_bank_is_never_asked(bank, program, eps):
    h, banks, acc = harness()
    t = session(h)
    with pytest.raises(GateBlocked) as ex:
        t.ask_bank(bank, program, eps)
    assert isinstance(ex.value, PermissionError)
    assert t.questions_log[0].gate_decision == "blocked" and t.questions_log[0].response is None
    assert all(b.calls == 0 for b in banks.values()) and not acc.alert_spent and t.eps_spent == 0


def test_the_per_alert_budget_stops_the_third_question():
    h, banks, acc = harness()
    t = session(h)
    t.ask_bank(2, PROG, 1.0)
    t.ask_bank(3, PROG, 1.0)
    with pytest.raises(BudgetRefused) as ex:
        t.ask_bank(2, "n_tx(ALERT.receiver, dir=in, window=before(72))", 0.25)
    assert "budget" in type(ex.value).__name__.lower()
    assert t.questions_log[-1].gate_decision == "over_budget"
    assert sum(b.calls for b in banks.values()) == 2 and acc.alert_spent["A1"] == pytest.approx(2.0)


def test_the_run_limit_can_be_tighter_than_the_accountants():
    h, banks, _ = harness()
    t = session(h, limits=Limits(eps_per_alert=1.0))
    t.ask_bank(2, PROG, 1.0)
    with pytest.raises(BudgetRefused):
        t.ask_bank(3, PROG, 0.25)
    assert sum(b.calls for b in banks.values()) == 1


def test_declines_and_timeouts_release_nothing_so_they_cost_nothing():
    h, _, acc = harness({2: Fixed(Decline(2, "declined"), 0.3), 3: Fixed(Timeout(3, 5.0), 5.0)})
    t = session(h, limits=Limits(max_sim_time_days=50))
    for _ in range(5):
        assert isinstance(t.ask_bank(2, PROG, 2.0), Decline)
    assert isinstance(t.ask_bank(3, PROG, 2.0), Timeout)
    assert t.eps_spent == 0 and not acc.alert_spent and t.elapsed_days == pytest.approx(5 * 0.3 + 5.0)


def test_time_runs_out():
    slow = Fixed(Reply(2, "count", 3, 1.0, 3.0, None), 3.0)
    h, _, _ = harness({2: slow, 3: slow})
    t = session(h, limits=Limits(max_sim_time_days=4.0))
    assert isinstance(t.ask_bank(2, PROG, 0.25), Reply)  # arrives at day 3
    r = t.ask_bank(3, "returns_home(ALERT.receiver, window=after(24))", 0.25)  # would arrive at day 6: cut at day 4
    assert isinstance(r, Timeout) and r.after_days == pytest.approx(1.0) and t.elapsed_days == pytest.approx(4.0)
    with pytest.raises(BudgetRefused):
        t.ask_bank(2, PROG, 0.25)


def test_native_mode_has_no_ask_bank():
    h, banks, _ = harness(mode="native")
    t = session(h)
    with pytest.raises(ToolUnavailable):
        t.ask_bank(2, PROG, 1.0)
    assert banks[2].calls == 0


def test_local_graph_obeys_the_time_guard_and_lookup_rules_goes_through_the_gate():
    seen = []
    h, _, _ = harness(graph_fn=lambda q, as_of: seen.append(as_of) or [{"edge": 1}])
    t = session(h)
    assert t.local_graph("q", (T0 - timedelta(hours=1)).isoformat()) == [{"edge": 1}] and seen
    with pytest.raises(TimeGuardError):
        t.local_graph("q", (T0 + timedelta(seconds=1)).isoformat())
    assert isinstance(t.lookup_rules("structuring cash deposits"), list)
    assert t.search_library({"shape": 1}) == []


def test_sessions_are_independent_and_the_toolbox_log_is_the_one_that_counts():
    h, _, acc = harness()
    tools = h.toolbox()

    class Liar:
        def investigate(self, a, tools, limits):
            tools.ask_bank(2, PROG, 1.0)
            return Verdict(a.alert_id, 0.5, questions_log=[], llm_calls=3)  # claims it asked nothing

    wrapped = tools.wrap(Liar())
    v1 = wrapped.investigate(alert("A1"), tools, Limits())
    v2 = wrapped.investigate(alert("A2"), tools, Limits())
    assert [e.program for e in v1.questions_log] == [canonical(parse(PROG))] and v1.llm_calls == 3
    assert len(v2.questions_log) == 1 and v2.questions_log[0].step == 1  # a fresh session
    assert acc.alert_spent["A1"] == acc.alert_spent["A2"] == 1.0
    with pytest.raises(SessionError):
        tools.ask_bank(2, PROG, 1.0)  # closed after the run
    assert isinstance(wrapped, Investigator)


def test_the_real_bank_answers_with_its_own_noise_and_the_clock_moves_the_ask_time():
    w = make_world(60, seed=5)
    acc = PrivacyAccountant()
    gate = Gate.from_config(validator=gate_validator, accountant=acc)
    h = Harness(w.banks, gate, acc, {a.alert_id: a.binding for a in w.alerts})
    a = w.alerts[0]
    ctx = replace(alert(a.alert_id, str(a.binding.home_bank)), as_of=a.binding.t0.isoformat())
    t = session(h, ctx)
    bank_id = a.binding.receiver_bank
    r = t.ask_bank(bank_id, PROG, 2.0)
    direct = w.banks[bank_id].handle(Question(canonical(parse(PROG)), 2.0, a.binding), a.binding.t0)
    assert r == direct  # same bank, same question, same ask time, same reply
    assert acc.account_spent[(bank_id, a.binding.receiver)] == 2.0 or isinstance(r, (Decline, Timeout))


def test_the_per_account_cap_holds_across_alerts_through_the_harness():
    w = make_world(40, seed=6)
    acc = PrivacyAccountant(eps_per_alert=2.0, per_account_cap=3.0)
    gate = Gate.from_config(validator=gate_validator, accountant=acc)
    quiet = {b: PartnerBank(b, bk._ledger, replace(bk.cfg, p_decline=0.0, p_timeout=0.0), 0) for b, bk in w.banks.items()}
    a1, a2, a3 = w.alerts[0], w.alerts[1], w.alerts[2]
    same = {aid: replace(a.binding, alert_id=aid, receiver=a1.binding.receiver, receiver_bank=a1.binding.receiver_bank)
            for aid, a in zip("abc", (a1, a2, a3), strict=True)}  # fmt: skip
    h = Harness(quiet, gate, acc, same)
    bank_id = a1.binding.receiver_bank
    for aid, eps, ok in (("a", 2.0, True), ("b", 1.0, True), ("c", 0.25, False)):
        t = session(h, replace(alert(aid), as_of=a1.binding.t0.isoformat()))
        if ok:
            assert isinstance(t.ask_bank(bank_id, PROG, eps), Reply)
        else:
            with pytest.raises(BudgetRefused):
                t.ask_bank(bank_id, PROG, eps)
    assert acc.account_spent[(bank_id, a1.binding.receiver)] == pytest.approx(3.0)
    assert acc.report()["refusals"]["account"] >= 1


def test_a_bank_with_a_ledger_is_never_asked_without_real_ids():
    w = make_world(30, seed=8)
    acc = PrivacyAccountant()
    h = Harness(w.banks, Gate.from_config(validator=gate_validator, accountant=acc), acc, {})  # no bindings
    t = session(h)
    with pytest.raises(SessionError):
        t.ask_bank(2, PROG, 1.0)
    assert not acc.alert_spent


def test_nothing_public_hands_out_a_bank_the_gate_or_the_accountant():
    h, _, _ = harness()
    t = session(h)
    forbidden = (PartnerBank, MockBank, Counting, Gate, PrivacyAccountant, Harness)
    for name in dir(t):
        if name.startswith("_"):
            continue
        val = getattr(t, name)
        assert not isinstance(val, forbidden), name
    assert {n for n in dir(t) if not n.startswith("_")} == {
        "begin", "end", "wrap", "ask_bank", "ask_banks", "local_graph", "lookup_rules", "search_library",
        "questions_log", "eps_spent", "elapsed_days", "last_elapsed_days",
    }  # fmt: skip


def test_fuzzing_cannot_exceed_any_budget():
    rng = random.Random(0)
    acc = PrivacyAccountant(eps_per_alert=2.0, per_account_cap=None)
    h, banks, _ = harness(acc=acc)
    progs = [PROG, "returns_home(ALERT.receiver, window=after(24))", "time_to_forward(ALERT.receiver)", "garbage(", "",
             "n_tx(ALERT.receiver, dir=in, window=before(72), account=A1b2c3d4)"]  # fmt: skip
    answered = 0
    for i in range(3):
        t = session(h, alert(f"F{i}"))
        for _ in range(120):
            try:
                r = t.ask_bank(rng.choice([1, 2, 3, 4, "x"]), rng.choice(progs), rng.choice([0.0, 0.25, 0.5, 1.0, 2.0, 5.0]))
                answered += isinstance(r, Reply)
            except (GateBlocked, BudgetRefused):
                pass
            except (ValueError, TypeError):
                pass  # bank "x": rejected before anything is spent
        assert t.eps_spent <= 2.0 + 1e-9 and acc.alert_spent.get(f"F{i}", 0.0) <= 2.0 + 1e-9
    assert answered >= 3 and sum(b.calls for b in banks.values()) >= answered


def test_mock_mode_is_deterministic_and_answers_every_kind():
    def run():
        t = make_toolbox(seed=7)
        t.begin(alert(home="1"), Limits(max_sim_time_days=50))
        return [t.ask_bank(5, p, 0.25) for p in (PROG, "returns_home(ALERT.receiver, window=after(24))",
                                                 "time_to_forward(ALERT.receiver)",
                                                 "overlap(ALERT.receiver, dir=out, window=after(72), among=ALERT.payers)")]  # fmt: skip

    a, b = run(), run()
    assert a == b and {r.kind for r in a if isinstance(r, Reply)} <= {"count", "bool", "kary", "set_count"}


def test_systems_outside_the_harness_cannot_import_banks_or_the_accountant():
    root = Path(__file__).resolve().parents[1] / "ftm"
    banned = ("ftm.banks", "ftm.privacy", "ftm.backtest.engine", "ftm.backtest.world", "ftm.backtest.synth")
    offenders = []
    for pkg in ("external", "planners", "llm", "agent", "library", "fastmodel", "baselines"):
        for f in (root / pkg).rglob("*.py") if (root / pkg).exists() else []:
            for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
                mods = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""] if isinstance(n, ast.ImportFrom) else []
                offenders += [f"{f.name}: {m}" for m in mods if m.startswith(banned)]
    assert not offenders, offenders
