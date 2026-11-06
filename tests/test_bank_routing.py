"""Bank routing (audit finding 1), prompt/menu changes (7, 4), library (3), distillation (6). Synthetic data only."""

import json
from datetime import datetime

from ftm.backtest.engine import asked_bank as engine_asked_bank
from ftm.banks.binding import AlertBinding, alert_context
from ftm.dsl.parser import parse
from ftm.harness.interface import ScriptedToolBox
from ftm.llm.backtest_table import accepted_menu, bank_routable
from ftm.llm.investigation import LREntry, LRTable, NaiveLRFusion
from ftm.llm.mock import MockLLM
from ftm.llm.routing import NO_BANK, asked_bank, expected_bank
from ftm.planners.llm import LLMPlanner
from ftm.planners.llm_dsl import V0Dsl
from ftm.planners.llm_prompt import AnswerSoFar, Mode, PlannerState, build_messages
from ftm.types import Limits, QuestionLogEntry, Reply

DSL = V0Dsl()
T0 = datetime(2022, 9, 5, 12)
PROGRAMS = ["dormant(ALERT.receiver, days=7)", "new_account(ALERT.sender, days=2)",
            "at_least(n_tx(ALERT.sender, dir=in, window=before(24)), k=2)",
            "overlap(ALERT.receiver, dir=out, window=after(72), among=ALERT.payers)",
            "set_contacts(ALERT.payers, dir=out, window=before(72))"]  # fmt: skip


def canon(q):
    return DSL.validate(q).canonical


def ctx(sender_bank, receiver_bank, home=70, p0=0.2):
    b = AlertBinding("V1", home, T0, 1, 2, 1.0, "USD", sender_bank=sender_bank, receiver_bank=receiver_bank)
    c = alert_context(b, p0)
    c.summary.update(sender="A0000002", receiver="A0000001", sender_bank=sender_bank, receiver_bank=receiver_bank)
    return b, c


def test_routing_matches_the_backtest_rule():
    for sb, rb in [(70, 220), (220, 70), (220, 1588), (70, 70), (None, 220)]:
        b, c = ctx(sb, rb)
        for p in PROGRAMS:
            want = engine_asked_bank(parse(canon(p)), b)
            got = asked_bank(canon(p), c.summary, c.home_bank)
            assert (None if got is None else int(got)) == want, (sb, rb, p)
            eb = expected_bank(canon(p), c.summary, c.home_bank)
            assert eb == (NO_BANK if want is None else str(want))


def test_expected_bank_unknown_without_bank_fields():
    assert expected_bank(canon(PROGRAMS[0]), {"sender": "x"}, "70") is None


def lr_table():
    return LRTable([LREntry(canon(PROGRAMS[0]), 1.0, True, {"true": 0.5, "false": 2.0}, "dormant"),
                    LREntry(canon(PROGRAMS[1]), 1.0, True, {"true": 0.5, "false": 2.0}, "new_account")])  # fmt: skip


def reply(v, bank="220"):
    return Reply(bank=bank, kind="bool", value=v, eps_spent=1.0, latency_days=0.5)


def step(program=PROGRAMS[0], bank=220, action="compose", qid=None):
    return json.dumps({"hypothesis": {"typology": "fan_in", "rationale": "r"}, "action": action, "question_id": qid,
                       "program": program if action == "compose" else None, "bank": bank if action != "stop" else None,
                       "eps": 1.0 if action != "stop" else 0.0, "rule_id": None})  # fmt: skip


STOP = step(action="stop")


def alert_ctx(sender_bank=220, receiver_bank=70):
    _, c = ctx(sender_bank, receiver_bank)
    c.summary["partner_banks"] = [220, 1588]
    c.summary["evidence"] = "row"
    return c


def test_fusion_ignores_answers_from_the_wrong_bank():
    f = NaiveLRFusion(lr_table())
    right = QuestionLogEntry(1, "220", canon(PROGRAMS[0]), 1.0, reply(True), expected_bank="70")
    wrong = QuestionLogEntry(1, "1588", canon(PROGRAMS[0]), 1.0, reply(True, "1588"), expected_bank="70")
    home = QuestionLogEntry(1, "220", canon(PROGRAMS[0]), 1.0, reply(True), expected_bank=NO_BANK)
    legacy = QuestionLogEntry(1, "220", canon(PROGRAMS[0]), 1.0, reply(True))
    assert f.uses(legacy) and not f.uses(wrong) and not f.uses(home) and not f.uses(right)  # right: expected 70, asked 220
    right.expected_bank = "220"
    assert f.uses(right) and f.posterior(0.5, [wrong]) == 0.5 and f.posterior(0.5, [right]) < 0.5


def test_compose_bank_is_overridden_and_home_subject_is_refused():
    # receiver at the home bank: ALERT.receiver questions have no partner; ALERT.sender is held by bank 220
    llm = MockLLM(script=[step(PROGRAMS[0], bank=1588), step(PROGRAMS[1], bank=1588), STOP])
    tools = ScriptedToolBox([reply(True)])
    p = LLMPlanner(llm, fusion=NaiveLRFusion(lr_table()))
    v = p.investigate(alert_ctx(220, 70), tools, Limits())
    assert [e.bank for e in v.questions_log] == ["220"] and v.questions_log[0].expected_bank == "220"
    assert tools.calls[0][0] == 220
    assert "home bank" in llm.requests[1]["messages"][1]["content"].split("## Your previous output was rejected")[1]
    tr = p.last_trace
    assert tr[0]["error"] == "no_partner_holds_subject" and tr[1]["bank_requested"] == 1588


def test_select_fills_the_routed_bank_and_menu_drops_home_subjects():
    menu = accepted_menu(lr_table())
    a = alert_ctx(220, 70)
    assert [bank_routable(q, a) for q in menu] == [False, True]  # sorted by family: dormant(receiver) then new_account(sender)
    llm = MockLLM(script=[step(action="select", qid="B02", bank=1588), STOP])
    tools = ScriptedToolBox([reply(False)])
    p = LLMPlanner(llm, questions=menu, question_filter=bank_routable)
    v = p.investigate(a, tools, Limits())
    prompt = llm.requests[0]["messages"][1]["content"]
    assert "B01" not in prompt and "B02" in prompt and "ask bank 220" in prompt
    assert tools.calls[0][0] == 220 and v.questions_log[0].bank == "220"


def test_menu_shows_the_full_bin_map_and_marginal_note():
    q = LRTable([LREntry(canon("n_tx(ALERT.receiver, dir=in, window=before(24))"), 2.0, True,
                         {"b0": 0.32, "b1": 0.9, "b2": 1.84}, "n_tx", (3.0, 10.0))])  # fmt: skip
    m = accepted_menu(q)
    assert m[0]["lr_text"] == "<3 x0.32; 3 to <10 x0.90; >=10 x1.84" and m[0]["max_shift"] > 1.0
    s = PlannerState({"a": 1}, "ev", [220], 2.0, 10, questions=[{**m[0], "bank": 220}], threshold=0.4)
    user = build_messages(Mode.COMPOSE, s, DSL, 6000)[1]["content"]
    assert "<3 x0.32; 3 to <10 x0.90; >=10 x1.84" in user and "marginal" in user and "ask bank 220" in user
    assert "decision threshold: escalate at probability >= 0.4" in user


def test_no_backtest_prompt_has_p0_and_no_acceptance_tag():
    ans = [AnswerSoFar(1, 220, "dormant(ALERT.receiver, days=7)", 1.0, "reply", True, used_in_posterior=False)]
    s = PlannerState({"a": 1}, "ev", [220], 2.0, 10, answers=ans, posterior=0.31, p0=0.2)
    nb = build_messages(Mode.NO_BACKTEST, s, DSL, 6000)[1]["content"]
    assert "no back-tested LR" not in nb and "0.310" not in nb and "0.200" in nb
    assert "no back-tested LR" in build_messages(Mode.COMPOSE, s, DSL, 6000)[1]["content"]


# -- library ----------------------------------------------------------------------------------------------
def test_library_search_has_a_distance_cutoff_and_prefers_exact_typology():
    from ftm.library.shape import SituationShape
    from ftm.library.store import Strategy, StrategyLibrary

    def strat(i, typ, in_deg):
        sh = SituationShape(typ, in_deg, "1", "2", "1k-10k flat", "<6h")
        return Strategy(f"S{i}", "success", f"t{i}", "q", "f", 1.0, 30, 0.1, 0.01, True, sh.as_dict())

    lib = StrategyLibrary()
    far = strat(0, "any", "8+")
    lib.add(far)
    lib.add(strat(1, "any", "2-3"))
    lib.add(strat(2, "fan_in", "2-3"))
    q = SituationShape("fan_in", "1", "1", "2", "1k-10k flat", "<6h")
    hits = lib.search(q)
    assert [h.strategy_id for h in hits] == ["S2", "S1"]  # typology match first; the far shape is cut
    assert lib.search(SituationShape("fan_in", "0", "8+", "5+", "<100 flat", ">3d")) == []
    assert len(lib.search(q, max_dist=100)) == 3


def test_library_family_unwraps_at_least_and_rejected_questions_get_no_note():
    from ftm.library.build import _family

    assert _family(canon("at_least(n_tx(ALERT.sender, dir=in, window=before(24)), k=2)")).startswith("at_least:n_tx")
    assert _family("garbage") == "garbage"


def test_account_keyed_library_is_not_empty_for_single_alerts():
    from ftm.library.build import build_library
    from ftm.library.shape import SituationShape

    sh = SituationShape("any", "1", "1", "2", "1k-10k flat", "<6h").as_dict()
    recs = [{"alert_id": f"A{i}", "split": "train", "shape": sh, "receiver": f"R{i}", "question": "q", "family": "f",
             "eps": 1.0, "accepted": True, "gain": 0.1} for i in range(3)]  # fmt: skip
    assert len(build_library(recs, keying="account")) == 3 and len(build_library(recs, keying="shape")) == 0
    nacc = [{**r, "accepted": False} for r in recs]
    assert len(build_library(nacc, keying="account")) == 0  # no tautological "rejected" notes
