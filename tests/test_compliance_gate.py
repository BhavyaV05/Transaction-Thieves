"""T4: compliance gate v0 (C6). Done-when (tracks.md 6.3): 100% of a hand-written disallowed suite is blocked."""

import pytest

from ftm.compliance import gate as g
from ftm.harness.interface import ScriptedToolBox
from ftm.types import Reply

PARTNERS = ["1588", "70", "12"]


def _validator(program):
    """Stand-in for T2's DSL: programs start with a kind tag, e.g. 'count: ALERT.receiver.payers(24h)'."""
    kind, _, body = program.partition(":")
    if kind not in {"bool", "count", "kary", "set_count"} or "ALERT." not in body:
        return False, None, "unparseable"
    return True, kind, None


class Acct:
    def __init__(self, cap=2.0):
        self.cap, self.spent = cap, {}

    def can_spend(self, alert_id, bank, eps):
        return self.spent.get(alert_id, 0.0) + eps <= self.cap + 1e-9

    def spend(self, alert_id, bank, eps):
        self.spent[alert_id] = self.spent.get(alert_id, 0.0) + eps


@pytest.fixture
def gate():
    return g.Gate.from_config(validator=_validator, accountant=Acct())


def ask(gate, **kw):
    base = dict(alert_id="V1", home_bank="7", bank="1588", program="bool: ALERT.receiver.passes_through(0.8, 24h)",
                eps=1.0, rule_id=None, partners=PARTNERS)  # fmt: skip
    base.update(kw)
    return gate.check("ask_bank", **base)


def test_valid_question_allowed_with_legal_basis(gate):
    d = ask(gate)
    assert d.allowed and d.rule_id == "US-314B-SHARE"


DISALLOWED = [
    ("unknown action", lambda gt: gt.check("delete_account")),
    ("send funds", lambda gt: gt.check("freeze_account", bank="1588")),
    ("raw records", lambda gt: gt.check("request_records", bank="1588")),
    ("share records", lambda gt: gt.check("share_records", bank="1588")),
    ("ask own bank", lambda gt: ask(gt, bank="7")),
    ("not a partner", lambda gt: ask(gt, bank="999")),
    ("eps zero", lambda gt: ask(gt, eps=0.0)),
    ("eps negative", lambda gt: ask(gt, eps=-1.0)),
    ("eps off-grid", lambda gt: ask(gt, eps=0.7)),
    ("eps too big", lambda gt: ask(gt, eps=5.0)),
    ("missing program", lambda gt: ask(gt, program=None)),
    ("unparseable program", lambda gt: ask(gt, program="SELECT * FROM transactions")),
    ("alias literal", lambda gt: ask(gt, program="bool: ALERT.receiver.pays(A00f3a1c)")),
    ("raw account number", lambda gt: ask(gt, program="bool: ALERT.receiver.pays(8000EBD30)")),
    ("date literal", lambda gt: ask(gt, program="count: ALERT.receiver.payers(2022-09-01)")),
    ("unknown rule cited", lambda gt: ask(gt, rule_id="US-314B-MADEUP")),
    ("rule from wrong jurisdiction", lambda gt: ask(gt, rule_id="SG-COSMIC-REQUEST")),
    ("rule that permits nothing", lambda gt: ask(gt, rule_id="US-314B-NOTICE")),
    ("SAR sent to another bank", lambda gt: gt.check("draft_sar", home_bank="7", bank="1588")),
    ("escalation to another bank", lambda gt: gt.check("escalate_to_human", home_bank="7", bank="70")),
]


@pytest.mark.parametrize("name,call", DISALLOWED, ids=[n for n, _ in DISALLOWED])
def test_disallowed_suite_is_blocked(gate, name, call):
    assert call(gate).decision == "blocked", name


def test_whole_suite_blocked_rate_is_one(gate):
    assert sum(call(gate).decision == "blocked" for _, call in DISALLOWED) == len(DISALLOWED)


def test_citation_required_mode():
    gt = g.Gate.from_config(validator=_validator, require_citation=True)
    assert ask(gt).decision == "blocked"
    assert ask(gt, rule_id="US-314B-SHARE").allowed


def test_cosmic_jurisdiction():
    gt = g.Gate.from_config(validator=_validator, jurisdiction_by_bank={"70": "SG"})
    assert ask(gt, bank="70").rule_id == "SG-COSMIC-REQUEST"
    assert ask(gt, bank="70", rule_id="US-314B-SHARE").decision == "blocked"


def test_raw_share_only_in_baseline_arm():
    assert g.Gate.from_config(allow_raw_share=True).check("request_records", bank="1588").allowed


def test_internal_actions_allowed():
    gt = g.Gate.from_config()
    assert gt.check("draft_sar", home_bank="7", bank="7").allowed
    assert gt.check("escalate_to_human", home_bank="7").allowed
    assert gt.check("stop").allowed


def test_gated_toolbox_raises_and_charges_once(gate):
    inner = ScriptedToolBox([Reply("1588", "bool", True, 1.0, 0.5), Reply("1588", "bool", False, 1.0, 0.5)])
    tb = g.GatedToolBox(inner, gate, alert_id="V1", home_bank="7", partners=PARTNERS)
    q = "bool: ALERT.receiver.passes_through(0.8, 24h)"
    assert tb.ask_bank("1588", q, 1.0).value is True
    with pytest.raises(g.GateBlocked):
        tb.ask_bank("1588", "bool: ALERT.receiver.pays(A00f3a1c)", 1.0)
    with pytest.raises(PermissionError):  # T3's Investigation treats PermissionError as a gate refusal
        tb.ask_bank("7", q, 1.0)
    assert tb.ask_bank("70", q, 1.0).value is False  # second eps 1.0: budget 2.0 now used up
    with pytest.raises(g.BudgetRefused):
        tb.ask_bank("12", q, 0.25)
    assert len(inner.calls) == 2  # refused requests never reach the bank
    assert [d.decision for d in tb.decisions] == ["allowed", "blocked", "blocked", "allowed", "over_budget"]


def test_lookup_rules_keyword(gate):
    top = gate.lookup_rules("314(b) sharing money laundering")
    assert top[0]["rule_id"] == "US-314B-SHARE"
    assert all(r["jurisdiction"] in ("SG", "ANY") for r in gate.lookup_rules("request", jurisdiction="SG"))


TYPOLOGIES = {"fan_in", "fan_out", "cycle", "scatter_gather", "gather_scatter", "bipartite", "stack", "random"}


def test_red_flag_corpus_integrity():
    flags = g.load_red_flags()
    assert len(flags) >= 40
    assert {f.jurisdiction for f in flags.values()} == {"US", "SG"}
    for f in flags.values():
        assert f.flag_id.startswith(f"RF-{f.jurisdiction}-")
        assert set(f.typologies) <= TYPOLOGIES, f.flag_id
        assert f.observable in {"yes", "partial", "no"}
        assert f.text and f.source and f.signal
    assert sum(f.observable == "yes" for f in flags.values()) >= 20  # enough to seed the expert question list


def test_lookup_finds_red_flags_but_they_are_not_citations(gate):
    hits = gate.lookup_rules("consolidated several banks onward", jurisdiction="SG")
    assert hits[0]["rule_id"] == "RF-SG-13" and hits[0]["kind"] == "red_flag"
    assert ask(gate, rule_id="RF-US-14").decision == "blocked"  # a red flag is not a legal basis
    assert any(h["rule_id"] == "RF-US-14" for h in gate.lookup_rules("moved back to the first bank cycle"))


def test_gate_without_accountant_never_charges():
    # the harness set-up: gate checks only, the harness charges the accountant itself (one charge)
    acct = Acct()
    gt = g.Gate.from_config(validator=_validator, accountant=None)
    tb = g.GatedToolBox(ScriptedToolBox([Reply("1588", "bool", True, 1.0, 0.5)]), gt, "V1", "7", PARTNERS)
    tb.ask_bank("1588", "bool: ALERT.receiver.passes_through(0.8, 24h)", 1.0)
    assert acct.spent == {}


def test_gate_accepts_t2_dsl_and_accountant():
    from ftm.dsl.parser import gate_validator
    from ftm.privacy.accountant import PrivacyAccountant

    gt = g.Gate.from_config(validator=gate_validator, accountant=PrivacyAccountant(2.0, 8.0, None))
    d = gt.check("ask_bank", alert_id="V9", home_bank="7", bank="1588", eps=1.0, partners=PARTNERS,
                 program="passed_on(ALERT.receiver, window=after(24), min_frac=0.8)")  # fmt: skip
    assert d.allowed, d.reason
    assert gt.check("ask_bank", alert_id="V9", home_bank="7", bank="1588", eps=1.0, partners=PARTNERS,
                    program="passed_on(A00f3a1c, window=after(24), min_frac=0.8)").decision == "blocked"  # fmt: skip
