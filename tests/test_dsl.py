"""C4 question language: parse, type check, canonical form, sensitivity, size, expert list."""

import pytest

from ftm.dsl.expert import expert_programs, load_specs
from ftm.dsl.parser import (
    arity,
    canonical,
    check,
    family,
    parse,
    reference_text,
    sensitivity,
    size,
)
from ftm.dsl.spec import PRIMS

EXPERT = expert_programs()


def test_expert_list_parses_and_matches_specs():
    specs = [f for f in load_specs() if f.get("status") != "needs_privacy_review"]
    assert len(EXPERT) == sum(len(f["variants"]) for f in specs) >= 50
    for p in EXPERT:
        c = check(p["program"])
        assert c.ok, (p["program"], c.error)
        assert c.kind == p["kind"], p["program"]


def test_expert_sensitivity_matches_specs_except_chain_length():
    # The spec yaml says its sensitivities are drafts and the DSL's function is authoritative. A chain can grow by
    # one hop per added transfer that joins two chains, so chain_length's sensitivity is its max_hops cap.
    for p in EXPERT:
        s = check(p["program"]).sensitivity
        if p["family"] == "internal_chain":
            assert s == 3
        else:
            assert s == p["sensitivity"], p["program"]


def test_expert_programs_are_distinct_and_canonical_is_stable():
    canon = [check(p["program"]).canonical for p in EXPERT]
    assert len(set(canon)) == len(canon)
    for c in canon:
        assert canonical(parse(c)) == c


@pytest.mark.parametrize("subject", ["receiver", "sender"])
def test_expert_list_works_for_both_alert_directions(subject):
    assert all(check(p["program"]).ok for p in expert_programs(subject))


def test_every_primitive_example_parses():
    for name, p in PRIMS.items():
        assert check(p.example, allow_restricted=True).ok, name


def test_canonical_fills_nothing_and_orders_everything():
    a = parse("n_tx( ALERT.receiver ,window=before(72), dir=in, min_amount=0, max_amount=3000.0 )")
    b = parse("n_tx(ALERT.receiver, dir=in, window=before(72), max_amount=3000)")
    assert a == b
    assert canonical(a) == "n_tx(ALERT.receiver, dir=in, window=before(72), max_amount=3000)"


def test_bank_lists_are_sorted_and_deduplicated():
    n = parse("top_outflow_bank(ALERT.sender, banks=[220, 70, 70, 1588], window=after(24))")
    assert canonical(n) == "top_outflow_bank(ALERT.sender, banks=[70, 220, 1588], window=after(24))"
    assert arity(n) == 4


def test_kinds_arity_family_size():
    n = parse("at_least(n_counterparties(ALERT.receiver, dir=in, window=before(72)), k=5)")
    assert (n.prim.kind, arity(n), family(n)) == ("bool", None, "at_least:n_counterparties:in")
    assert size(n) == 1 + 1 + (1 + 2)  # at_least + k, wrapped call + dir + window
    assert arity(parse("time_to_forward(ALERT.receiver)")) == 4


def test_chain_length_sensitivity_follows_its_cap():
    assert sensitivity(parse("chain_length(ALERT.receiver, window=after(24), max_hops=5)")) == 5


@pytest.mark.parametrize(
    "program, fragment",
    [
        ("", "empty"),
        ("n_counterparties(", "syntax"),
        ("nope(ALERT.receiver)", "unknown function"),
        ("n_counterparties(ALERT.receiver, dir=in)", "missing"),
        ("n_counterparties(ALERT.receiver, dir=in, window=before(72), foo=1)", "unknown"),
        ("n_counterparties(ALERT.payers, dir=in, window=before(72))", "account slot"),
        ("set_contacts(ALERT.receiver, dir=in, window=before(72))", "set slot"),
        ("n_counterparties(ALERT.receiver, dir=sideways, window=before(72))", "one of"),
        ("n_counterparties(ALERT.receiver, dir=in, window=before(0))", "1..336"),
        ("n_counterparties(ALERT.receiver, dir=in, window=before(9999))", "1..336"),
        ("n_counterparties(ALERT.receiver, dir=in, window=72)", "window"),
        ("n_new_counterparties(ALERT.receiver, window=before(24))", "after"),
        ("at_least(passed_on(ALERT.receiver, window=after(24), min_frac=0.5), k=2)", "count or set_count"),
        ("at_least(n_tx(ALERT.receiver, dir=in, window=before(24)), k=0)", "out of range"),
        ("top_outflow_bank(ALERT.receiver, banks=[70], window=after(24))", "2-5"),
        ("top_outflow_bank(ALERT.receiver, banks=[a, b], window=after(24))", "integer bank ids"),
        ("passed_on(ALERT.receiver, window=after(24), min_frac=1.5)", "out of range"),
        ("n_round(ALERT.receiver, window=around(24), unit=1000, unit=100)", "twice"),
        ("n_counterparties(acct_77, dir=in, window=before(24))", "slot"),
        ("overlap(ALERT.receiver, dir=in, window=before(24), among=ALERT.sender)", "ALERT.payers"),
        ("entity_accounts(ALERT.receiver)", "not askable"),
        ("n_tx(ALERT.receiver, dir=in, window=before(24), min_amount=true)", "number"),
    ],
)
def test_invalid_programs_get_actionable_errors(program, fragment):
    c = check(program)
    assert not c.ok
    assert fragment in c.error, c.error


def test_no_literal_account_or_date_can_be_expressed():
    for program in [
        "n_counterparties(ALERT.receiver, dir=in, window=before(72), account=12345)",
        "n_counterparties(ALERT.receiver, dir=in, window=before(72), since=2022-09-01)",
        "n_counterparties('0x8F31', dir=in, window=before(72))",
    ]:
        assert not check(program).ok


def test_reference_text_lists_every_askable_primitive():
    text = reference_text()
    for name, p in PRIMS.items():
        assert (name in text) != p.restricted
