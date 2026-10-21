import pytest
from lark import Lark
from lark.exceptions import LarkError

from ftm.dsl import GRAMMAR_PATH

PARSER = Lark(GRAMMAR_PATH.read_text(), parser="lalr")


@pytest.mark.parametrize(
    "program",
    [
        "passthrough(ALERT.receiver, min_frac=0.8, within_h=24)",
        "n_counterparties(ALERT.receiver, direction=out, window_h=24)",
        "top_outflow_bank(ALERT.receiver, banks=[b1, b2, b3], window_h=72)",
        "set_overlap(ALERT.payers, ALERT.payees, bank=b2)",
    ],
)
def test_toy_programs_parse(program):
    assert PARSER.parse(program).data == "start"


@pytest.mark.parametrize("program", ["passthrough(", "f(ALERT)", "f(x=)", "f(ALERT.receiver) extra"])
def test_malformed_programs_are_rejected(program):
    with pytest.raises(LarkError):
        PARSER.parse(program)
