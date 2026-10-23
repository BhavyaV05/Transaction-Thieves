"""T3: DSPy tuning on val `fit`, selection on val `eval` (T4 review of PR #14). No dspy needed."""

import pytest

from ftm.external.dspy_react import CountingClient, optimize, require_part, select_best
from ftm.harness.interface import ScriptedToolBox
from ftm.llm.mock import MockLLM
from ftm.types import AlertContext

ALERT = AlertContext("V9", "70", "2022-09-05T12:00:00", "val", 0.2, {"receiver": "R", "evidence": "rows", "partner_banks": [220]})


def ex(part, label=1):
    return {"alert": ALERT, "tools": ScriptedToolBox(), "label": label, "part": part}


class Fixed:
    def __init__(self, p):
        self.p = p

    def investigate(self, alert, tools, limits):
        return type("V", (), {"probability": self.p})()


def test_tuning_and_selection_parts_are_never_mixed():
    with pytest.raises(ValueError, match="fit"):
        optimize(MockLLM(), [ex("fit"), ex("eval")])  # refused before dspy is imported
    with pytest.raises(ValueError, match="fit"):
        optimize(MockLLM(), [ex("cal")])
    with pytest.raises(ValueError):
        require_part([{"alert": ALERT}], "fit", "x")  # no part named
    with pytest.raises(ValueError, match="eval"):
        select_best({"a": Fixed(0.5)}, [ex("fit")])
    require_part([ex("fit")], "fit", "x")


def test_select_best_scores_brier_on_eval_and_counting_client_counts():
    best, scores = select_best({"lo": Fixed(0.1), "hi": Fixed(0.9), "none": Fixed(None)}, [ex("eval"), ex("eval", 0)])
    assert best == "none" and scores["hi"] == pytest.approx(0.41) and scores["none"] == pytest.approx(0.34)  # None scores p0
    c = CountingClient(MockLLM())
    c.chat([{"role": "user", "content": "x"}])
    assert c.totals()["llm_calls"] == 1 and c.totals()["tokens_in"] >= 0
