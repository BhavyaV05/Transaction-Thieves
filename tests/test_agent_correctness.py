"""Group D correctness fixes (finding 13): Reflexion's forced final answer."""

from ftm.external.reflexion import Reflexion
from ftm.harness.interface import ScriptedToolBox
from ftm.llm.mock import MockLLM
from ftm.types import AlertContext, Limits, Reply

Q = "dormant(ALERT.receiver, days=7)"
ALERT = AlertContext("V9", "70", "2022-09-05T12:00:00", "val", 0.2, {"receiver": "R", "evidence": "e", "partner_banks": [220, 1588]})


def test_reflexion_forces_a_final_answer_like_react_when_the_clock_runs_out():
    llm = MockLLM(script=[f"Thought 1: ask\nAction 1: ask_bank[220 | 1.0 | {Q}]", "Thought 2: so\nAction 2: finish[0.4]", "CONFIDENT"])
    v = Reflexion(llm).investigate(ALERT, ScriptedToolBox([Reply("220", "bool", True, 1.0, 6.0)]), Limits())  # 6 days > the 5-day clock
    assert v.probability == 0.4 and v.trials == 1
    assert "No further question can be asked" in llm.requests[1]["messages"][-1]["content"]
