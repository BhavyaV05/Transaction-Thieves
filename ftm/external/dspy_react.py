"""DSPy-tuned ReAct (MIPROv2), objective = Brier on validation. The strongest tuned external agent.

Owner: T3 (Akshat).  Tier: Secondary.  Target week: 9.
Use official code where released, else reimplement from the paper and label it a reimplementation.
Same backbone (Qwen3-8B), same gate, same eps, same caps as every other system (plan.md 7.2).
"""

from ftm.harness.interface import ToolBox
from ftm.types import AlertContext, Limits, Verdict


class DspyReact:
    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict:
        raise NotImplementedError("dspy_react: not implemented yet")
