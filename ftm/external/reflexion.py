"""Reflexion (Shinn et al.).

Owner: T3 (Akshat).  Tier: Core.  Target week: 6.
Use official code where released, else reimplement from the paper and label it a reimplementation.
Same backbone (Qwen3-8B), same gate, same eps, same caps as every other system (plan.md 7.2).
"""

from ftm.harness.interface import ToolBox
from ftm.types import AlertContext, Limits, Verdict


class Reflexion:
    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict:
        raise NotImplementedError("reflexion: not implemented yet")
