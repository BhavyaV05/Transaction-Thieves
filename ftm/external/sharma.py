"""Sharma [3], faithful port. Native + ported modes. Domain agents run in two modes: native (one-bank view, no ask_bank) and ported (ask_bank, same eps).

Owner: T4 (Shreyas).  Tier: Core.  Target week: 6.
Use official code where released, else reimplement from the paper and label it a reimplementation.
Same backbone (Qwen3-8B), same gate, same eps, same caps as every other system (plan.md 7.2).
"""

from ftm.harness.interface import ToolBox
from ftm.types import AlertContext, Limits, Verdict


class Sharma:
    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict:
        raise NotImplementedError("sharma: not implemented yet")
