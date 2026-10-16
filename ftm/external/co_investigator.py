"""Co-Investigator AI (arXiv:2509.08380). Native + ported modes. Domain agents run in two modes: native (one-bank view, no ask_bank) and ported (ask_bank, same eps).

Owner: T2 (Bhavya).  Tier: Secondary.  Target week: 7.
Use official code where released, else reimplement from the paper and label it a reimplementation.
Same backbone (Qwen3-8B), same gate, same eps, same caps as every other system (plan.md 7.2).
"""

from ftm.harness.interface import ToolBox
from ftm.types import AlertContext, Limits, Verdict


class CoInvestigator:
    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict:
        raise NotImplementedError("co_investigator: not implemented yet")
