"""LLMCompiler (Kim et al.). Parallel calls; relevant to RQ5.

Owner: T3 (Akshat).  Tier: Secondary.  Target week: 8.
Use official code where released, else reimplement from the paper and label it a reimplementation.
Same backbone (Qwen3-8B), same gate, same eps, same caps as every other system (plan.md 7.2).
"""

from ftm.harness.interface import ToolBox
from ftm.types import AlertContext, Limits, Verdict


class Llmcompiler:
    def investigate(self, alert: AlertContext, tools: ToolBox, limits: Limits) -> Verdict:
        raise NotImplementedError("llmcompiler: not implemented yet")
