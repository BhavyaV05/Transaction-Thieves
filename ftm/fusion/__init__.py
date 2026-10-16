"""fusion: logit p = a*logit(p0) + sum_f w_f*log LR_f(answer) + c. No LLM text enters. Per-family weights, shuffled-answers test.

Owner: T4 (Shreyas).  Component: C8.  Tier: Core.
Spec: plan.md 6.9; tracks.md 6.4.

First steps:
  - v0 in week 2 (needed for the G2 Brier-gain check), v1 in week 7
"""
