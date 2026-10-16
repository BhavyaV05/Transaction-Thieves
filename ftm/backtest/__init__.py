"""backtest: Back-testing: LR estimation, e-values, e-BH at 10% FDR, online e-value rule, question families, cache by canonical form + eps. Statistics reviewed by T4.

Owner: T2 (Bhavya).  Component: C7.  Tier: Core.
Spec: plan.md 6.8; tracks.md 4.6.

First steps:
  - realistic mode first, exact mode as ablation
  - design for sharding by question from the start
  - false-acceptance <=10% on synthetic null questions
"""
