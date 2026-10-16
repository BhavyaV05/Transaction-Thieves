"""planners: Non-LLM planners (random, fixed, EIG, learned, enumerator) owned by T2. llm.py (System 2 planner) is owned by T3.

Owner: T2 (Bhavya).  Component: C9 (+C10 in llm.py).  Tier: Core.
Spec: plan.md 6.10-6.11; tracks.md 4.7, 5.4.

First steps:
  - same budget and stop rule for every planner
"""
