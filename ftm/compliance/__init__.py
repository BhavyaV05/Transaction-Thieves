"""compliance: Allowlist compliance gate and regulatory RAG over the rules corpus (BM25 + small embeddings); rule-ID citations are checked.

Owner: T4 (Shreyas).  Component: C6.  Tier: Core (gate) / Secondary (RAG).
Spec: plan.md 6.7; tracks.md 6.3.

First steps:
  - gate + lookup_rules to T2/T3 by week 5
  - 100% of disallowed actions in the hand-written suite blocked
"""
