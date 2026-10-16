"""banks: PartnerBank nodes: exact DSL execution on a private view, DP noise, SimPy latency/decline/timeout, untrusted text note.

Owner: T2 (Bhavya).  Component: C3.  Tier: Core.
Spec: plan.md 6.4; tracks.md 4.4.

First steps:
  - PartnerBank.handle(question, as_of) -> Reply | Decline | Timeout
  - no code path may return raw rows (static check + test)
"""
