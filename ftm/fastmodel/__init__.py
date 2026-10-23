"""fastmodel: System 1 fast model (own Qwen3-4B, SemIf, Laya) and the conformal cascade.

Owner: T3 (Akshat).  Component: C13.  Tier: Secondary / Stretch.
Spec: plan.md 6.14; tracks.md 5.6.

decisions.py: the five decisions, single-token labels, safe defaults, expected dataset format (hand-off #13).
model.py: LogprobFastModel (own 4B, one forward pass) and JsonDecisionModel (8B-writes-JSON baseline).
calibration.py: temperature scaling + top-label ECE.  cascade.py: Mondrian split-conformal two-tier cascade.
splits.py: which val part fits what (temperature on `fit`, conformal sets on `cal`, report on `eval`).
Not started: SemIf, Laya (stretch). Fallback if the 4B fails: the 8B makes every small decision.
"""
