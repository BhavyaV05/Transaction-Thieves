"""intake: Class-conditional (Mondrian) conformal alert filter, selective conformal risk control, time-weighted calibration.

Owner: T1 (Abhishek).  Component: C2.  Tier: Core.
Spec: plan.md 6.3; tracks.md 3.6.

Done (draft): conformal.py (Mondrian, time weights, selective risk bound), alerts.py (drop rules, projection),
scripts/make_alerts.py. Not done: test alert set and the frozen 1,000-alert subset (need test labels; locked).
See README.md for schemas, results and the open decisions.
"""
