"""library: Strategy library: short strategies filed by situation shape (never by account ID); failure notes only from back-test statistics.

Owner: T3 (Akshat).  Component: C12.  Tier: Secondary.
Spec: plan.md 6.13; tracks.md 5.5.

shape.py: the SituationShape key (schema frozen at G3).  store.py: StrategyLibrary (search, freeze, hashed
save/load; "account" keying is the E11 ablation only).  build.py: successes and failures from >= n_min alerts.
"""
