"""orchestrator: runs investigators over alerts through the harness (C15), sequential mode for now.

Owner: T2 (Bhavya).  Component: C15.  Tier: Core.
Spec: plan.md 6.16; tracks.md 4.8.

`runner.py`: build a planner, run it over alerts in fresh harness sessions, score the verdicts, write run-log rows.
Batched (async) mode for RQ5 is week 8 to 11. This package builds the banks, so it sits outside the packages that
systems live in (the static test in tests/test_harness.py keeps those from importing banks).
"""
