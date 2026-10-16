"""System 2 LLM planner (C10). Owner: T3 (Akshat). See plan.md 6.11, tracks.md 5.4.

Qwen3-8B, structured JSON output (hypothesis, action select/compose/stop, target bank, eps, rule citation).
Modes: compose (main), fixed-list, act-only, no-back-testing, LLM-decides.
Done when >=95% of outputs parse into valid DSL on validation and the loop runs at 8-20 s per alert.
"""
