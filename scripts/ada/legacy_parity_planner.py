"""Run `scripts/run_llm_agent.py` with the FROZEN pre-graph planner loop (tests/legacy_llm_planner.py) instead of the graph planner, for the real-model parity check.  Owner: T3 (Akshat).

    (from the repo root) python scripts/ada/legacy_parity_planner.py <the usual run_llm_agent.py arguments>

Test tooling, not a production path: it only swaps the class `run_llm_agent.LLMPlanner` for the legacy copy (same constructor), so a legacy run and a graph run
differ in nothing but the loop.  For the external agents the switch is the existing env var FTM_LEGACY_AGENT_LOOPS=1 (ftm/graph/agents/registry.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path.cwd()  # run from the repo root of the tree under test (the script may be copied into it)
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "scripts")]

import legacy_llm_planner  # noqa: E402
import run_llm_agent  # noqa: E402

run_llm_agent.LLMPlanner = legacy_llm_planner.LLMPlanner
sys.exit(run_llm_agent.main(sys.argv[1:]))
