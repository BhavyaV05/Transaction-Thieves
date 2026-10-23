"""harness: the shared agent harness (C16). `interface.py` is the contract (frozen at week 4); `toolbox.py` is the real
ToolBox: gate + accountant + clock in front of `ask_bank`, a mock mode, and `run_one`.

Owner: T2 (Bhavya).  Component: C16.  Tier: Core.
Spec: plan.md 6.16; tracks.md 4.8.

    from ftm.harness import make_toolbox
    tools = make_toolbox(seed=0)               # mode="mock" (offline) or mode="world", data="hi_small"
    system = tools.wrap(system)                # one session per alert, authoritative questions log
"""

from ftm.harness.toolbox import (  # noqa: F401
    Harness,
    HarnessToolBox,
    MockBank,
    SessionError,
    ToolUnavailable,
    make_toolbox,
    run_one,
)
