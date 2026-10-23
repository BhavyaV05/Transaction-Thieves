# ftm/harness: the shared agent harness, v0 (C16)

Owner: T2 (Bhavya). Spec: `plan.md` 6.16, `tracks.md` 4.8. **Harness v0 (hand-off to T3 and T4). The interface (`interface.py`) freezes at v1, week 4.**

```python
from ftm.harness import make_toolbox

tools = make_toolbox(seed=0)                 # offline: random typed answers, but the real gate and accountant
tools = make_toolbox(seed=0, mode="world", data="hi_small", split="val")   # real banks, real noise, real alerts
system = tools.wrap(system)                  # one session per alert; the verdict carries the toolbox's log
verdict = system.investigate(alert_ctx, tools, limits)
```

## What `ask_bank(bank, program, eps)` does

Always the same road: **gate** (T4: default deny, valid question, no identifying literal, legal basis, allowed eps level) then **accountant** (per-alert budget, the smaller of `Limits.eps_per_alert` and the accountant's, and the per-partner-account cap with the accounts the question can see) then the **clock** (a reply later than `Limits.max_sim_time_days` is cut off as a `Timeout`) then the bank.

* Refusals raise `GateBlocked` (a `PermissionError`) or `BudgetRefused`, the classes `ftm/llm/investigation.py` already recognises. Each refusal is logged (`gate_decision` blocked / over_budget) and the bank is never called.
* **eps is charged only when a `Reply` comes back.** A decline or timeout releases nothing, so it costs no eps (it still costs time).
* Asking the same question again gives the same reply (the bank's draws are seeded per question) and is charged again.
* An optional `rule_id=` argument is passed to the gate for citation checks (`require_citation`).
* `local_graph(query, as_of)` raises `TimeGuardError` if `as_of` is after the alert; `lookup_rules` goes through the gate; `search_library` is empty until T3 plugs in `library_fn`. In `native` mode `ask_bank` raises `ToolUnavailable`.

## Sessions (what T3's runner needs)

`ask_bank` carries no alert id, and the runner builds one toolbox for many alerts, so a toolbox serves **one alert at a time**: `begin(alert, limits)` / `end()`. `tools.wrap(system)` does both around `investigate`, and `run_one` returns the verdict with `questions_log` **replaced by the toolbox's own log**, so a system cannot misreport what it asked (the eval scores that log). For `scripts/run_llm_agent.py` this is one line after the toolbox is built: `system = tools.wrap(system)` (T3's call; or call `tools.begin/end` inside `Investigation`).

## Bypass tests (`tests/test_harness.py`)

Eleven kinds of malformed or forbidden question never reach a bank; the third question over budget never reaches a bank; the per-account cap holds across alerts; a bank with a ledger refuses to answer without the alert's real ids; nothing public on the toolbox hands out a bank, the gate or the accountant (the public surface is pinned); 360 random asks over three alerts never exceed any budget; and a static test checks that nothing in `ftm/external`, `planners`, `llm`, `agent`, `library`, `fastmodel` or `baselines` imports `ftm.banks`, `ftm.privacy` or `ftm.backtest`.

## Mock mode

`MockBank`s 1 to 30 answer at random (bool 50/50, counts Poisson, k-ary uniform), seeded per question; replies are never results. Everything else is real, so a system developed against the mock meets the same refusals as on the real run. Mode `world` needs T1's data on the machine.

## Not in v0

Orchestrator (sequential and batched SimPy mode, run-log writer), the batched asks of RQ5, and the harness-level fairness report: weeks 4 to 5. The `ToolBox` contract is unchanged; `SessionToolBox` is an added protocol.
