# T3: LLM planner (C10) and Group D agents: design, how to run, what is still blocked

Owner: T3 (Akshat). Spec: `tracks.md` §5.4 and §5.7, `plan.md` §6.11, §6.17 and §7.2. This page is the reference for code under `ftm/planners/llm*.py`, `ftm/llm/` and `ftm/external/{react,reflexion,rewoo,...}.py`. It covers each component and when to update it.

## 1. Layers

```
scripts/run_llm_agent.py           alerts -> system.investigate(alert, tools, limits) -> run log, traces, question batch
        │
        ├── ftm/planners/llm.py         LLMPlanner: our System 2 planner, 5 modes (one JSON step per call)
        ├── ftm/external/react.py       React (and later Reflexion, ReWOO, ...): text Thought/Action/Observation loops
        │     └── external/text_agent.py   tool grammar + ToolExecutor shared by every Group D port
        │
        ├── ftm/llm/investigation.py    Investigation: per-alert eps / sim-time / questions log / posterior (SAME for all)
        │                               Fusion protocol + NaiveLRFusion + LRTable (placeholders for T4 / T2)
        ├── ftm/llm/budget.py           BudgetedClient: max_llm_calls, prompt cap, cost fields (SAME for all)
        ├── ftm/planners/llm_prompt.py  prompt + per-state JSON schema;   llm_dsl.py: DslAdapter (+ provisional DSL)
        └── ftm/llm/client.py / mock.py endpoint client; mock with script / record-replay / HTTP
```

Fairness (`plan.md` §7.2) is enforced by construction. Every LLM system asks banks only through `Investigation.ask`, which goes through `ToolBox.ask_bank` with the gate and the accountant behind it. Every system calls the LLM only through one `BudgetedClient` per alert, and sees the same alert summary, evidence text, partner banks and question language. Only the control loop differs.

## 2. The planner (C10)

One step = one LLM call that returns one JSON object (schema in `llm_prompt.planner_schema`, built per state):

| Key | When | Notes |
|---|---|---|
| `hypothesis` {typology, rationale} | all modes except `act_only` | Logged only; never enters fusion |
| `action` | always | `select` (listed question), `compose` (new DSL question), `stop` |
| `question_id` / `program` | select / compose | Enum of listed ids; program is checked by the `DslAdapter` |
| `bank` | select / compose | Enum of partner banks |
| `eps` | always | One of {0, 0.25, 0.5, 1, 2}, never more than the eps left. Discrete, so the back-test cache (canonical + ε) is reusable |
| `rule_id` | optional | Enum of retrieved rule ids; T4's gate checks it |
| `belief` | `no_backtest`, `llm_decides` | The LLM's P(laundering); the verdict in those two modes only |

**Modes and what decides the verdict**

| Mode | Questions | Shown the back-tested posterior | Verdict |
|---|---|---|---|
| `compose` (main) | select + compose | yes | formula (Fusion posterior) |
| `fixed_list` | select only | yes | formula |
| `act_only` | select + compose, no hypothesis | yes | formula |
| `no_backtest` (H4) | select + compose | **no** | LLM `belief` |
| `llm_decides` (H7) | select + compose | yes | LLM `belief` |

**Loop rules.**
- A bad JSON, an invalid DSL program, or the same question to the same bank again is fed back as "Your previous output was rejected: …" and retried. A retry costs a call.
- The loop ends on `stop`, when the calls are used up, when the eps left is below 0.25, when the simulated time is used up, or after 3 failures in a row.
- Composed questions that reached a bank are appended to the question batch (hand-off #8).

**Metrics** (`LLMPlanner.stats`, printed in `summary.json`):
- `json_parse_rate`: share of outputs that parse.
- `valid_dsl_rate`: share of proposed questions that are valid DSL.
- `usable_output_rate`: share of outputs that are a stop or a valid question. **This is the ≥95% acceptance number.**
- `median_seconds_per_alert`: the target is 8–20 s on the reference machine.

## 3. Group D ports (ported mode)

The tool grammar is the same for every port (`text_agent.py`): `ask_bank[bank | eps | question]`, `lookup_rules[query]`, `finish[p]`. Observations are typed values, notes come wrapped in `<untrusted_note>`, and errors are observations too.

| System | Status | Source and licence | What we keep from the paper |
|---|---|---|---|
| ReAct | **done** (week 4) | Reimplementation; official code MIT, nothing copied | Interleaved Thought/Action/Observation, a stop sequence before Observation, Finish ends the episode. One in-domain exemplar, written by us and shared by every port |
| Reflexion | **done** (week 6) | Reimplementation; official code MIT, nothing copied | Actor (our ReAct loop), Evaluator, verbal Self-Reflection, episodic memory across trials on the same alert. **Adaptations:** the evaluator is the LLM judging itself (labels are never shown); all trials share one ε budget, call cap and `Investigation`, so later trials see the real answers already received |
| ReWOO | **done** (week 6) | Reimplementation; official code MIT, nothing copied | Planner writes the whole plan (`#E<n> = tool[args]`, later steps may cite `#E<k>`), the Worker runs it with no LLM call, the Solver gives `finish[p]`. 2 LLM calls per alert. The Solver output must be `finish[p]` or a bare number; a stray number is never taken as the answer |
| LATS | **done** (week 8), **adapted** | Reimplementation; official code MIT, nothing copied | Expansion (n sampled children, one call each), an LM value function (one call scores all children), executing the best child, and backtracking to the best unexecuted child when an action fails. **Dropped:** simulated rollouts and value back-propagation, because every `ask_bank` spends ε and cannot be reset. Labelled "LATS (adapted: no rollouts)" |
| LLMCompiler | **done** (week 8) | Reimplementation; official code MIT, nothing copied | Planner (numbered tasks, `$k` dependencies, `join()`), Task Fetching Unit (waves of independent tasks inside `Investigation.parallel()`, so simulated time = the slowest answer per wave; relevant to RQ5/E13), and a Joiner that finishes or replans (up to 2 replans) |
| DSPy-tuned ReAct | **code done** (week 9); **not yet optimised** | The library (dspy 3.4.x, MIT) | `dspy.ReAct` over the shared tools. Every LM call goes through our `BudgetedClient` (DSPy cache off), `max_iters = max_llm_calls − 1`. `optimize()` runs MIPROv2 with 1 − Brier as the metric on **validation** alerts (tuning parity) |

End-to-end probability is the number in `finish[...]`. It is `None` if the calls run out first. The run script scores a `None` as `p0` and counts these cases in `summary.json`; T4's eval code should use the same rule (to agree). Questions-only scoring uses the questions log, as for every system.

## 4. Strategy library (C12)

- **Key** (`ftm/library/shape.py`, schema v1, **frozen at G3** once the team agrees): typology guess (or `any`), in-degree and out-degree buckets of the receiver in the evidence, number of banks, median-amount bucket + flat/varied, and the time span. No account id. The buckets are coarse on purpose.
- **Build** (`ftm/library/build.py`, `scripts/build_library.py`):
  - Input is one record per (past train/val alert, question asked).
  - Gain = Brier(before) − Brier(after) for that alert's label.
  - Records are grouped by (shape key, question, ε). A group needs **≥ `n_min` = 20 alerts** before it can produce anything:
    - **success:** the question was accepted in back-testing and mean gain − 2·se > 0;
    - **failure:** it was rejected, or mean gain + 2·se < 0.
  - So a failure note never comes from one noisy answer. Test records raise an error.
- **Search** (`StrategyLibrary.search`): exact key first, then the nearest shape with the same typology (or `any`); successes before failures. The planner looks up once at the start (typology `any`) and again whenever its typology guess changes.
- **Lifecycle:** `freeze()` returns the content hash. A frozen library refuses `add`. `load` checks the schema versions and the hash. `run_llm_agent.py --library` refuses an unfrozen file and writes the hash into `summary.json`.
- **E11 ablation:** `keying="account"` files strategies by the receiver alias, to show that this memorises known criminals. Never use it in the main system.
- **Until T2/T4 wire it**, `LibraryToolBox(tools, library)` makes any ToolBox's `search_library` answer from a library file.

## 5. Running

```bash
# offline smoke run: no GPU, no data, random dev answers (variant=devtools, never a result)
python scripts/run_llm_agent.py --system planner --mode compose --n 5
python scripts/run_llm_agent.py --system react --n 5

python scripts/run_llm_agent.py --system reflexion --n 5        # or rewoo, lats, llmcompiler, dspy_react (needs pip install -e .[llm])

# record a real run, replay it later without a GPU (token counts and latency replay too)
python scripts/run_llm_agent.py --system planner --llm endpoint --record outputs/cassettes/p20.jsonl ...
python scripts/run_llm_agent.py --system planner --replay outputs/cassettes/p20.jsonl ...

# G2 (planner, 20 val alerts) and G3 (ReAct, 50 val alerts), once the blockers below are cleared
python scripts/run_llm_agent.py --system planner --alerts val --data hi_small --n 20 --llm endpoint --toolbox harness --machine t3-5070ti
python scripts/run_llm_agent.py --system react   --alerts val --data hi_small --n 50 --llm endpoint --toolbox harness --machine t3-5070ti
```

Real-alert runs build the evidence with `ftm.retrieval.views.top_k_by_amount_view` and the summary with `alert_summary`. The partner banks are the 30 view banks minus the home bank. Outputs go to `outputs/runs/<run_id>/`:
- `runlog.parquet` (schema `ftm.eval.runlog`)
- `traces.jsonl` (per-step model text, aliased; used for case studies and distillation)
- `questions.jsonl` (hand-off #8)
- `summary.json`

## 6. Blocked on other tracks

| Needed for | What | From | Workaround now |
|---|---|---|---|
| Real DSL validation in the planner | Grammar v0 + parser/type checker (hand-off #3) | T2 | `ProvisionalDsl` behind `DslAdapter`; swap in one place |
| G2 / G3 runs | Harness v0 with `ftm.harness.make_toolbox(seed=...)`, partner banks, gate, accountant (hand-offs #4, #7) | T2, T4 | `DevRandomToolBox` (random answers, `--toolbox dev`) |
| Posterior in compose / fixed_list / act_only | Back-test table (hand-off #9) and fusion v0 (#18) | T2, T4 | `NaiveLRFusion` over an `LRTable` (posterior = p0 until LRs exist) |
| How gate and accountant refusals arrive | Exception types from harness v0 | T2, T4 | Classified by class name (`Investigation` docstring); unknown errors propagate |
| Real val alerts on this machine | `data/alerts/<ds>/val.parquet`, views, aliases | T1 (rebuild with `prepare_data.py` and `make_alerts.py`) | synthetic alerts |
| Real LLM | Endpoint | T3 (GPU smoke test) | `MockLLM` + `dev_responder` |
| A real strategy library | Teacher / planner runs on train+val, and the back-test table (accepted flags) | T3 runs, T2 table | `build_library.py` takes the back-test table as `LRTable.from_rows` JSONL until hand-off #9 fixes the format |
