# End-to-end demo

One alert, end to end: what the home bank sees, how the investigation unfolds step by step, and the decision. Train and val alerts
only (test ids `F......` are refused; the test tables are never opened). Code: `ftm/demo/`, entry point `scripts/demo.py`.
Use the repo venv: `export PYTHONPATH=$PWD; PY=/home/akshat/projects/Transaction-Thieves/.venv/bin/python`.

## Commands

The demo runs the production profile (`configs/llm/tuned.yaml`: p0 v2, edge scorer v2 evidence view, headroom_v1 table and fitted fusion) and escalates at the
threshold of `configs/decision/headroom_v1.yaml` (0.3227, fitted on val fit+cal). The default is a real model: live on the endpoint, or a replay of a recorded real run.
The scripted LLM exists only behind an explicit `--llm mock` and is labelled "SCRIPTED MOCK LLM" in every output.

```bash
# present recorded real runs (no GPU, no LLM); repeat --replay for a side-by-side comparison
$PY scripts/demo.py --alert V001549 --replay outputs/demo/V001549/planner --replay outputs/demo/V001549/react_adapted --pause 1.5

# run it now against the model endpoint (vLLM; FTM_LLM_ENDPOINT / FTM_LLM_MODEL or --endpoint / --model), keep the runs for later
$PY scripts/demo.py --data hi_medium --alert V000438 --compare react_adapted --record-dir outputs/demo/V000438

# one self-contained HTML page (inline CSS, no external assets) for slides; combine with any mode above
$PY scripts/demo.py --alert V001549 --replay outputs/demo/V001549/planner --replay outputs/demo/V001549/react_adapted --html outputs/demo/V001549.html

# reveal the true label at the end (train / val only; off by default)
$PY scripts/demo.py --alert V001549 --replay outputs/demo/V001549/planner --reveal-label

# no GPU, development only: the scripted stand-in (banner + labels in the output; never a result)
$PY scripts/demo.py --data hi_medium --alert V000438 --llm mock --compare react_adapted
```

Recorded examples from before the profile (SCRIPTED MOCK LLM, back-test `7df6bad5337d`, not committed; shown with the mock banner; re-record on the endpoint for real ones): `outputs/demo/{V000438,V001549,V001835,V001453}/{planner,react_adapted}`.
V001549 (laundering): the questions lift p from 0.441 to 0.666 and the case moves from human review to escalate. V001835 (clean): 0.713 to 0.431,
a false escalation becomes a human review. V001453 (laundering): the answers pull 0.618 down to 0.313, a failure case, honestly shown.
The new back-test namespace `7613b89b5dcc` loads too (`--backtest data/backtest/hi_medium/7613b89b5dcc`, 440 rows) but only 6 questions are accepted
there (all at eps 2), none that the mock finds applicable on V000438 / V001835, so the mock stops without asking.

## What each mode does

* `--live` (the default without `--replay`): records a one-alert run through the normal `scripts/run_llm_agent.py` wiring (harness toolbox, gate,
  accountant, back-test fusion, your `--llm`), into `--record-dir` (default a temp dir; one sub dir per system), then plays it. Live and replay
  are the same renderer, so a run recorded on the GPU box can be shown anywhere. No change to `ftm/graph` was needed (no hook).
* `--replay RUN_DIR [--replay ...]`: renders from `runlog.parquet`, `traces.jsonl`, `per_alert.jsonl`, `summary.json` of each run dir.
  Two or more run dirs (or `--compare` live) render side by side, with a comparison table and "p0 only".
* `--llm mock` (explicit only): a scripted stand-in (`ftm/demo/mock.py`), labelled "[scripted mock LLM]" in the output. It works through the back-tested
  menu (largest possible odds shift per unit eps, two questions) so the posterior visibly moves; its thoughts are canned. Not a result.
* `--pause S`: seconds between steps on a terminal (0 or a non-terminal: all at once). `--width`, `--evidence-rows`, `--evidence-view` tune the layout.
* `--threshold` (default: `configs/decision/headroom_v1.yaml`) and the optional `--clear-below` (adds a human-review band; off by default because the deployed decision is two-way).

## What the replay needs from a run dir

| Panel field | Source | If missing |
|---|---|---|
| question, bank, eps, gate decision, answer, posterior after | `runlog.parquet` step rows | required |
| p0, final probability, LLM calls, eps spent | `runlog.parquet` verdict row | required |
| planner thought / action, requested bank, rejected proposals | `traces.jsonl` (`rationale` / `thought`, `action`, `bank_requested`, `error`) | "not recorded" |
| LR of the answer | `log_lr` in the run log | looked up in the back-test table |
| answer bin | back-test table (`--backtest`, else `summary.json` backtest namespace under `data/backtest/<data>/`) | LR shown without the bin |
| eps left | `summary.json` `eps_per_alert` (default 2) minus the replies so far | "not recorded" |
| cited rule | `rule_id` in the run log, else re-derived from `configs/compliance` (marked "re-derived") | omitted |
| alert panel (amount, banks, evidence trail, true label) | the alert tables in `data/` (train / val), rebuilt with the run's evidence view | p0-only note |

The harness does not log the rule it cited for an allowed question today (`rule_id` / `gate_reason` are empty in run logs), hence the re-derivation.
