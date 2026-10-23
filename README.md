# Transaction-Thieves: Follow the Money Without Moving the Data

Team **Cache me if you can** (LMA Group Project 2026). Banks see only their own customers' transactions, but launderers route money across banks. Our system lets a bank ask other banks a few small, private, typed questions (each answered with one differentially-private value), back-tests every question before it may affect a decision, and makes the final call with a formula, never with LLM text.

| Read this | For |
|---|---|
| [`plan.md`](plan.md) | The full project plan: research questions, hypotheses, architecture, components, experiments, timeline, risks |
| [`tracks.md`](tracks.md) | Four tracks, one per person: what to build, how, hand-offs, week-by-week, cut order. **Start here for your task** |
| [`tasks/`](tasks/) | One checklist per track, generated from `tracks.md` |
| [`STATUS.md`](STATUS.md) | One line per track per week |
| [`docs/hardware.md`](docs/hardware.md) | Machine setup, week-0 smoke tests, shared LLM server, GPU rules |
| [`docs/ta_addendum.md`](docs/ta_addendum.md) | Draft addendum to the TA: what changed since the proposal (G0) |
| [`docs/external_code_check.md`](docs/external_code_check.md) | Code and licence status of every Group C/D external system |
| [`GPU_CALENDAR.md`](GPU_CALENDAR.md) | Who has which GPU when (QLoRA and teacher runs take a machine offline) |

## Who owns what

| Track | Owner | Packages (`ftm/`) | Scripts | Checklist |
|---|---|---|---|---|
| **T1** Data, detection models, retrieval | Abhishek | `data`, `baselines`, `intake`, `retrieval` | `prepare_data`, `train_gfp`, `train_gnn`, `make_alerts`, `train_federated` | [`tasks/T1_abhishek.md`](tasks/T1_abhishek.md) |
| **T2** Banks, questions, back-testing | Bhavya | `banks`, `dsl`, `privacy`, `backtest`, `planners` (all but `llm.py`), `agent`, `harness`, `external/{co_investigator,finfre_rag}` | `backtest`, `run_experiment` | [`tasks/T2_bhavya.md`](tasks/T2_bhavya.md) |
| **T3** LLM planner, fine-tuning, agent frameworks | Akshat | `llm`, `planners/llm*.py`, `retrieval/views.py` (stub, until week 4), `library`, `fastmodel`, `external/{react,reflexion,rewoo,lats,llmcompiler,dspy_react}` | `serve_llm`, `bench_llm`, `run_llm_agent`, `build_library`, `train_planner` | [`tasks/T3_akshat.md`](tasks/T3_akshat.md) |
| **T4** Rules, safety, evaluation, domain agents | Shreyas | `compliance`, `fusion`, `decision`, `safety`, `eval`, `external/{torres,sharma,debate_autogen,uot,bed_llm}` | `run_final` | [`tasks/T4_shreyas.md`](tasks/T4_shreyas.md) |

Every module's docstring names its owner, plan section, tier and first steps. Review pairs: T1 ↔ T2, T3 ↔ T4.

## Setup

```bash
git clone <repo-url> && cd Transaction-Thieves
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev]"         # base install (what CI runs)
pytest -m "not gpu and not slow"
```

Extras by track (install only what you need):

| Extra | Contents | Who |
|---|---|---|
| `gbt` | LightGBM, XGBoost, scikit-learn | T1, T2 |
| `gnn` | PyTorch, PyG, pcst_fast (install PyTorch from pytorch.org for your CUDA first) | T1 |
| `fed` | Flower, Opacus | T1 |
| `llm` | OpenAI client, DSPy, AG2 (client side only) | T3, T4, T2 |
| `serve` | vLLM, Unsloth (serving machine only; WSL2/Docker on Windows) | T3 (and T1 if the 8B fits) |
| `rag` | BM25, sentence-transformers | T4 |

`snapml` (IBM GraphFeaturePreprocessor) is installed separately by T1; see the licence check in [`tasks/T1_abhishek.md`](tasks/T1_abhishek.md).

## Layout

```
configs/      Hydra configs (data, banks, privacy, llm, experiments)
data/         gitignored: raw/, parquet/, views/, alerts/, runs/   (see data/README.md)
docs/         hardware and setup notes
ftm/          main package; one subpackage per component (docstrings list owner and first steps)
  types.py        shared types: AlertContext, Reply, Limits, Verdict   (CONTRACT, T2)
  guards.py       test-split lock + as_of time guard                    (all tracks)
  harness/interface.py   Investigator / ToolBox protocols                (CONTRACT, frozen week 4)
  eval/runlog.py         run-log schema                                  (T2 + T4, week 1)
notebooks/    exploration only, never used for results
prereg/       pre-registered thresholds and Holm order (committed before the final run)
scripts/      entry points
tasks/        per-track checklists
tests/        pytest; CI runs the CPU-only tests
```

## Rules (full list in `tracks.md` section 8)

1. **Test split is locked.** Any code that reads test data calls `ftm.guards.require_final_run()`. `FTM_FINAL_RUN=1` is set only by `scripts/run_final.py` at G5.
2. **Time guard.** Every tool takes `as_of` and filters `timestamp <= as_of` (`ftm.guards.enforce_as_of`).
3. **No raw rows leave a partner bank**; **LLM text never enters the decision**; **LLM sees aliased IDs only**; **every system goes through the same gate and accountant.**
4. **Every result comes from a committed script + config.** Fixed seeds, at least 3 for anything trained. Log the commit hash to MLflow.
5. **Contracts** (`ftm/types.py`, `ftm/harness/interface.py`, `ftm/eval/runlog.py`, week-1 data schemas) change only by agreement of the owners involved; after the week-4 harness freeze, all four.
6. **Branches:** `t<N>/<short-topic>`, PR into `main`, reviewed by your pair partner within two working days. CI must pass (ruff + CPU tests).
7. **Don't commit** data, weights, or CSV/Parquet. Large derived artifacts are shared via the agreed cloud folder with a hash manifest (decision 6 in `plan.md` section 16).
8. **Trained weights are never released** (`plan.md` section 10).
