# Follow the Money Without Moving the Data

Team **Cache me if you can** (LMA Group Project 2026): Abhishek, Bhavya, Shreyas Deb, Akshat Srivastava.

A bank sees only its own customers, but money launderers move money through many banks. In this project the
home bank's LLM agent (Gemma 4 12B on vLLM, orchestrated with LangGraph) looks at an anti-money-laundering (AML)
alert and asks partner banks a few small, typed questions. Each question is written in a small question language,
passes a compliance gate and a privacy accountant, and is answered with one differentially private (DP) number.
Every question type was back-tested on historical alerts before the final run, so each answer comes with a measured
likelihood ratio. A fitted fusion formula turns the one-bank score and the answers into the final probability.
LLM text never enters the decision. We test this on IBM AMLworld (HI-Small, HI-Medium) against one-bank and
see-everything detection models, raw data sharing, non-LLM question planners and external LLM agents, with
pre-registered hypotheses and a single frozen test run.

**Reports:** final report [`docs/report/final/report.pdf`](docs/report/final/report.pdf) ·
slides [`docs/presentation/slides.pdf`](docs/presentation/slides.pdf) ·
test tables [`docs/results/test_analysis/`](docs/results/test_analysis/) ·
results summary [`docs/RESULTS.md`](docs/RESULTS.md) ·
doc index [`docs/README.md`](docs/README.md)

## Contents

1. [Headline results](#headline-results)
2. [How the system works](#how-the-system-works)
3. [Repository map](#repository-map)
4. [Install](#install)
5. [Data](#data)
6. [Tests](#tests)
7. [Quick look without data or GPU](#quick-look-without-data-or-gpu)
8. [Validation pipeline](#validation-pipeline)
9. [The frozen final run](#the-frozen-final-run)
10. [Regenerating the analysis](#regenerating-the-analysis)
11. [Hardware](#hardware)
12. [Known deviations](#known-deviations)
13. [Rules the code enforces](#rules-the-code-enforces)
14. [Team](#team)
15. [Licences and external code](#licences-and-external-code)

## Headline results

Test split of HI-Medium, the 771 askable alerts (alerts with a partner bank the home bank can ask), privacy budget
ε = 2 per alert unless stated. Brier score: lower is better. Gap closed = (B_one_bank − B_system) / (B_one_bank − B_see_everything),
so 0 means no better than the home bank alone and 1 means as good as a model that sees every bank. 95% cluster-bootstrap CIs.

| System | Brier | Gap closed [95% CI] |
|---|---|---|
| One bank only (GFP + LightGBM score, p0) | 0.1668 | 0 (reference) |
| GFP see-everything (upper limit, not private) | 0.0966 | 1 (upper limit) |
| **Ours** (LLM composes questions, ε = 2) | **0.1673** | **−0.007 [−0.038, 0.026]** |
| Ours, ε = 1 | 0.1666 | |
| Ours, ε = 4 | 0.1679 | |
| Raw sharing with 3 banks (no privacy) | 0.1640 | 0.040 [−0.107, 0.161] |
| Non-LLM planner: random | 0.1698 | |
| Non-LLM planner: fixed order | 0.1694 | |
| Non-LLM planner: expected information gain (EIG) | 0.1823 | |
| Non-LLM planner: learned policy | 0.1814 | |
| Non-LLM planner: enumerator | 0.1708 | |
| BED-LLM (best external LLM agent, frozen on validation) | 0.1956 | −0.410 |

Cost of our system: 2.4 LLM calls, 12.9k tokens and 29 GPU-seconds per alert.

Pre-registered hypotheses (Holm-corrected within the primary family):

| Hypothesis | Result |
|---|---|
| H1: questions close ≥ 30% of the gap | −0.007, **not supported** |
| H2: questions get ≥ 70% of the benefit of raw sharing | **not supported** |
| H3: LLM composing ≥ best simple planner and enumerator, and > fixed list | **not supported** (p_holm 0.33) |
| H4: back-testing helps, and < 10% of accepted questions are useless later | −0.005 [−0.010, −0.000], p_holm 0.072, **not supported** (the FDR audit fails) |
| H9: ours ≥ best ported external agent | ours minus BED-LLM, gap closed: 0.266, p_holm 0.004, **supported** |
| H7: formula beats the LLM deciding | **not supported** |
| H5, H6, H8, H10 | not run on test (reduced final run) |

Ablations on the subset (208 askable alerts), Brier of the ablation minus the full system (positive = the ablation is worse):

| Ablation | Δ Brier [95% CI] |
|---|---|
| Fixed question list instead of composed questions | +0.0080 [0.0031, 0.0131] |
| No back-testing | +0.0049 [0.0003, 0.0099] |
| Act only (no planning step) | +0.0016 (not significant) |
| LLM makes the final decision | +0.0001 (not significant) |

In plain words: on test, the private questions did not improve on the home bank's own model, and raw sharing did
not improve on it by a clear margin either. Within the system, composed questions and back-testing both matter (the
fixed-list and no-back-test ablations are clearly worse), and our system has a much lower Brier score than the best
external LLM agent.
Final run: freeze hash `b094561b57d2f8e3da6bd6945e8146904c8baec4b44910c633fce0b04722b97f`, run id `g5`,
44 GPU-hours on the RTX 5070 Ti. More detail: [`docs/RESULTS.md`](docs/RESULTS.md).

## How the system works

```
HOME BANK (sees only its own customers' transactions)
  alert intake      one-bank detection model (GFP + LightGBM) gives p0; a conformal filter keeps the unsure alerts
  evidence view     time-respecting subgraph around the alert (edge scorer, top 40 edges), account ids aliased
  planner (LLM)     Gemma 4 12B, LangGraph graph: prepare -> check -> plan -> act -> fuse -> check ... -> decide
                    the LLM selects a listed question or composes a new one in the question language (DSL)
  gate + accountant compliance rules, DP budget per alert (ε) and per account
        |  typed question
        v
PARTNER BANKS (simulated, independent)  run the question on their own ledger, return one DP-noised value
        |  noisy typed answer
        v
  back-tested LR    each question type has a likelihood ratio measured on training alerts (e-BH FDR control)
  fusion formula    p0-conditional fusion fitted on train / validation, no LLM text
  decision          one calibrated threshold: escalate or not
```

Invariants (checked in code and tests): no transaction record leaves a partner bank, only single noisy values do;
LLM text never enters the decision; tools only see data up to the alert time; anything an LLM reads uses aliased
account ids; every system, ours or external, goes through the same gate and accountant. The full design is in
[`plan.md`](plan.md) (research questions, hypotheses, components C1–C18, experiments E1–E18).

## Repository map

| Path | What is there |
|---|---|
| `ftm/` | The Python package. One subpackage per component; every module docstring names its owner and spec section. |
| `ftm/data`, `baselines`, `intake`, `retrieval` | Data loading and splits, detection models (GFP, Multi-GNN, MEGA-GNN, FraudGT, DP-FedAvg, layer-wise), conformal alert intake, evidence views |
| `ftm/banks`, `dsl`, `privacy`, `backtest`, `planners`, `orchestrator`, `harness`, `agent` | Partner banks, question language, DP accountant, back-testing engine, non-LLM planners, raw sharing, the shared agent harness |
| `ftm/graph`, `llm`, `library`, `fastmodel`, `external` | LangGraph system graph, our LLM planner, LLM client and budget, strategy library, System 1 fast model, external agents (Group C/D) |
| `ftm/compliance`, `fusion`, `decision`, `safety`, `eval`, `final` | Compliance gate, fusion, threshold, injection tests, metrics and final analysis, the final-run machinery (freeze, preflight, plan, runner) |
| `scripts/` | Command-line entry points. `run_final.py` (final run), `final_analysis.py` (analysis), `run_llm_agent.py` (any LLM run), `scripts/ada/` (HPC cluster jobs) |
| `configs/` | YAML/JSON settings: `data`, `banks`, `privacy`, `questions`, `llm`, `fusion`, `decision`, `compliance`, `intake`, `final` (final-run jobs), `t1_final` (frozen T1 settings), `analysis`, `experiments` |
| `prereg/` | Pre-registration: hypotheses, E18 decision rule, external agent list, subset rule, and `FREEZE.json` (the frozen hashes). Do not edit. |
| `tests/` | pytest suite (CPU tests run in CI) |
| `docs/` | Notes, results and reports; start at [`docs/README.md`](docs/README.md) |
| `plan.md` | The research design (cited by section number in code docstrings, e.g. "plan.md 6.11") |
| `tracks.md` | The build-phase work plan, one track per person (also cited by section in docstrings) |
| `data/`, `outputs/`, `models/` | Not in git: datasets, run outputs, trained weights (weights are never released) |

## Install

Python 3.11 or newer (the team used 3.12). The base install is what CI runs; heavier parts are extras in
`pyproject.toml`.

```bash
git clone https://github.com/BhavyaV05/Transaction-Thieves.git && cd Transaction-Thieves
python3.12 -m venv .venv && source .venv/bin/activate       # or: uv venv --python 3.12 .venv
pip install -e ".[dev]"                                     # base + pytest, ruff, langgraph (what CI runs)
pip install -e ".[dev,gbt,llm,rag]"                         # the CPU pipeline and the LLM client side
```

| Extra | Contents | Needed for |
|---|---|---|
| `dev` | pytest, ruff, langgraph 1.2.19, rich | tests, every run |
| `gbt` | LightGBM, XGBoost, scikit-learn | the GFP + LightGBM detection model and the other scikit-learn models |
| `llm` | OpenAI client, DSPy 3.4, AG2, langgraph | every LLM run (client side) |
| `rag` | rank-bm25, sentence-transformers | regulatory RAG, retrieval bench |
| `gnn` | torch, torch_geometric, pcst_fast | GNN baselines, edge scorer, evidence views (install a CUDA build of PyTorch first) |
| `enc` | torch, transformers, scikit-learn | ModernBERT/DeBERTa routers (System 1 baselines) |
| `fed` | Flower, Opacus | DP-FedAvg baseline |
| `serve` | vLLM, Unsloth | the serving machine only |

Other environments we used:

- **Serving:** a separate venv with `vllm==0.31.0` (we used `~/projects/vllm-venv`; `scripts/t3_rerun.sh` puts it on
  `PATH`). Start a server with `python scripts/serve_llm.py --gen-key` once, then
  `python scripts/serve_llm.py --model gemma4-12b --backend native`. Model revisions are pinned in
  `configs/llm/serving.yaml`; setup details in [`docs/hardware.md`](docs/hardware.md) section 5.
- **QLoRA training (E16 only):** a separate venv with Unsloth (`TRAIN_PY`, default `~/projects/train-venv/bin/python`).
- **snapml** (IBM Graph Feature Preprocessor, used by GFP): `pip install snapml`. Not in any extra because of its
  licence (see [Licences](#licences-and-external-code)).
- **External detection models** (Multi-GNN, MEGA-GNN, FraudGT) run from their own checkouts with our patches in
  `ftm/baselines/*.patch`; see [`ftm/baselines/README.md`](ftm/baselines/README.md).

Clients find the LLM server through `FTM_LLM_ENDPOINT`, `FTM_LLM_API_KEY` and `FTM_LLM_MODEL` (the key lives in a
gitignored `.env`).

## Data

IBM AMLworld from Kaggle (CDLA-Sharing 1.0). You need `kaggle >= 1.8` and a Kaggle token.

```bash
for ds in Small Medium; do
  d=data/raw/hi_$(echo $ds | tr A-Z a-z)
  for f in HI-${ds}_Trans.csv HI-${ds}_accounts.csv HI-${ds}_Patterns.txt; do
    python -m kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f $f -p $d
  done
done
python scripts/prepare_data.py --data hi_small     # typed Parquet, bank views, aliases, manifest (about 5 s)
python scripts/prepare_data.py --data hi_medium    # same code, streaming
```

(Unzip a file if Kaggle delivers it zipped.) Splits are by time and stored in every row: HI-Medium days 1–9 train,
10–12 validation, 13–28 test (13–16 is the headline "bulk"; the sparse tail 17–28 is reported separately).
Schemas, keys and the split rules are in [`ftm/data/README.md`](ftm/data/README.md).

**The test split is locked.** Every reader that can return test rows calls `ftm.guards.require_final_run()`,
which needs `FTM_FINAL_RUN=1`. Only `scripts/run_final.py` sets it (and the caller of the test analysis, below).

## Tests

```bash
ruff check .
pytest -m "not gpu and not slow"     # the CPU suite; this is what CI runs (.github/workflows/ci.yml)
```

The tests use synthetic worlds, a mock LLM and temporary directories; they need no data and no GPU.

## Quick look without data or GPU

```bash
# one planner run on synthetic alerts with the mock LLM
python scripts/run_llm_agent.py --system planner --mode compose --alerts synthetic --n 5 --llm mock --toolbox dev
# the whole final-run flow on a synthetic world (mock LLM; writes outputs/final_synthetic/; sets nothing)
python scripts/run_final.py --dry-run
# every table and figure of the analysis on a synthetic world (never a result)
python scripts/final_analysis.py --synthetic --outdir outputs/analysis/synthetic
```

With data, `scripts/demo.py` shows one validation alert end to end; see [`docs/demo.md`](docs/demo.md).

## Validation pipeline

Everything below uses train and validation data only. Steps are in dependency order; each links to the note with the
full detail. HI-Medium is the headline dataset.

```bash
# 1. one-bank and see-everything detection scores (p0), then the conformal alert sets (T1)
python scripts/train_gfp.py --data hi_medium --mode one_bank
python scripts/train_gfp.py --data hi_medium --mode see_everything
python scripts/make_alerts.py --data hi_medium
python scripts/recal_p0.py --data hi_medium                   # p0 v2 (recalibrated), used by the final system
# 2. back-test every question type on training alerts (T2): add --shard i/n to split the work over processes, then assemble
python scripts/backtest.py --data hi_medium --split train --eps 0.25 0.5 1 2 --questions expert --time-blocks 4
python scripts/backtest.py --data hi_medium --split train --eps 0.25 0.5 1 2 --questions expert --time-blocks 4 --assemble
#    the same questions on every askable validation alert
python scripts/backtest.py --data hi_medium --split val --eps 0.25 0.5 1 2 --questions expert
# 3. fit the fusion and the threshold on train + val fit/cal; writes the frozen configs/fusion/headroom_v1* files
#    (it reads the two back-test namespaces named in ftm/fusion/headroom.py, TRAIN_NS and VAL_NS)
python scripts/t3_headroom.py select
# 4. every LLM stage: teacher runs, library, all systems on val fit/cal/eval, E15, E18, RAG, System 1. The script starts and
#    stops the vLLM server itself (EXTERNAL_SERVER=1 uses one that is already running at FTM_LLM_ENDPOINT)
DRY_RUN=1 scripts/t3_rerun.sh all                              # prints every command; drop DRY_RUN to run
# 5. non-LLM arms on the same 160 askable val eval alerts, package all logs, run the analysis on validation
python scripts/final_non_llm.py --arm eig --split val --alert-ids <val ids.txt> --out outputs/analysis/val_runs_nonllm/eig --data hi_medium
python scripts/build_val_run.py --llm outputs/t3_val_v3/eval --nonllm outputs/analysis/val_runs_nonllm --out outputs/analysis/val_run_v3
python scripts/final_analysis.py --split val --run outputs/analysis/val_run_v3 --inputs configs/analysis/inputs_val.yaml --outdir outputs/analysis/val
```

Notes: step 1 also has the GNN, federated and layer-wise baselines ([`docs/t1_baselines_val.md`](docs/t1_baselines_val.md),
[`ftm/baselines/README.md`](ftm/baselines/README.md)); the back-test namespaces the frozen files were built from are
listed in [`docs/t3_headroom.md`](docs/t3_headroom.md); the LLM stages are explained in [`docs/t3_rerun.md`](docs/t3_rerun.md);
`python scripts/final_non_llm.py --list` lists the non-LLM arms. The validation analysis output that the test analysis
takes as fixed is committed in [`docs/results/val_analysis/`](docs/results/val_analysis/) (`fixed_on_val.json`).

## The frozen final run

`scripts/run_final.py` is the only code that may set `FTM_FINAL_RUN=1`. Its docstring and
[`docs/t4_final_run.md`](docs/t4_final_run.md) are the runbook; the job list is `configs/final/jobs.yaml`.

```bash
python scripts/run_final.py --plan          # job list, hours, machines and the preflight report; sets nothing, reads no data
python scripts/run_final.py --dry-run       # the whole script on the synthetic world with a mock LLM; writes outputs/final_synthetic/
# the real run: same --run-id on every machine, resumable after a crash
python scripts/run_final.py --run-id g5 --confirm-final <combined hash> --machine s1
python scripts/run_final.py --run-id g5 --confirm-final <combined hash> --machine s3 --endpoint http://localhost:8000/v1
python scripts/run_final.py --run-id g5 --confirm-final <combined hash> --machine cpu
# gather the job directories of all machines into outputs/final/g5/, then
python scripts/run_final.py --run-id g5 --confirm-final <combined hash> --manifest-only
```

- `<combined hash>` is `combined_sha256` in `prereg/FREEZE.json`: for g5 it was
  `b094561b57d2f8e3da6bd6945e8146904c8baec4b44910c633fce0b04722b97f` (`python scripts/freeze_prereg.py --check` verifies it).
- Before anything runs, the preflight requires: hashes that match `prereg/FREEZE.json`, a clean working tree, the freeze
  commit in the history of HEAD, the recorded TA status (here: waived), the E18 backbone decision, no `fix_on_val` left,
  the G4 external-agent file, the subset rule, T1's own check (`scripts/t1_final_check.py`), and `--confirm-final` equal
  to the combined hash. Any failure exits with code 2 and sets nothing.
- Then, in order: T1's `--final` steps that build the test alert set, the frozen 1,000-alert subset (drawn first),
  T1's remaining steps, then the jobs (our system and ablations, external agents, detection and non-LLM arms).
- `--machine s3|s1|cpu` picks the jobs of one machine class (S3 = the 5070 Ti, S1 = the 3070 Ti laptop);
  `--only <job ids or patterns>` selects jobs (the reduced g5 selection is in [`docs/t4_reduced_final.md`](docs/t4_reduced_final.md));
  `--shard i/n` and `--max-jobs` split the work.
- `--manifest-only` runs nothing: it checks parity between machines (alert-set hash, seeds, limits, backbone, commit)
  and writes `MANIFEST.json`, or `MANIFEST.partial.json` while arms are missing. The analysis reads only the manifest.
- The runner never prints a label or a Brier score.

## Regenerating the analysis

The test analysis reads the manifest of the run, joins the test labels, and writes every table, figure and
`numbers.json` to `outputs/final/g5/analysis/`. It needs `FTM_FINAL_RUN=1` set by the caller (the script never sets it)
and the values fixed on validation:

```bash
FTM_FINAL_RUN=1 python scripts/final_analysis.py --split test --run outputs/final/g5 \
    --inputs configs/analysis/inputs_test.yaml --fixed outputs/analysis/val/fixed_on_val.json
```

`--fixed` can also point at the committed copy, `docs/results/val_analysis/fixed_on_val.json`. The tables of our
run are archived in [`docs/results/test_analysis/`](docs/results/test_analysis/). Other options: `--no-figures`,
`--n-boot`, `--seed`, `--outdir`; `python scripts/final_analysis.py --help` lists all. The analysis pipeline is
described in [`docs/t4_final_analysis.md`](docs/t4_final_analysis.md).

## Hardware

| Machine | Used for |
|---|---|
| RTX 5070 Ti PC (16 GB VRAM), Ryzen 9 7900X, 32 GB RAM, Fedora 43, vLLM 0.31.0 | Reference machine: LLM serving and every LLM run; the g5 LLM jobs (44 GPU-hours) |
| RTX 3070 Ti laptop (8 GB VRAM), i7-12650H | Detection models; T1's `--final` steps of g5 |
| RTX 3050 Ti laptop (4 GB), and a CPU-only laptop | Back-testing, non-LLM arms, analysis, tests |
| ada HPC cluster, RTX 2080 Ti nodes | Some validation stages (RAG bench, E15, graph-vs-legacy parity); recipe in [`docs/t3_ada.md`](docs/t3_ada.md) |

Timing numbers (latency, GPU-seconds) are measured on the 5070 Ti only. Details: [`docs/hardware.md`](docs/hardware.md).

## Known deviations

Every one is recorded in [`docs/t4_reduced_final.md`](docs/t4_reduced_final.md) or [`docs/t1_final_run.md`](docs/t1_final_run.md).
Read them before you interpret a test table.

1. **Reduced final run.** The full plan needed about 188 GPU-hours; a subset of the arms ran. Arms that did not run
   are reported as "not run" in every table, never replaced by validation numbers. Only one G4 external agent
   (BED-LLM) instead of two to three.
2. **No TA hash.** The pre-registration hash was sent to nobody; it is fixed by the git commit of `prereg/FREEZE.json`
   only (`ta_waived_hash`).
3. **Equivalence margin** fixed at 0.01 Brier instead of the derived value; the H9 margin (0.166 gap closed) is a rule
   computed on validation.
4. **Upper limit.** Only time-respecting see-everything scores count as the upper limit (GFP see-everything). The
   stored HI-Medium GNN see-everything scores use future edges and are baseline rows only.
5. **p0 v2.** The tolerance of the recalibration rule was set after validation eval numbers existed; the map is frozen
   with a hash.
6. **FraudGT** was scored on test once, early (7 Nov), outside `run_final.py` and before the freeze; nothing was retrained.
   It is HI-Small only, so it is a baseline row, not an upper limit.
7. **Layer-wise exchange** failed its validation-reproduction check in g5 (rank correlations 0.960, 0.931, 0.950 against a
   0.95 gate, from non-deterministic GPU scatter). HI-Small test scores are reported with that note; HI-Medium was dropped.
8. **Crash fixes during g5.** A few bugs that stopped jobs (dtype of edge scores, the p0 v2 test alerts, the edge-scorer
   `--final` print, a layer-wise key error, the manifest check of T1 inputs) were fixed and merged while g5 ran. None
   changed a frozen file; the preflight would have refused it.
9. **Validation-only arms.** The RAG bench and E15 ran on the ada cluster, whose data copy lacks two bank views
   (453 instead of 460 askable alerts); compare them only with each other. The `*_adapted` external arms are
   exploratory.

## Rules the code enforces

1. The test split is locked behind `FTM_FINAL_RUN=1`, which only `scripts/run_final.py` sets (`tests/test_static_checks.py` checks this).
2. Every tool takes `as_of` and filters `timestamp <= as_of` (`ftm.guards.enforce_as_of`).
3. No raw rows leave a partner bank; LLM text never enters the decision; LLMs see aliased ids only; every system goes
   through the same gate and accountant.
4. Every result comes from a committed script and config with fixed seeds.
5. Data, weights and CSV/Parquet are never committed (`.gitignore`); trained weights are never released.

## Team

| Track | Person | Owned |
|---|---|---|
| T1 | Abhishek | Data, detection models (GFP, GNNs, FraudGT, federated, layer-wise), conformal intake, retrieval |
| T2 | Bhavya | Partner banks, question language, DP accountant, back-testing, non-LLM planners, raw sharing, harness |
| T3 | Akshat Srivastava | LLM planner, serving, fine-tuning, strategy library, System 1, agent frameworks, LangGraph rewrite |
| T4 | Shreyas Deb | Compliance gate, fusion, decision, safety, evaluation and statistics, pre-registration, final run, report |

## Licences and external code

There is no licence file for this repository's own code yet; ask the team before reusing it.

- **Data:** IBM AMLworld (Kaggle), CDLA-Sharing 1.0. Not redistributed here.
- **Backbone:** Gemma 4 12B QAT checkpoint (`google/gemma-4-12B-it-qat-w4a16-ct`, revision pinned in
  `configs/llm/serving.yaml`). Qwen3 models for E16 / System 1. Model weights are downloaded, not stored here.
- **snapml** (GFP): IBM proprietary licence. Used as an installed library for research, never vendored, not a
  base dependency ([`docs/t1_week0.md`](docs/t1_week0.md) section 4).
- **Multi-GNN, MEGA-GNN** (Apache-2.0) and **FraudGT** (no licence): run from their own checkouts; only our patches
  (`ftm/baselines/*_t1.patch`) are in this repo, no copied code.
- **Sharma et al.** (MIT): a faithful port of the official code, with the original notice in `ftm/external/sharma.py`.
- **ReAct, Reflexion, ReWOO, LATS, LLMCompiler** (MIT repositories), **debate, UoT** (no licence), **MediQ** (CC-BY-4.0),
  **Torres, Co-Investigator, FinFRE-RAG, BED-LLM**: reimplemented from the papers; no code copied. DSPy and AG2 are used
  as libraries.
- Per-system licence and code status: [`docs/external_code_check.md`](docs/external_code_check.md); fidelity of each
  baseline: [`docs/baseline_fidelity.md`](docs/baseline_fidelity.md).
