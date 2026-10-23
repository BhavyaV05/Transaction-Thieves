# T3 week 0: serving, model cards, benchmark, machine report

Owner: T3 (Akshat). Covers the week-0 row of `tracks.md` §5.8 (serving smoke test, recipe, conversion benchmark, report specs) and the model-card check that `plan.md` §6.11 asks T3 to do in week 0.

**Status (2026-10-23).** The scripts, the recipe and the model-card check are done. The GPU smoke test ran on native Linux (no Docker, no WSL) on the reference PC: **Qwen3-8B-AWQ and Qwen3-4B-AWQ start on sm_120 and are benchmarked (section 4); gpt-oss-20b does not fit in 16 GB while the Linux desktop is running (needs about 1 GiB more), so it has no number.** Section 4 lists the one config change (`qwen3-8b` `gpu_memory_utilization` 0.90 to 0.85). The week-0 "load each E18 candidate at the context cap" check is not done in week 0: it moves into the E18 sweep itself (`docs/t3_e18.md`, per-candidate procedure).

## 1. Machine report (reference machine)

| | T3 Akshat |
|---|---|
| GPU / VRAM | NVIDIA GeForce RTX 5070 Ti (Blackwell, sm_120) / 16,303 MiB, driver 580.178.04 on Linux (CUDA 13.0); the same PC ran 610.88 under Windows |
| RAM | 31.1 GiB usable (32 GB installed) |
| CPU | AMD Ryzen 9 7900X, 12 cores / 24 threads |
| Free disk | / (btrfs, 463 GB): 126 GB free after downloading the three profiles (22 GB in `~/hf_cache`; weights go there via `FTM_HF_HOME`) |
| OS | Fedora Linux 43 (kernel 7.2.10), KDE/Wayland. Dual boot on the same physical PC as the Windows 11 reference machine, so timings here are the reference timings (factor 1.0) |
| WSL2 / Docker | Not used. Native vLLM 0.31.0 (torch 2.13.0+cu130) in a separate uv venv, `~/projects/vllm-venv`; the Windows install has Docker Desktop 29.6.2 and WSL2 but they were not needed |
| Python | 3.12.12 in `.venv` (uv); CI uses 3.11 |
| Benchmark | Reference machine (factor 1.0 by definition). Measured, qwen3-8b (vLLM 0.31.0): prefill 5,138 tok/s, decode 117.4 tok/s, batch (4 parallel) 112.6 tok/s (section 4) |

## 2. Model-card check (`plan.md` §6.11)

Checked on 2026-10-19 against the Hugging Face API (`/api/models/<repo>` for licence and commit, `config.json` for context length). The revisions are pinned in `configs/llm/serving.yaml`.

| Role | Repo | Revision (pinned) | Licence | Native context | Quantisation | Size on disk (approx.) |
|---|---|---|---|---|---|---|
| Planner / Group D default | `Qwen/Qwen3-8B-AWQ` | `4da05a8edb55` | Apache-2.0 | 40,960 | AWQ 4-bit | ≈6 GB |
| Fast model base (C13) | `Qwen/Qwen3-4B-AWQ` | `74d4bd2bd4bf` | Apache-2.0 | 40,960 | AWQ 4-bit | ≈2.7 GB |
| Teacher candidate | `openai/gpt-oss-20b` | `6cee5e81ee83` | Apache-2.0 | 131,072 (YaRN) | MXFP4 (native) | ≈13 GB |
| QLoRA base (planner SFT) | `Qwen/Qwen3-8B` | `b968826d9c46` | Apache-2.0 | 40,960 | bf16 (QLoRA loads in 4-bit) | ≈16 GB |
| QLoRA base (fast model) | `Qwen/Qwen3-4B` | `1cfa9a720891` | Apache-2.0 | 40,960 | bf16 | ≈8 GB |

E18 backbone-sweep candidates (not downloaded; the licence and context were checked so the sweep can be planned. The load check at the 8,192 context cap is not part of week 0: it happens in the sweep, one candidate at a time, see `docs/t3_e18.md`):

| Model | Repo | Licence | Native context | Note |
|---|---|---|---|---|
| Qwen3-14B | `Qwen/Qwen3-14B-AWQ` | Apache-2.0 | 40,960 | AWQ ≈10 GB; fits 16 GB at our 8k context |
| Gemma 4 12B | `google/gemma-4-12B-it` | Apache-2.0 | 262,144 | Not gated |
| Gemma 4 26B A4B | `google/gemma-4-26B-A4B-it` | Apache-2.0 | 262,144 | MoE; 4-bit quant needed for 16 GB |
| Gemma 4 31B | `google/gemma-4-31B-it` | Apache-2.0 | 262,144 | "If time"; Q3 GGUF only |
| Qwen3.8-27B | `unsloth/Qwen3.8-27B-GGUF` (`Qwen/Qwen3.8-27B` base) | Apache-2.0 | 262,144 | `UD-Q3_K_XL` is 13.1 GB, which leaves ≈2.5 GB for the KV cache at 8k context; tight, check in the sweep. Runs on llama.cpp, not vLLM |
| Phi-4-reasoning | `microsoft/Phi-4-reasoning` | **MIT** | 32,768 | The only non-Apache licence; MIT is compatible |

All are open-weight and allowed for research use. None conflicts with plan §10 (we never release trained weights).

**vLLM version.** PyPI has 0.32.0 (released 2026-10-18), 0.31.0 (2026-10-05) and 0.30.0 (2026-09-22). We pin **0.31.0** (`vllm/vllm-openai:v0.31.0`, about 10 GB compressed). 0.32.0 is a day old; move to it only together with Abhishek's server, because both servers must run the same version (`tracks.md` §1.2).

## 3. What is in the repo

| File | What |
|---|---|
| `scripts/serve_llm.py` | Starts vLLM for one profile (Docker on Windows, native on Linux/WSL). `--dry-run` prints the command; `--gen-key` writes `FTM_LLM_API_KEY` to `.env` (gitignored). The key reaches vLLM through the `VLLM_API_KEY` environment variable, never the command line. Binding is restricted to loopback, private and tailnet (100.64/10) addresses; `0.0.0.0` and public IPs are refused |
| `configs/llm/serving.yaml` | Pinned vLLM version, image, model revisions and server flags. Every serving machine uses this file unchanged |
| `scripts/bench_llm.py` | The conversion benchmark (`docs/hardware.md` §4) |
| `ftm/llm/client.py` | Stdlib-only OpenAI-compatible client (no `openai` package, so CI and CPU machines can import it) |
| `tests/test_serving.py` | Host safety, key handling, pinned revisions, the same flags on Docker and native, benchmark maths on a fake server |

Server flags that matter for parity: `--seed 0`, `--generation-config vllm` (ignore each model's default sampling; the client always sends `temperature`), `--max-model-len 8192` (the 6,000-token prompt cap from `Limits.max_tokens` plus room for the answer), `--enable-prefix-caching`, and `--reasoning-parser qwen3` for Qwen3, so thinking text comes back separately from the JSON answer.

## 4. Smoke test: steps and results (run 2026-10-23, native Linux)

Steps that ran, in order (the recipe in `docs/hardware.md` §5 is the short version). The same dual-boot PC is the Windows reference machine, so these numbers are the reference numbers.

1. `git checkout -b t3/week0-smoke-test`; `export FTM_HF_HOME=$HOME/hf_cache`; `export PATH=$HOME/projects/vllm-venv/bin:$PATH` (so `vllm` resolves for `--backend native`).
2. `.venv/bin/python scripts/serve_llm.py --gen-key`, then `set -a && source .env && set +a`.
3. Download each model at its pinned revision before timing the load: `HF_HOME=$HOME/hf_cache ~/projects/vllm-venv/bin/hf download <repo> --revision <rev>`. For gpt-oss-20b add `--exclude "original/*" --exclude "metal/*"`: the repo also holds `original/` and `metal/` copies of the weights that vLLM does not use (about 40 GB in total). Sizes on disk: Qwen3-8B-AWQ 5.7 GB, Qwen3-4B-AWQ 2.5 GB, gpt-oss-20b 13 GB.
4. For each of `qwen3-8b`, `qwen3-4b`, `gpt-oss-20b`: `serve_llm.py --model <profile> --dry-run` (binds 127.0.0.1, flags as in `serving.yaml`), start the server (`--backend native` is the default on Linux), wait for `Application startup complete` (load time = launch to that line, weights already on disk), sample VRAM with `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500`, then `FTM_LLM_ENDPOINT=http://127.0.0.1:8000/v1 python scripts/bench_llm.py --machine t3-5070ti` in a second shell. Stop the server before the next profile.
5. The `qwen3-8b` JSON (`outputs/bench/t3-5070ti_qwen3-8b.json`, not committed) is the `--reference` for the other machines.

Idle VRAM before any server: 1,329 to 1,600 MiB (KDE/Wayland desktop; it drifted between the two values, 1,600 MiB at every server start). Peak VRAM below is the maximum of the sampler (absolute, including the desktop) and the same minus 1,600 MiB.

| Profile | Starts on sm_120? | Load time | Peak VRAM (abs / over idle) | Prefill tok/s | Decode tok/s | Batch tok/s (4 parallel) | Notes |
|---|---|---|---|---|---|---|---|
| qwen3-8b | Yes. AWQ via `MarlinLinearKernel`, `FLASH_ATTN` attention | 159 s (cold: first `torch.compile` and CUDA-graph capture) | 15,020 / 13,420 MiB | 5,138 | 117.4 | 112.6 | `gpu_memory_utilization` 0.90 fails, 0.85 works (below). KV cache 40,496 tokens |
| qwen3-4b | Yes. Same kernels | 102 s | 10,938 / 9,338 MiB | 8,412 | 172.1 | 167.0 | Profile unchanged (0.60). KV cache 38,560 tokens |
| gpt-oss-20b | Loads the MXFP4 MoE with the `MARLIN` backend and `FLASHINFER` attention, then **runs out of memory while allocating weights** | n/a | 14,894 / 13,294 MiB at the failure | n/a | n/a | n/a | Fails with the desktop running; see below. No number |

The benchmark prompt is 7,498 tokens with the Qwen3 tokenizer (the docstring in `bench_llm.py` says about 3,000 and `Limits.max_tokens` is 6,000), so the prefill figure is for a long prompt; with 256 forced output tokens it still fits the 8,192 context. `bench_version` was left at 1 so results stay comparable. No WARNING from `bench_llm.py` (`short_outputs` 0). vLLM logs `FlashInfer top-p/top-k sampling unavailable: SM 12.x requires CUDA >= 12.9; falling back` and `Failed to get device capability: SM 12.x requires CUDA >= 12.9`; both are harmless here (the torch build is cu130 and the kernels ran).

Thinking stays off: the client sends `enable_thinking: false` through `chat_template_kwargs` (`ftm/fastmodel/model.py`, `scripts/run_llm_agent.py`); `bench_llm.py` itself sends no `reasoning_effort` and no thinking switch, so the Qwen benchmark ran with the Qwen3 default chat template, which has thinking on (with 256 forced tokens the decode rate is unaffected, but the benchmark is not the thinking-off setting the agents use). Not changed here; decide whether `bench_llm.py` should pass `enable_thinking: false` (that would bump `bench_version`).

**Errors and fixes.**

| Profile | Error | Fix |
|---|---|---|
| qwen3-8b | `ValueError: Free memory on device cuda:0 (13.68/15.47 GiB) on startup is less than desired GPU memory utilization (0.9, 13.92 GiB)`. Twice, identical. vLLM sees 15.47 GiB total and the desktop holds about 1.6 GB, so 0.90 is 0.25 GiB too much | `configs/llm/serving.yaml`, `qwen3-8b`: `gpu_memory_utilization` 0.90 to 0.85 (13.15 GiB). Only the KV-cache budget changes (40,496 tokens, 4.9x concurrency at 8,192). Not a behaviour change, but the other serving machines should use the same value for parity |
| gpt-oss-20b | Attempt 1 (0.92): the same free-memory `ValueError` (0.92 = 14.23 GiB, 13.68 free). Attempt 2 (0.88, the most the free-memory check allows): `torch.OutOfMemoryError ... Tried to allocate 1.08 GiB ... 817 MiB free`, raised in `ParallelLMHead` creation, with 13.10 GiB already in use. Attempt 3 (0.88 plus `PYTORCH_ALLOC_CONF=expandable_segments:True`): the same OOM, 943 MiB free | None worked, so `serving.yaml` is unchanged for this profile (0.92). It is a weight-allocation OOM, so the fallback table's `--max-model-len 6656` cannot help, and `--gpu-memory-utilization 0.95` is refused by the free-memory check on this machine. The model needs roughly 1 GiB more than the GPU has left next to the desktop. Not tried: running the GPU headless (stop the display manager, SSH in), which would free the desktop's 1.3 to 1.6 GB; that is the next thing to try, otherwise Qwen3-14B as teacher (`tracks.md` §5.10) |

Download and run time: about 10 minutes of server time in total (8B 253 s including two failed starts, 4B 165 s, gpt-oss 3 failed starts about 75 s), 0.17 GPU-hours, logged in `GPU_CALENDAR.md`.

**Windows (Docker) alternative.** The Windows steps are kept for completeness: move Docker's disk image off C: (Docker Desktop, Settings, Resources, Advanced, Disk image location), `setx FTM_HF_HOME E:\hf_cache`, check `docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi`, then `python scripts/serve_llm.py --model <profile>`. The reference machine now uses Linux (this section); the Windows image (about 10 GB compressed) was not pulled.

**If something fails on Blackwell:**

| Failure | Fallback (in order) |
|---|---|
| AWQ kernels missing for sm_120 | `Qwen/Qwen3-8B-FP8` (Blackwell has FP8 tensor cores; ≈9 GB, still fits 16 GB at 8k context), then bitsandbytes 4-bit on the bf16 checkpoint. Not needed: AWQ runs through Marlin on sm_120 |
| vLLM lacks sm_120 kernels | Not the case for the 0.31.0 wheel with torch 2.13.0+cu130 |
| gpt-oss-20b runs out of memory | Observed: weight-allocation OOM with the desktop on, so context length does not help. Headless GPU, else Qwen3-14B as teacher (`tracks.md` §5.10) |

## 5. Open points for the team

1. **Thinking mode.** Plan §6.11 says that in the E18 sweep thinking is on and thinking tokens count toward the cap. With thinking on, 8B decode speed probably breaks the 8–20 s per alert target (`tracks.md` §5.4). Proposal: `enable_thinking: false` for the working default and every Group D system, and on for every E18 candidate. Decide with Shreyas (fairness protocol).
2. **Abhishek's second server** needs vLLM 0.31.0 with the same AWQ checkpoint. His ollama Q4_K_M number (62 tok/s) is a different quantisation and engine, so it cannot give the factor. He should rerun `bench_llm.py` against vLLM once the reference JSON exists.
3. **Tailnet.** Plan: `tailscale serve --bg --tcp 8000 tcp://127.0.0.1:8000` on the serving PC, so the server stays bound to loopback and only tailnet members reach it. Never `tailscale funnel` (that is public). Everyone needs a Tailscale account in one tailnet.
