# T3 week 0: serving, model cards, benchmark, machine report

Owner: T3 (Akshat). Covers the week-0 row of `tracks.md` §5.8 (serving smoke test, recipe, conversion benchmark, report specs) and the model-card check that `plan.md` §6.11 asks T3 to do in week 0.

**Status (2026-10-19).** The scripts, the recipe and the model-card check are done. **The GPU smoke test has not been run yet.** Docker's disk image is still on C:, which has only 38 GB free, and the vLLM image plus three checkpoints need about 35 GB. Section 4 has the exact steps; the results table there is empty until they run. Nothing in this document is a measured GPU number.

## 1. Machine report (reference machine)

| | T3 Akshat |
|---|---|
| GPU / VRAM | NVIDIA GeForce RTX 5070 Ti (Blackwell, sm_120) / 16,303 MiB, driver 610.88 |
| RAM | 31.1 GiB usable (32 GB installed) |
| CPU | AMD Ryzen 9 7900X, 12 cores / 24 threads |
| Free disk | E: 406 GB free of 467 GB (all project data, caches and weights go here); C: 38 GB free of 269 GB (keep off it) |
| OS | Windows 11 Pro, build 26200 |
| WSL2 | Yes: default version 2, Docker Desktop 29.6.2 with the WSL2 backend. No Ubuntu distro yet, so vLLM runs in Docker (`--backend docker`, the default on Windows) |
| Python | 3.12.10 in `.venv` (uv); CI uses 3.11 |
| Benchmark | Reference machine (factor 1.0 by definition); tokens/s pending the smoke test |

## 2. Model-card check (`plan.md` §6.11)

Checked on 2026-10-19 against the Hugging Face API (`/api/models/<repo>` for licence and commit, `config.json` for context length). The revisions are pinned in `configs/llm/serving.yaml`.

| Role | Repo | Revision (pinned) | Licence | Native context | Quantisation | Size on disk (approx.) |
|---|---|---|---|---|---|---|
| Planner / Group D default | `Qwen/Qwen3-8B-AWQ` | `4da05a8edb55` | Apache-2.0 | 40,960 | AWQ 4-bit | ≈6 GB |
| Fast model base (C13) | `Qwen/Qwen3-4B-AWQ` | `74d4bd2bd4bf` | Apache-2.0 | 40,960 | AWQ 4-bit | ≈2.7 GB |
| Teacher candidate | `openai/gpt-oss-20b` | `6cee5e81ee83` | Apache-2.0 | 131,072 (YaRN) | MXFP4 (native) | ≈13 GB |
| QLoRA base (planner SFT) | `Qwen/Qwen3-8B` | `b968826d9c46` | Apache-2.0 | 40,960 | bf16 (QLoRA loads in 4-bit) | ≈16 GB |
| QLoRA base (fast model) | `Qwen/Qwen3-4B` | `1cfa9a720891` | Apache-2.0 | 40,960 | bf16 | ≈8 GB |

E18 backbone-sweep candidates (not downloaded; checked so the sweep can be planned):

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

## 4. Smoke test: steps and results (to run)

Steps, in order. The recipe in `docs/hardware.md` §5 is the short version.

1. **Move Docker's data off C:** Docker Desktop → Settings → Resources → Advanced → *Disk image location* → `E:\DockerDesktop` → Apply & restart. Today `docker_data.vhdx` is at `%LOCALAPPDATA%\Docker\wsl\disk` on C:.
2. `setx FTM_HF_HOME E:\hf_cache`, then open a new shell. Weights go to `E:\hf_cache\hub`.
3. GPU visible in Docker: `docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi`.
4. `python scripts/serve_llm.py --gen-key`, then load `.env` into the shell (see the recipe).
5. For each of `qwen3-8b`, `qwen3-4b`, `gpt-oss-20b` (one at a time; gpt-oss-20b needs the whole GPU):
   - `python scripts/serve_llm.py --model <profile>` and wait for `Application startup complete`
   - `python scripts/bench_llm.py --machine t3-5070ti` in a second shell
   - write down the load time and peak VRAM (`nvidia-smi`) in the table below
6. Post the `qwen3-8b` JSON (`outputs/bench/t3-5070ti_qwen3-8b.json`) so the other machines can pass it as `--reference`.

| Profile | Starts on sm_120? | Load time | Peak VRAM | Prefill tok/s | Decode tok/s | Batch tok/s (4 parallel) | Notes |
|---|---|---|---|---|---|---|---|
| qwen3-8b | pending | | | | | | |
| qwen3-4b | pending | | | | | | |
| gpt-oss-20b | pending | | | | | | |

**If something fails on Blackwell:**

| Failure | Fallback (in order) |
|---|---|
| AWQ kernels missing for sm_120 | `Qwen/Qwen3-8B-FP8` (Blackwell has FP8 tensor cores; ≈9 GB, still fits 16 GB at 8k context), then bitsandbytes 4-bit on the bf16 checkpoint |
| The v0.31.0 image lacks sm_120 kernels | The `cu13x` build of the same version, if one exists; otherwise WSL2 Ubuntu with `pip install vllm==0.31.0` |
| gpt-oss-20b runs out of memory at 8k | `--max-model-len 6656` (6,000 prompt + 656 out), then `--gpu-memory-utilization 0.95`; else Qwen3-14B as teacher (`tracks.md` §5.10) |

## 5. Open points for the team

1. **Thinking mode.** Plan §6.11 says that in the E18 sweep thinking is on and thinking tokens count toward the cap. With thinking on, 8B decode speed probably breaks the 8–20 s per alert target (`tracks.md` §5.4). Proposal: `enable_thinking: false` for the working default and every Group D system, and on for every E18 candidate. Decide with Shreyas (fairness protocol).
2. **Abhishek's second server** needs vLLM 0.31.0 with the same AWQ checkpoint. His ollama Q4_K_M number (62 tok/s) is a different quantisation and engine, so it cannot give the factor. He should rerun `bench_llm.py` against vLLM once the reference JSON exists.
3. **Tailnet.** Plan: `tailscale serve --bg --tcp 8000 tcp://127.0.0.1:8000` on the serving PC, so the server stays bound to loopback and only tailnet members reach it. Never `tailscale funnel` (that is public). Everyone needs a Tailscale account in one tailnet.
