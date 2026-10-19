# Hardware and setup

Companion to `tracks.md` section 1 and `plan.md` section 12.

## 1. Machines

| Track | Owner | Machine | Main role |
|---|---|---|---|
| T1 | Abhishek | RTX 3070 Ti laptop, 8 GB VRAM | GNNs, federated, edge scorer; possibly a second 8B server |
| T2 | Bhavya | RTX 3050 Ti laptop, 4 GB VRAM | CPU-heavy simulation and back-testing; small-model tests |
| T3 | Akshat | RTX 5070 Ti PC, 16 GB VRAM, 32 GB RAM | **Reference machine**; 8B serving, QLoRA, teacher, final run |
| T4 | Shreyas | No GPU | CPU only; uses the shared LLM servers |

## 2. Week-0 machine report (fill in)

| | T1 Abhishek | T2 Bhavya | T3 Akshat | T4 Shreyas |
|---|---|---|---|---|
| GPU / VRAM (check `nvidia-smi`) | RTX 3070 Ti Laptop / 8 GB (driver 580.159.03) |  | RTX 5070 Ti / 16 GB (16,303 MiB, driver 610.88) | none |
| RAM | 15 GiB (about 9.5 GiB available) |  | 31.1 GiB usable (32 GB) | 14 GiB (about 5 GiB available) |
| CPU model / cores | i7-12650H, 10 cores / 16 threads |  | Ryzen 9 7900X, 12 cores / 24 threads | i7-12650H, 10 cores / 16 threads |
| Free disk | 6.0 GB at start (149 GB disk, 96% used); 16 GB after cleanup |  | E: 406 GB (use this); C: 38 GB (keep off it) | 19 GB |
| OS | Pop!_OS 22.04 LTS (Linux 6.16.3) |  | Windows 11 Pro (build 26200) | Ubuntu 26.04.1 LTS (Linux 7.0.0) |
| WSL2 works? | n/a (native Linux) |  | Yes (Docker Desktop 29.6.2, WSL2 backend; no Ubuntu distro) | n/a (native Linux) |
| Python version | 3.12.13 (system, user site; no venv) |  | 3.12.10 (`.venv`, uv) | 3.12 venv via uv (system Python is 3.14) |
| Benchmark tokens/s (see 4) | pending T3 benchmark script (ollama Q4_K_M proxy: 62 tok/s generation, not comparable) |  | (reference) pending the vLLM smoke test | n/a |

HI-Medium needs lazy Polars/DuckDB scans on machines with less than 32 GB. Full-graph jobs default to Akshat's PC.

## 3. Week-0 smoke tests (tick when done; details in `tasks/`)

- [ ] **T3:** vLLM + Qwen3-8B (4-bit), Qwen3-4B, gpt-oss-20b run on the 5070 Ti. Blackwell needs a CUDA 12.8+ build of PyTorch/vLLM; on Windows use WSL2 or Docker. Write the recipe below. Recipe and scripts are written (section 5); **the GPU run is pending** (steps and the results table: `docs/t3_week0.md` section 4).
- [x] **T1 (proxy only):** Qwen3-8B Q4_K_M via **ollama/GGUF, not vLLM**, loads 100% on the 3070 Ti (6.4 GB of 8.2 GB VRAM peak) and answers a ~5,900-token prompt in ~3.3 s (about 2,100 prompt tok/s, 62 generation tok/s). The vLLM 4-bit check on the same checkpoint is still open, so the second-server decision is not final. Details: `docs/t1_week0.md` section 6.
- [ ] **T2:** Qwen3-1.7B / 4B-4bit run on the 3050 Ti (for harness smoke tests).
- [ ] **T4:** the mock LLM / record-replay mode works (no GPU needed); a request to the shared endpoint succeeds.
- [x] **T1:** snapml and PyG install; the snapml licence is checked. snapml 1.17.2 has a cp312 wheel and the GraphFeaturePreprocessor runs on a 5-edge toy graph; PyG 2.7.0 two-layer GCN runs on the 3070 Ti (CUDA); pcst_fast solves a toy tree. snapml licence is proprietary (IBM ILAN), see `docs/t1_week0.md`.
- [ ] **T2:** SimPy and Lark install and a toy grammar parses.

## 4. Benchmark for converting laptop hours to reference-hours

Script: `scripts/bench_llm.py` (T3). One fixed prompt (a ~3,000-token aliased transaction trail, the same on every machine), greedy decoding, `seed=0`, 256 output tokens forced with `ignore_eos`, 5 repeats plus a warm-up, then 4 requests in parallel. Each request has a different fixed first line, so prefix caching cannot hide the prefill. Run it against a vLLM server started with `scripts/serve_llm.py` (same checkpoint and vLLM version as the reference); ollama or llama.cpp numbers are not comparable.

```
python scripts/bench_llm.py --machine t3-5070ti                                                   # reference, first
python scripts/bench_llm.py --machine t1-3070ti --reference outputs/bench/t3-5070ti_qwen3-8b.json  # any other machine
```

It writes `outputs/bench/<machine>_<model>.json` and prints one row for the table below. **Factor** = decode tok/s on that machine ÷ decode tok/s on the 5070 Ti (single stream, because agent loops are sequential); reference-hours = hours on that machine × factor. Do not assume it.

| machine | model | server | prompt tok | prefill tok/s | decode tok/s | batch tok/s | factor |
|---|---|---|---|---|---|---|---|
| t3-5070ti | qwen3-8b | vLLM 0.31.0 | pending | | | | 1.000 (reference) |

## 5. Shared LLM server (T3 writes this in week 0-2)

- Serve Qwen3-8B 4-bit with vLLM, OpenAI-compatible, API key required.
- Reach it over a private network (Tailscale or similar). Never expose it to the public internet.
- Same checkpoint, same vLLM version and same sampling settings on every serving machine. Only timing differs.
- `Limits.max_tokens` (in `ftm/types.py`) is set so a prompt fits the smallest serving machine.
- A mock server for offline development lives in `ftm/agent/` (T2/T3).

Recipe (T3; versions and revisions are pinned in `configs/llm/serving.yaml`; details and fallbacks in `docs/t3_week0.md`):

**Windows (Akshat's PC, Docker Desktop)**

```powershell
# once: move Docker's data and the HF cache off C:
#   Docker Desktop > Settings > Resources > Advanced > Disk image location = E:\DockerDesktop, Apply & restart
setx FTM_HF_HOME E:\hf_cache                              # then open a new shell
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi   # GPU visible in Docker?
python scripts/serve_llm.py --gen-key                     # writes FTM_LLM_API_KEY to .env (gitignored)

# every time (load .env into this shell, then start one profile)
Get-Content .env | ForEach-Object { $k, $v = $_ -split '=', 2; Set-Item "env:$k" $v }
python scripts/serve_llm.py --model qwen3-8b              # or qwen3-4b / gpt-oss-20b; --dry-run prints the command
```

**Linux or WSL2 (e.g. Abhishek's laptop as the second server)**

```bash
python -m venv ~/vllm && source ~/vllm/bin/activate && pip install vllm==0.31.0
export FTM_HF_HOME=~/hf_cache
python scripts/serve_llm.py --gen-key && set -a && source .env && set +a
python scripts/serve_llm.py --model qwen3-8b --backend native
```

**Sharing over the tailnet.** The server binds to 127.0.0.1. Expose it to tailnet members only with `tailscale serve --bg --tcp 8000 tcp://127.0.0.1:8000` (never `tailscale funnel`, which is public). Clients set:

```
FTM_LLM_ENDPOINT=http://<serving-pc-tailnet-name>:8000/v1
FTM_LLM_API_KEY=<the key, sent privately; never commit it>
FTM_LLM_MODEL=qwen3-8b
```

**Key handling.** The key lives in `.env` on the serving machine (gitignored) and reaches vLLM through the `VLLM_API_KEY` environment variable, so it never appears in the command line or the process list. Rotate it by deleting the line in `.env` and running `--gen-key` again; tell the team.

**Checking the server:** `curl -H "Authorization: Bearer $FTM_LLM_API_KEY" $FTM_LLM_ENDPOINT/models` lists the served model name.

## 6. Rules

1. Timing numbers (latency, GPU-seconds) are measured on the reference machine only. Brier/accuracy can come from any machine with the identical checkpoint and settings.
2. One workload per GPU: training and serving do not share a GPU. Block time in `GPU_CALENDAR.md`.
3. Laptops: plugged in, cooling, checkpoint at least hourly, no multi-day single runs.
4. Log GPU-hours per machine weekly in `GPU_CALENDAR.md`.
