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
| GPU / VRAM (check `nvidia-smi`) | RTX 3070 Ti Laptop / 8 GB (driver 580.159.03) | | RTX 5070 Ti / 16 GB | none |
| RAM | 15 GiB (about 9.5 GiB available) | | 32 GB | |
| CPU model / cores | i7-12650H, 10 cores / 16 threads | | | |
| Free disk | 6.0 GB at start (149 GB disk, 96% used); 16 GB after cleanup | | | |
| OS | Pop!_OS 22.04 LTS (Linux 6.16.3) | | | |
| WSL2 works? | n/a (native Linux) | | | |
| Python version | 3.12.13 (system, user site; no venv) | | | |
| Benchmark tokens/s (see 4) | pending T3 benchmark script (Qwen3-8B Q4_K_M ollama proxy numbers in `docs/t1_week0.md`) | | (reference) | n/a |

HI-Medium needs lazy Polars/DuckDB scans on machines with less than 32 GB. Full-graph jobs default to Akshat's PC.

## 3. Week-0 smoke tests (tick when done; details in `tasks/`)

- [ ] **T3:** vLLM + Qwen3-8B (4-bit), Qwen3-4B, gpt-oss-20b run on the 5070 Ti. Blackwell needs a CUDA 12.8+ build of PyTorch/vLLM; on Windows use WSL2 or Docker. Write the recipe below.
- [ ] **T1:** Qwen3-8B 4-bit loads and answers a 6,000-token prompt on the 3070 Ti. This decides whether T1 is a second LLM server.
- [ ] **T2:** Qwen3-1.7B / 4B-4bit run on the 3050 Ti (for harness smoke tests).
- [ ] **T4:** the mock LLM / record-replay mode works (no GPU needed); a request to the shared endpoint succeeds.
- [x] **T1:** snapml and PyG install; the snapml licence is checked. snapml 1.17.2 has a cp312 wheel and the GraphFeaturePreprocessor runs on a 5-edge toy graph; PyG 2.7.0 two-layer GCN runs on the 3070 Ti (CUDA); pcst_fast solves a toy tree. snapml licence is proprietary (IBM ILAN), see `docs/t1_week0.md`.
- [ ] **T2:** SimPy and Lark install and a toy grammar parses.

## 4. Benchmark for converting laptop hours to reference-hours

T3 defines one fixed prompt and settings (same checkpoint, `max_tokens` cap, greedy decoding) and publishes the script here. Each LLM-capable machine reports tokens/s. The reference-hour factor = tokens/s on that machine ÷ tokens/s on the 5070 Ti. Do not assume it.

## 5. Shared LLM server (T3 writes this in week 0-2)

- Serve Qwen3-8B 4-bit with vLLM, OpenAI-compatible, API key required.
- Reach it over a private network (Tailscale or similar). Never expose it to the public internet.
- Same checkpoint, same vLLM version and same sampling settings on every serving machine. Only timing differs.
- `Limits.max_tokens` (in `ftm/types.py`) is set so a prompt fits the smallest serving machine.
- A mock server for offline development lives in `ftm/agent/` (T2/T3).

Recipe (T3 fills in):

```
(vLLM version, install steps, launch command, API key handling)
```

## 6. Rules

1. Timing numbers (latency, GPU-seconds) are measured on the reference machine only. Brier/accuracy can come from any machine with the identical checkpoint and settings.
2. One workload per GPU: training and serving do not share a GPU. Block time in `GPU_CALENDAR.md`.
3. Laptops: plugged in, cooling, checkpoint at least hourly, no multi-day single runs.
4. Log GPU-hours per machine weekly in `GPU_CALENDAR.md`.
