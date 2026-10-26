# GPU calendar

Block a machine here **before** you take it offline. QLoRA and the gpt-oss-20b teacher need the whole GPU on the 5070 Ti PC, so its 8B server is down during those windows. Other LLM jobs move to the second server (3070 Ti laptop, if the week-0 test passes) or wait.

Times are local to the machine owner; say the timezone if it matters. Keep it chronological.

| Date / window | Machine | Who | Job | Offline for others? | Est. hours |
|---|---|---|---|---|---|
| _example_ 2026-xx-xx 22:00 → 06:00 | 5070 Ti PC | Akshat | Teacher run, gpt-oss-20b, 500 alerts | Yes (8B server down) | 8 |
| 2026-10-23 (done) | 5070 Ti PC (booted into Linux) | Akshat | vLLM smoke test and `bench_llm.py`, three profiles | No | 0.2 |
| 2026-10-23 (done) | 5070 Ti PC (Linux) | Akshat | First real runs: planner/ReAct on 20 val-fit alerts, untuned 4B/8B cascade baseline | 8B server down between runs | 0.26 |
| 2026-10-23/24 (done) | 5070 Ti PC (Linux) | Akshat | Draft strategy library: planner on 1,983 HI-Medium train alerts, Qwen3-8B | 8B server was mine | 1.31 |
| 2026-10-24 (done) | 5070 Ti PC (Linux) | Akshat | E13 engineering check: planner sequential vs `--batch-schedule 2,1` on 20 val-fit alerts, Qwen3-8B | 8B server was mine | 0.17 |
| 2026-10-25 22:19 → 23:20 (done, wasted) | 5070 Ti PC (Linux) | Akshat | E18 stage 1, Qwen3-8B and Qwen3.8-27B, 160 alerts each: invalid (no back-test table passed, every verdict p0); rerun | Yes (no shared 8B server) | 1.05 |
| 2026-10-25 23:25 → 26 Oct 01:39 (done) | 5070 Ti PC (Linux) | Akshat | E18 stage 1 queue (`scripts/e18_queue.sh`): Qwen3-8B, Qwen3.8-27B Q3, gpt-oss-20b, Qwen3-14B, Gemma 4 12B, Gemma 4 26B-A4B, one at a time | Yes | 2.2 |
| 2026-10-26 01:43 → 04:53 (done) | 5070 Ti PC (Linux) | Akshat | E18 stage 2 (Qwen3-14B, Gemma 4 12B, Qwen3.8-27B, 460 alerts), thinking arm stopped after 15 alerts, gpt-oss 1,024-token sensitivity arms (160 + 460) | Yes | ~3.2 |
| 2026-10-26 04:58 → 08:08 (done) | 5070 Ti PC (Linux) | Akshat | Teacher runs: Gemma 4 12B planner on 1,983 HI-Medium train alerts (`scripts/chunked_runs.sh`) | Yes | 3.2 |
| 2026-10-26 08:11 → 08:57 (done, superseded) | 5070 Ti PC (Linux) | Akshat | E8 first pass, four evidence views on 160 val eval alerts: hit the Gemma prompt-cap bug, re-run below | Yes | 0.77 |
| 2026-10-26 08:57 → 09:08 (done) | 5070 Ti PC (Linux) | Akshat | Decision rows for T2: untuned Gemma 4 12B, `large` and `large-logprob` tiers, 8,066 val rows each | Yes | 0.18 |
| 2026-10-26 09:08 → 10:51 (done) | 5070 Ti PC (Linux) | Akshat | E8 re-run with the prompt-refit fix, four arms | Yes | 1.70 |

## Weekly GPU-hour log (reference-hours)

Convert laptop hours with the week-0 benchmark factor (see `docs/hardware.md`). Budget: 250 reference-hours total.

| Week | 5070 Ti PC | 3070 Ti laptop | 3050 Ti laptop | CPU-only | Total GPU | Cumulative |
|---|---|---|---|---|---|---|
| 0 | 0.2 (2026-10-23: vLLM smoke test, ~10 min of server time incl. failed starts, factor 1.0) | | | n/a | 0.2 | 0.2 |
| 8 | 2.18 (2026-10-23/24: 0.70 h earlier: first real runs 0.26 h, Qwen3-8B 795 s + Qwen3-4B 136 s; planner-quality reruns 0.44 h, Qwen3-8B 1,493 s + Qwen3-4B about 100 s; plus draft-library train runs 1.31 h, Qwen3-8B, 1,983 alerts, server up 1.62 h incl. load and idle; plus E13 batched-planner check 0.17 h, Qwen3-8B, server up 625 s; factor 1.0; `docs/t3_first_runs.md`, `docs/t3_library_draft.md`, `docs/t3_e13.md`) | | | n/a | 2.18 | 2.38 |
| Phase A (25-26 Oct) | 12.4 (server up incl. loads, from the run logs: E18 stage 1 incl. the invalid first pass 3.3 h (22:19-01:39), stage 2 + thinking arm + gpt-oss 1,024-token arms 3.2 h (01:42-04:53); teacher runs 3.2 h; E8 first pass 0.77 h; decision rows 0.18 h; E8 re-run 1.70 h; factor 1.0; `docs/t3_e18.md`, `docs/t3_e8.md`) | | | HI-Medium edge scorer retrain (5 min), GFP see-everything (1 h 04 min) | 12.4 | 14.8 |
