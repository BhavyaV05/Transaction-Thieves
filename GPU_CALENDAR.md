# GPU calendar

Block a machine here **before** you take it offline. QLoRA and the gpt-oss-20b teacher need the whole GPU on the 5070 Ti PC, so its 8B server is down during those windows. Other LLM jobs move to the second server (3070 Ti laptop, if the week-0 test passes) or wait.

Times are local to the machine owner; say the timezone if it matters. Keep it chronological.

| Date / window | Machine | Who | Job | Offline for others? | Est. hours |
|---|---|---|---|---|---|
| _example_ 2026-xx-xx 22:00 → 06:00 | 5070 Ti PC | Akshat | Teacher run, gpt-oss-20b, 500 alerts | Yes (8B server down) | 8 |
| 2026-10-23 (done) | 5070 Ti PC (booted into Linux) | Akshat | vLLM smoke test and `bench_llm.py`, three profiles | No | 0.2 |
| 2026-10-23 (done) | 5070 Ti PC (Linux) | Akshat | First real runs: planner/ReAct on 20 val-fit alerts, untuned 4B/8B cascade baseline | 8B server down between runs | 0.26 |
| 2026-10-23/24 (done) | 5070 Ti PC (Linux) | Akshat | Draft strategy library: planner on 1,983 HI-Medium train alerts, Qwen3-8B | 8B server was mine | 1.31 |

## Weekly GPU-hour log (reference-hours)

Convert laptop hours with the week-0 benchmark factor (see `docs/hardware.md`). Budget: 250 reference-hours total.

| Week | 5070 Ti PC | 3070 Ti laptop | 3050 Ti laptop | CPU-only | Total GPU | Cumulative |
|---|---|---|---|---|---|---|
| 0 | 0.2 (2026-10-23: vLLM smoke test, ~10 min of server time incl. failed starts, factor 1.0) | | | n/a | 0.2 | 0.2 |
| 8 | 2.01 (2026-10-23/24: 0.70 h earlier: first real runs 0.26 h, Qwen3-8B 795 s + Qwen3-4B 136 s; planner-quality reruns 0.44 h, Qwen3-8B 1,493 s + Qwen3-4B about 100 s; plus draft-library train runs 1.31 h, Qwen3-8B, 1,983 alerts, server up 1.62 h incl. load and idle; factor 1.0; `docs/t3_first_runs.md`, `docs/t3_library_draft.md`) | | | n/a | 2.01 | 2.21 |
