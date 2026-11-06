# T3 ada: the college HPC cluster as an offload target (2 Nov 2026)

Owner: T3 (Akshat).  Worker ADA.  What `ssh ada` is, the sbatch recipe that works, what ran there, and what can and cannot run on its RTX 2080 Ti.
Results of the jobs that ran there are in `docs/t1_baselines_val.md` (GNN time-respecting re-score) and `docs/t3_s1bench.md` (S1BENCH); this file is the recipe.
Sections marked (smoke) were measured with a short real run, not guessed; the date and the job id are given.

## 1. The cluster in one table

| item | value |
|---|---|
| login | `ssh ada` (ada-gw1.iiit.ac.in), user `garimella.abhishek`; the gateway has python 3.9 only: nothing heavy there |
| scheduler | SLURM 26.05; the only working triple is `-p u22 -A research --qos=medium` (`--qos=low` is rejected for this association: its default and only QoS is `medium`) |
| `medium` QoS | per user 4 GPU, 40 CPU, 125000 MB RAM, 4 days wall. `-p u22-cpu` is refused for this account |
| account cap | the `research` account has a group cap of 164 GPUs; when other users hold it a request waits with reason `AssocGrpGRES` (a 4-GPU job waited about 1 h) |
| GPUs | node type is a *feature*: `-C 2080ti` = RTX 2080 Ti 11 GB (sm_75), 4 per node (gnode043..063 ...). **Without `-C` you can land on GTX 1080 Ti nodes (sm_61, no tensor cores, no cu128 kernels): gnode005 did.** Driver 570.211 (CUDA 12.8 runtime max) |
| CPUs per node | 40 (2 sockets); a 4-GPU request accepts at most 24 CPUs here (`-c 32`/`-c 36` fail at submit with "CPU count per node can not be satisfied") |
| RAM per node | 128 GB (QoS cap 125000 MB) |
| internet | works from the gateway and from compute nodes: pytorch.org 15 MB/s, huggingface.co 10-15 MB/s, wheels.vllm.ai 14 MB/s, GitHub releases 25 KB/s (never use), PyPI 0.1-0.4 MB/s (small wheels only) |
| `/home2/<user>` | NFS, 50 GB quota. Reads are fine; **writes are about 5 MB/s and tiny-file installs stall in D state for tens of minutes** (a plain `pip install torch` into a venv on it took more than 40 min and a 12 GB `tar` never finished). Keep code, data and logs here; build nothing here |
| `/ssd_scratch/<user>` | node-local SSD, 880 GB, 440 MB/s, world-writable dir, **gone when the allocation ends**: venvs, HF cache, pip cache go here |
| `/tmp` (= `/scratch`) | node-local shared HDD, 155 MB/s while idle, other users hammer it: avoid |
| `/share1/<user>` | exists on the **gateway only**, not on compute nodes (44 GB of models downloaded there first were unreachable) |

## 2. The recipe

One allocation, then job steps (the lead's rule: one node, 4 GPUs, no swarm of small jobs, nothing heavy on the gateway):

```bash
# ~/tt/sbatch/hold4.sbatch   (holds the node until ~/tt/STOP exists; work goes in as job steps)
#SBATCH -p u22
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -N1
#SBATCH -C 2080ti
#SBATCH --gres=gpu:4
#SBATCH -c 24
#SBATCH --mem=100G
#SBATCH -t 4-00:00:00
#SBATCH -J ada4
#SBATCH -o /home2/garimella.abhishek/tt/logs/ada4_%j.out
hostname; nvidia-smi -L
while [ ! -e ~/tt/STOP ]; do sleep 30; done
```

```bash
JID=<job id from squeue>
# environments (node-local SSD; ~10 min for gnn, ~15 min for vllm; run both at once)
srun --jobid=$JID --overlap -n1 -c 8 --mem=24G bash ~/tt/ada_scripts/setup_envs.sh gnn
srun --jobid=$JID --overlap -n1 -c 8 --mem=24G bash ~/tt/ada_scripts/setup_envs.sh vllm
# one GPU task = one step (-n1 is required: without it the step runs 4 tasks)
srun --jobid=$JID --overlap -n1 --gres=gpu:1 -c 6 --mem=44G bash ~/tt/ada_scripts/gnn_causal.sh multi_gnn 2048
# release
touch ~/tt/STOP   # or scancel $JID
```

Other facts that cost time: a step that asks for less RAM than the process needs is OOM-killed with just "Killed"; `pkill -f` patterns kill your own shell; `timeout N srun` kills the step.

### Environments (`scripts/ada/setup_envs.sh`)

| venv | contents | why separate |
|---|---|---|
| `venv312` | python 3.12.4 (`/usr/local/apps/python-3.12.4`, exists on compute nodes only), torch 2.8.0+cu128, torch_geometric 2.7.0, pyg_lib/torch_scatter/torch_sparse (pt28cu128 wheels), polars, lightgbm, scikit-learn, wandb (imported by the Multi-GNN code), transformers 5.21, peft 0.21.2, trl 1.16, bitsandbytes 0.50.2 | the same pins as the local `gnn-venv` and `train-venv` |
| `venv312_vllm` | torch 2.13.0+cu129, vLLM 0.31.0+cu129 (wheel from wheels.vllm.ai; the PyPI default wheel is cu130 and needs driver >= 580, ada has 570) | vLLM pins its own torch |

`~/venv312` on the NFS home (the user's old Jupyter env, torch 2.6+cu124) was left half-modified by an interrupted `pip install torch==2.8.0` and should be treated as broken; its `bin/python` symlink points to `/usr/local/apps/python-3.12.4`, which exists on compute nodes but not on the gateway (that is why it looked broken there).

## 3. Self-bootstrapping job on the node SSD (3 Nov 2026)

Why: the home NFS (`/home2`) writes at about 5 MB/s, reads cold files at about 1 MB/s (`cp` of the 43 MB uv binary took 42 s) and is metadata-slow. Installing torch + vLLM into `~/venv312` ran 6.5 h without finishing (job 54524, cancelled). `~/venv312` is the user's and is left as it is (half-modified, torch 2.13 files partly installed; do not use it, do not delete it). Nothing is installed into the NFS home any more.

The recipe (`scripts/ada/bootstrap_ssd.sh`, called by `scripts/ada/run_driver.sbatch` at the start of the 4-GPU job; the CPU-only variant is `run_bootstrap_cpu.sbatch`):

1. Scratch dir: `/ssd_scratch/$USER/$SLURM_JOBID` (880 GB SSD per node, 300-470 GB free on most nodes; on gnode012 it was 100 % full, which fails with `No space left on device`; the sbatch falls back to `/tmp` and requires 80 GB free, and a CPU test can use `-x gnode012`). Python 3.12.4 is at `/usr/local/apps/python-3.12.4` on compute nodes only.
2. `uv` (static musl binary) is downloaded from GitHub to the SSD by `curl` at job start (about 1 s; reading it from `~/bin/uv` over NFS made `uv venv` take 201 s). A copy sits in `~/bin/uv` as a fallback. The bootstrap script itself is copied to the SSD before it is run (replacing a script on NFS while bash reads it gives `Stale file handle`).
3. `uv venv -p /usr/local/apps/python-3.12.4/bin/python3.12`; `UV_CACHE_DIR` and `TMPDIR` on the SSD.
4. `uv pip install` torch 2.13.0 + torchvision 0.28.0 + torchaudio 2.11.0 from the pytorch cu129 index (`--index-strategy unsafe-best-match`, the driver is 570 so cu129, not cu130), then the vLLM 0.31.0+cu129 wheel with `-c ~/tt/ada_scripts/vllm_constraints.txt`, then polars/pyarrow/pandas/scikit-learn/lightgbm/wandb, peft 0.21.2, trl 1.16.0, bitsandbytes 0.50.2, accelerate 1.15.0, datasets 5.1.0, sentence-transformers, gliner2 2.1.0, laya 0.3.20, torch_geometric 2.7.0.
5. `HF_HOME` on the SSD; Qwen3-4B-AWQ, Gemma 4 12B w4a16 and Qwen3-8B are downloaded in the background from the moment torch is installed (other models are fetched by the tasks that use them).
6. PyG compiled extensions, answer to "are they needed": **yes**. `LinkNeighborLoader(..., time_attr=..., edge_label_time=...)` on a `HeteroData` (the call pattern of `ftm/baselines/multi_gnn.py` / `mega_gnn.py` with `--causal`) raises `ImportError: Edge-level temporal sampling requires a more recent 'pyg-lib' installation` without `pyg_lib` (checked in the bootstrap on the new venv: `WITH_PYG_LIB False WITH_TORCH_SPARSE False`; `torch_sparse` alone does not help), and `MEGA-GNN/models.py` and `training.py` import `torch_scatter` directly. There are no wheels for torch 2.13 (data.pyg.org stops at 2.12), so they are built from source once, CPU kernels only, and cached as wheels in `~/tt/wheels` (a few MB each, one NFS write): later jobs `uv pip install --no-deps ~/tt/wheels/*.whl` in seconds. `torch_scatter` 2.1.2 and `torch_sparse` 0.6.18 build with `pip wheel --no-deps --no-build-isolation`. `pyg_lib` needs `CUDAARCHS=75 TORCH_CUDA_ARCH_LIST=7.5 PATH=/usr/local/cuda/bin:$PATH` even with `WITH_CUDA=OFF` (torch's CMake config enables the CUDA language and fails with "Failed to detect a default CUDA architecture" otherwise) and about 30 GB of RAM for the template-heavy C++ (an 8 GB job step was OOM-killed in `cc1plus`).
7. MEGA-GNN's two pybind11 extensions are rebuilt with `g++` (seconds).

Timings (job 54898, gnode076, 24 cores): venv 1 s; torch 105 s; vLLM wheel 5.7 min (cumulative 450 s); data + QLoRA + S1 stack 7 s (cached); torch_scatter build 5.5 min and torch_sparse 7 min (cumulative 787 s and 1218 s, run in the background while vLLM downloads); import check and MEGA extensions 45 s; **bootstrap wall 1366 s (22.8 min) without pyg_lib**. **pyg_lib** needs two more things, found the hard way: (a) `CUDAARCHS=75 TORCH_CUDA_ARCH_LIST=7.5` and `/usr/local/cuda/bin` on `PATH` (see 6); (b) the venv's `bin` first on `PATH`, because pyg-lib's CMake runs `python -c "import torch; ..."` to pick the C++ ABI, and the pytorch.org torch 2.13 wheels use the cxx11 ABI: with the wrong `python` it builds with ABI 0, `import pyg_lib` then fails with `undefined symbol _ZNK5torch8autograd4Node4nameEv` and PyG silently disables it (`WITH_PYG_LIB False`, then the temporal-sampler ImportError). Build 3.4 min with 12 jobs and 40 GB; the wheel (`pyg_lib-0.10.0-cp311-abi3`, 2.3 MB) is in `~/tt/wheels` next to `torch_scatter-2.1.2` and `torch_sparse-0.6.18`. After installing it, the temporal check prints `WITH_PYG_LIB True WITH_TORCH_SPARSE True` and `TEMPORAL_OK`. A job that starts with the three wheels cached skips the 13 min of building (torch_scatter and torch_sparse only needed `pip wheel`, run in the background during the vLLM download), so a bootstrap should take about 10 min.

Other fixes found while watching the first tasks: `common.sh` built `FTM_LLM_API_KEY=$VLLM_API_KEY` in the same `export` statement that defined `VLLM_API_KEY` (empty key, every vLLM server died with "no API key" and the smoke_gemma_tp2 task wrote a false `DOES_NOT_SERVE` and a done marker, deleted); `ftm/fastmodel/s1bench_adapters.py` on rw/s1bench had a `_dtype` helper inserted between `@dataclass(frozen=True)` and `class TypedQuestion` (every JevK5/Tev1/Laya/GLiNER task died with `'function' object has no attribute '__mro__'`; fixed on rw/s1bench). A step started with `srun --overlap` is killed when its srun returns (a `nohup` build died with it): run long things in the foreground of an `srun` that you keep alive, always with `-n1` (without it the step runs once per allocated CPU).

Driver restart inside a running job (`scripts/ada/restart_driver.sh <jobid>`): the sbatch ends with `rm -rf $SC`, so the script `SIGSTOP`s the batch shell, kills the old `driver.py` and its tasks and execs a new driver from a long-lived `srun --overlap -n1` step. When the new driver is finished, `kill -CONT` the batch shell (it then removes the scratch dir) or `scancel` the job and `rm -rf /ssd_scratch/$USER/<jobid>` on the node.

## 4. Time-respecting GNN re-score (job 54898, 3 Nov 2026)

`scripts/ada/tasks/gnn.sh {multi_gnn|mega_gnn}` ran `train_gnn.py --our-split --data hi_medium --mode see_everything --causal --resume --max-epochs 0 --batch-size 2048 --workers 0` on one RTX 2080 Ti each (nothing is trained, the stored checkpoint is loaded and scored with PyG `time_attr` / `edge_label_time`): MEGA-GNN 1,619 s (first try failed twice on the missing temporal sampler, see section 3, item 6), Multi-GNN 4,178 s. Batch 2048 fit in 11 GB. Outputs (`p0_<model>_causal*.parquet`, `p0_hi_medium_<model>_causal_see_everything.json`) went to `data/scores/hi_medium` and `data/runs`. The comparison and the numbers are in `docs/t1_baselines_val.md` ("GNN look-ahead, measured"): Multi-GNN val Brier 0.000658 (look-ahead) to 0.001113 (time-respecting), PR-AUC 0.611 to 0.199; MEGA-GNN 0.000988 to 0.001266, PR-AUC 0.342 to 0.019; GFP LightGBM 0.000859. The time-respecting numbers are for checkpoints that were trained with the look-ahead loader.

## 5. Software freeze

The venv built by `scripts/ada/bootstrap_ssd.sh` is frozen into `~/tt/venv_ssd_freeze.txt` at the end of each bootstrap (`uv pip freeze`). Freeze of job 61908 (6 Nov 2026, after the torchcodec / pcst_fast fixes): `docs/t3_ada_venv_freeze.txt` (`uv pip freeze`, 257 packages), sha256 `e1b244b7004bc4d625f518a844d8b3164892745a75036fa457980e21a326f1aa`. This freeze was taken from the venv whose bootstrap predated the pcst_fast build step and the langgraph line (both were installed by hand with the commands now in the script).

## 6. Gemma 4 12B on two RTX 2080 Ti (tensor parallel 2, fp16) and the cross-machine check (6 Nov 2026, job 61908, gnode078)

**It serves.** `scripts/serve_llm.py --model gemma4-12b --turing --tp 2 --gpu-util 0.88` (the profile of `configs/llm/serving.yaml`; `--turing` drops the fp8 KV cache and adds `--dtype float16`; vLLM 0.31.0+cu129, Marlin w4a16 kernels, TRITON_ATTN forced by vLLM for Gemma 4's mixed head sizes). Two things stood in the way, both fixed in `scripts/ada/bootstrap_ssd.sh`:

1. `OSError: libnvrtc.so.13` at `import vllm`: the PyPI linux wheel of torchcodec 0.17.0 is the CUDA 13 build, vLLM imports torchcodec unconditionally (`vllm/multimodal/media/audio.py`), and the earlier constraints file pinned exactly that wheel. The driver here is 570 (CUDA 12.8) so cu13 libraries are not an option; vLLM uses torchcodec only to decode audio, which the profile switches off. Fix: `torchcodec==0.17.0+cpu` from the PyTorch CPU index, installed after vLLM.
2. After the weights loaded, the first attention call failed: `triton OutOfResources: shared memory, Required: 98304, Hardware limit: 65536`. Gemma 4's global-attention layers have head size 512; vLLM's Triton unified-attention kernel tiles 32 KV tokens, which needs 96 KB of shared memory, and an sm_75 GPU has 64 KB. `scripts/ada/patch_vllm_turing.py` changes the launch parameters only (KV tile 16 and 2 pipeline stages for head size >= 512 below compute capability 8.0). Checked standalone against an fp32 reference on a 2080 Ti with the TP2 shapes (8 query heads, 1 KV head per GPU, head 512): max abs error 2e-3 (prefill 2 x 1,500 tokens), 2e-4 (decode), 6e-5 (300-token chunk on 4,000 of context); tile 32 fails at any stage count.

Measured (4 server starts, the load and compile times vary with the page cache): server up after 226-261 s cold (weights 10-14 s from the node SSD, the rest is torch.compile 0.5-88 s, CUDA-graph capture 14 s and profiling); 9.7 GB used per GPU at `--gpu-memory-utilization 0.88`; KV cache 4.44 GiB per GPU = **34,822 tokens, 4.25 concurrent requests of 8,192 tokens** (about 6 requests at the planner's usual 5-6k-token prompt); single-token logprob request 0.08-0.11 s warm.

**Real planner prompts** (ours_compose, tuned profile, library `shape_gemma_train.json`, the first 20 alerts of val `fit`, 13 of which reach the LLM; 32-35 LLM calls, 179-199k prompt tokens, 4.3-4.7k generated tokens per run):

| | PC (RTX 5070 Ti, recorded run c000, same 20 alerts) | ada serial | ada concurrency 4 |
|---|---|---|---|
| median seconds per alert (alerts that call the LLM) | 15.2 | 18.5 (summary median over the 16 timed alerts: 17.8) | 51.2 per alert (queueing), 19.2 in the run summary |
| seconds per LLM call | 6.5 | 8.2 | 19.7 per call under load |
| generated tokens per second (tokens out / call seconds, prompts included) | 19.9 | 16.4 | 6.8 per stream |
| wall for the 16 concurrent-phase alerts | 266 s for 76 alerts (c000, concurrency 4) = 3.5 s/alert | 193 s = 12.1 s/alert | 169 s = 10.6 s/alert |

So two 2080 Ti at TP2 serve Gemma about 25 % slower per call than the 5070 Ti and gain only about 14 % throughput from concurrency 4 (the PC's fp8 KV holds many more requests); on this node three other GPUs were running S1BENCH servers at the same time. Practical rate for planning: about 11-12 s per planner alert on the whole 2-GPU pair.

**Do the results match the PC run?** Compared per alert with `scripts/ada/cross_compare.py` (raw output `docs/t3_ada_crosscheck.json`; same 20 alert ids, same commit f949bf3 code, same library hash 39464ed7b909466c, same data files). No, not exactly, and the size of the difference is stated here:

| ada vs PC | serial | concurrency 4 | ada serial vs ada concurrency 4 |
|---|---|---|---|
| same first step (action, question, bank, eps) | 12/20 (0.60) | 12/20 (0.60) | 18/20 (0.90) |
| same full question sequence and outcomes | 9/20 (0.45) | 10/20 (0.50) | 16/20 (0.80) |
| same number of LLM calls | 0.65 | 0.85 | 0.80 |
| identical final probability | 9/20 (0.45) | 11/20 (0.55) | 17/20 (0.85) |
| final probability within 0.01 / mean abs diff / max abs diff | 0.55 / 0.0150 / 0.054 | 0.60 / 0.0125 / 0.054 | 0.95 / 0.0025 / 0.038 |
| Brier on the 20 alerts | 0.0864 vs 0.0842 | 0.0862 vs 0.0842 | 0.0864 vs 0.0862 |
| paired Brier difference (bootstrap 95 % CI over alerts) | +0.0022 [-0.0046, +0.0099] | +0.0020 [-0.0046, +0.0096] | -0.0002 [-0.0017, +0.0008] |

Reading: the planner's choices change on about half of the alerts between the machines, against about a fifth between a serial and a concurrent run on ada itself, so the machine (fp16 on Turing with TP2 and Marlin against the PC's bf16 and fp8 KV) matters more than batching noise. The aggregate Brier is the same within what 20 alerts can show (difference +0.002, CI covering 0), the verdict probabilities stay close (mean 0.013-0.015), and valid-DSL rate is 1.0 on both machines (ada usable-output rate 0.91 serial / 0.80 concurrent, PC 0.79 over 80 alerts). The inputs match: server-counted prompt tokens differ by exactly 4 on all 13 first prompts with the same evidence rows and the same menu, and the same code on the PC's CPU (a capture server that stores the request) builds byte-identical requests to the ones built on ada (checked on 9 requests), so the offset is not an ada effect; the recorded PC run's prompt estimate also sits 1-2 tokens higher than a fresh build with the same code and data, and server-side template accounting between vLLM versions is the likely cause (not verified). Consequence for placing work: planner runs on ada reproduce the PC's *aggregate* behaviour but not alert-by-alert choices, so a result must never be assembled from PC and ada alerts of the same arm. A whole arm (same chunk set) can run on either machine; arms that are compared with each other must run on the same machine.

## 7. Legacy loops against the graph versions on the real model (6-7 Nov 2026)

Question: can the pre-graph hand-written loops (`tests/legacy_llm_planner.py`, `ftm/external/*` `investigate`, reachable as `investigate_legacy` / `FTM_LEGACY_AGENT_LOOPS=1`) be deleted? The mock-LLM parity tests already pass; this is the check on the real model (Gemma 4 12B, TP2 fp16 on two 2080 Ti, `scripts/ada/tasks/parity_real.sh`). Design: every arm runs twice as the graph version (g1, g2: the run-to-run noise floor of the serving stack, temperature 0) and once as the legacy loop (l1) on the same alerts; parity = the legacy-vs-graph agreement is not lower than graph-vs-graph.

**Our planner** (20 val `fit` alerts, 113-114 LLM calls per run, serial). The frozen oracle has none of the options added after the graph port (`rules_mode` alert, ranked top-8 menu, cannot-cross stop, evidence floor), and its whole 204-question menu overflows the 8k context by itself (which is why the ranked menu exists), so the comparison runs in the legacy-equivalent configuration `--profile none --rules fixed --no-select-menu` (the wrapper `scripts/ada/legacy_parity_planner.py` refuses any option the legacy loop lacks). Legacy l1 against graph g2: 20/20 identical question sequences, outcomes, LLM calls and final probabilities (Brier 0.0903 both). g1 against g2: 19/20 (one alert, 113 vs 114 calls: serving noise); l1 against g1 the same 19/20. The options added after the port are covered by the graph version only (tuned-profile tests), not by any oracle.

**External agents** (8 val `fit` alerts each, concurrency 4, tuned profile). Compared: number of LLM calls and final probability (the external agents do not log a step trace, so the "same question sequence" column is vacuous for them and is not used). Agreement of l1 with g1 against the noise floor g2 with g1, by system (identical final probability / identical LLM calls):

| system | g2 vs g1 | l1 vs g1 |
|---|---|---|
| mediq, rewoo, sharma_native, sharma_ported | 8/8, 8/8 | 8/8, 8/8 |
| llmcompiler | 7/8, 8/8 | 7/8, 8/8 |
| bed_llm | 7/8, 7/8 | 7/8, 7/8 |
| uot | 7/8, 8/8 | 7/8, 8/8 |
| debate | 5/8, 8/8 | 5/8, 8/8 |
| react | 0.80 (4/5), 6/8 | 1.0, 8/8 |
| lats | 0.2, 7/8 | 0.5, 5/8 |
| reflexion | 0.83, 5/8 | 0.5, 4/8 |

(lats, reflexion and react answer only part of the alerts with a probability, the fractions are over those; torres, co_investigator and finfre_rag were still running or not started when this table was made: PENDING.) Reading: for every system but reflexion the legacy loop agrees with the graph at least as well as the graph agrees with itself, and for react, mediq, rewoo, sharma it is identical; reflexion's l1 agreement (0.5) is below its noise floor (0.83) but at n = 8 with 5 probabilities this is within what chance allows (a binomial interval around 0.83 at n = 6 reaches 0.5); more alerts would be needed to call it. `dspy_react` was not run (no MIPROv2 program file in `outputs/t3_dspy`).

**Verdict so far: not yet deletable as a whole.** The planner loop is equal on the real model (in the legacy-equivalent configuration, which is all the oracle can express). The external agents are equal within the serving noise on 9 of 10 systems tested; reflexion needs more alerts, five systems are pending, and nothing was deleted. Deleting needs: (1) the pending systems and a larger reflexion sample, (2) a decision to keep `tests/legacy_llm_planner.py` as the test oracle (the mock parity tests import it) or retire those tests with it, (3) a separate branch and PR, (4) confirmation that nothing on a production path imports `ftm.external.*.investigate` (`scripts/run_final.py` reaches the agents through `ftm.graph.agents.registry`, which returns the graph classes unless `FTM_LEGACY_AGENT_LOOPS=1`).
