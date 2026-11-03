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

Timings (job 54898, gnode076, 24 cores): venv 1 s; torch 105 s; vLLM wheel 5.7 min (cumulative 450 s); data + QLoRA + S1 stack 7 s (cached); torch_scatter build 5.5 min and torch_sparse 7 min (cumulative 787 s and 1218 s, run in the background while vLLM downloads); import check and MEGA extensions 45 s; **bootstrap wall 1366 s (22.8 min) without pyg_lib**. pyg_lib build time: see the note below.
