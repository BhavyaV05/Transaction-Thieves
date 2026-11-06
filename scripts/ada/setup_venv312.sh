#!/bin/bash
# ONE CPU-only job on u22 (no --gres) that builds EVERYTHING into the shared-home ~/venv312, pre-stages weights and builds the MEGA-GNN C++ extensions.
#   sbatch -p u22 -A research --qos=medium -c 16 --mem=48G -t 8:00:00 -J venv312 -o ~/tt/logs/venv312_%j.out scripts/ada/setup_venv312.sh
# Owner: T3 (Akshat).  Decision (user, 2 Nov): one venv, ~/venv312 (python 3.12.4 at /usr/local/apps, visible on compute nodes only).
# Wheels are fetched to the node-local SSD first (fast), then installed from there into the NFS venv (writes there are slow: --no-compile, no cache).
set -ex
for d in /ssd_scratch /tmp; do [ "$(df --output=avail -BG $d | tail -1 | tr -dc 0-9)" -gt 60 ] && { S=$d/$USER; break; }; done  # node-local disk with room (other users fill /ssd_scratch on some nodes)
echo "scratch: $S"; mkdir -p $S/tmp $S/wheels
export PIP_CACHE_DIR=$S/pipcache TMPDIR=$S/tmp PIP_NO_COMPILE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
V=$HOME/venv312; P=$V/bin/python; W=$S/wheels
export PATH=/usr/local/apps/cmake-3.31.7/bin:$PATH
rm -rf $HOME/venv312_vllm
T0=$(date +%s); lap() { echo "LAP $1 $(( $(date +%s) - T0 ))s"; }
$P -m pip --version

# 1. torch 2.13.0+cu129 (pinned by vLLM 0.31.0; cu129 because the driver is 570) from the fast pytorch index, then the vLLM cu129 wheel
$P -m pip download -q -d $W torch==2.13.0 torchaudio==2.11.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu129 --find-links $W; lap dl_torch
VW="https://wheels.vllm.ai/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm-0.31.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
$P -m pip download -q -d $W "$VW" --extra-index-url https://download.pytorch.org/whl/cu129 --find-links $W -c $HOME/tt/ada_scripts/vllm_constraints.txt; lap dl_vllm   # constraints = the exact PyPI pins of the working local vllm-venv: without them pip backtracked for 30+ min on slow PyPI
$P -m pip install -q --no-index --find-links $W torch==2.13.0 torchaudio==2.11.0 torchvision==0.28.0; lap inst_torch
$P -m pip install -q --no-index --find-links $W -c $HOME/tt/ada_scripts/vllm_constraints.txt vllm; lap inst_vllm
# 2. the PyG stack.  There are NO prebuilt pyg_lib/torch_scatter/torch_sparse wheels for torch 2.13 (data.pyg.org stops at 2.12), so build them from source here, CPU kernels
#    only (this node has no GPU at build time; the neighbour sampler, the only thing the GNN scripts need from them, is CPU code)
$P -m pip install -q torch_geometric==2.7.0 setuptools wheel ninja cmake pybind11 scipy; lap pyg_py
cd $S
export FORCE_ONLY_CPU=1 MAX_JOBS=12 FORCE_CUDA=0
for pkg in torch_scatter torch_sparse; do $P -m pip wheel -q --no-deps --no-build-isolation -w $W $pkg || echo "BUILD_FAILED $pkg"; done; lap build_scatter_sparse
rm -rf pyg-lib; git clone -q --depth 1 --recursive --shallow-submodules https://github.com/pyg-team/pyg-lib.git && (cd pyg-lib && $P -m pip wheel -q --no-deps --no-build-isolation -w $W . || echo "BUILD_FAILED pyg_lib"); lap build_pyg_lib
$P -m pip install -q --no-index --find-links $W torch_scatter torch_sparse pyg_lib || true
# 3. everything else: data libs, the QLoRA stack, the S1BENCH open-model stack
$P -m pip install -q polars pyarrow pandas scikit-learn lightgbm tqdm pyyaml pytest omegaconf requests wandb huggingface_hub
$P -m pip install -q peft==0.21.2 trl==1.16.0 bitsandbytes==0.50.2 accelerate==1.15.0 datasets==5.1.0 || $P -m pip install -q peft trl bitsandbytes accelerate datasets
$P -m pip install -q sentence-transformers gliner2==2.1.0 laya==0.3.20 pydantic; lap inst_rest
$P -m pip check || true
$P -m pip freeze > $HOME/tt/venv312_freeze.txt; sha256sum $HOME/tt/venv312_freeze.txt; wc -l $HOME/tt/venv312_freeze.txt
$P -c "import torch,torch_geometric,vllm,transformers,peft,bitsandbytes,polars,lightgbm;print('IMPORT_OK',torch.__version__,torch.version.cuda,torch_geometric.__version__,vllm.__version__,transformers.__version__)"
# 4. temporal-sampler check on CPU: the exact call pattern of ftm.baselines.multi_gnn.causal_kwargs (hetero edge-level time_attr + edge_label_time)
$P - <<'PY'
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
d = HeteroData(); n = 50; E = 400
ei = torch.randint(0, n, (2, E)); t = torch.arange(E)
d["node"].x = torch.zeros(n, 1); d["node", "to", "node"].edge_index = ei; d["node", "to", "node"].t_sample = t; d["node", "to", "node"].edge_attr = torch.randn(E, 3)
d["node", "rev_to", "node"].edge_index = ei.flip(0); d["node", "rev_to", "node"].t_sample = t; d["node", "rev_to", "node"].edge_attr = torch.randn(E, 3)
pos = torch.arange(100, 200)
lo = LinkNeighborLoader(d, num_neighbors=[10, 10], batch_size=32, edge_label_index=(("node", "to", "node"), ei[:, pos]), edge_label=torch.zeros(100), time_attr="t_sample", edge_label_time=t[pos], temporal_strategy="uniform")
b = next(iter(lo)); tmax = max(int(b[k].t_sample.max()) for k in b.edge_types if "t_sample" in b[k]); print("TEMPORAL_OK max edge time in batch", tmax, "<= seed max", int(t[pos].max()))
from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE; print("WITH_PYG_LIB", WITH_PYG_LIB, "WITH_TORCH_SPARSE", WITH_TORCH_SPARSE)
PY
lap temporal_check
# 5. MEGA-GNN C++ extensions (pybind11, no GPU)
cd $HOME/tt/t1_external/MEGA-GNN
for X in ports_cpp negative_sampling; do rm -f $X.cpython-*.so; g++ -O3 -shared -std=c++17 -fPIC $($P -m pybind11 --includes) $X.cpp -o $X$($P -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))"); done; ls *.so; lap mega_ext
# 6. pre-download weights that fit in the 50 GB home quota (after the venv: Qwen3-4B-AWQ 2.5 GB, Gemma 4 12B w4a16 9 GB); the rest is fetched inside the GPU job
export HF_HOME=$HOME/tt/hf HF_HUB_DISABLE_PROGRESS_BARS=1
$V/bin/hf download Qwen/Qwen3-4B-AWQ --revision 74d4bd2bd4bff9cafc9345221320bffb08b406a3
$V/bin/hf download google/gemma-4-12B-it-qat-w4a16-ct --revision 1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee; lap hf
df -h $HOME | tail -1; lap DONE
