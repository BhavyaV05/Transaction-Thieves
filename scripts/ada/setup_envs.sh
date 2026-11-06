#!/bin/bash
# Build the two python 3.12 venvs on the NODE-LOCAL SSD (/ssd_scratch, 440 MB/s; /tmp is a slower shared disk), then (NOT tar them: the NFS home writes at ~5 MB/s).
# Why: writing thousands of small files to the NFS home (/home2) is ~100x slower than local /tmp (a plain pip install of torch took >40 min there).
# Run as a job step inside the allocation:   srun --jobid=$JID --overlap -c 8 --mem=24G bash setup_envs.sh {gnn|vllm}
# Owner: T3 (Akshat).
set -ex
L=/ssd_scratch/$USER; mkdir -p $L
export PIP_CACHE_DIR=$L/pipcache
PY312=/usr/local/apps/python-3.12.4/bin/python3.12
case $1 in
gnn)   # venv312: GNN (torch 2.8.0 + cu128 + PyG 2.7.0 + pyg_lib/scatter/sparse), QLoRA training stack, small data libs
  rm -rf $L/venv312; $PY312 -m venv $L/venv312; P=$L/venv312/bin/python
  $P -m pip install -q --upgrade pip
  $P -m pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # the nvidia-* wheels come from the same fast index
  $P -m pip install -q torch_geometric==2.7.0 polars pyarrow pandas scikit-learn pybind11 lightgbm tqdm pyyaml pytest omegaconf requests huggingface_hub
  $P -m pip install -q pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
  $P -m pip install -q wandb transformers==5.21.0 peft==0.21.2 trl==1.16.0 bitsandbytes==0.50.2 accelerate==1.15.0 datasets==5.1.0
  $P -c "import torch,torch_geometric,torch_scatter,torch_sparse,pyg_lib,polars,transformers,peft,bitsandbytes;print(torch.__version__,torch.cuda.is_available())" ;;
vllm)  # venv312_vllm: vLLM 0.31.0 cu129 wheel (the PyPI default wheel is cu130 and needs driver >= 580; ada has 570) + torch 2.13.0+cu129
  rm -rf $L/venv312_vllm; $PY312 -m venv $L/venv312_vllm; P=$L/venv312_vllm/bin/python
  $P -m pip install -q --upgrade pip
  $P -m pip install -q torch==2.13.0 torchaudio==2.11.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu129
  $P -m pip install -q "https://wheels.vllm.ai/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm-0.31.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl" --extra-index-url https://download.pytorch.org/whl/cu129
  $P -m pip install -q polars pyyaml omegaconf pandas pyarrow requests
  $P -m pip list 2>/dev/null | grep -i -E "^(torch|vllm|flashinfer|transformers|triton|xformers|compressed)" ;;
esac
