#!/bin/bash
# Build the whole ada environment on the compute node's local SSD (the home NFS writes at ~5 MB/s).  Usage: source-able; run as `bash bootstrap_ssd.sh <scratch_dir>`.
# Owner: T3 (Akshat).  Leaves $SC/venv (python 3.12 + torch 2.13 cu129 + vLLM 0.31 + PyG + QLoRA/S1 stack) and $SC/hf; extension wheels cached in ~/tt/wheels.
set -x
SC=${1:?scratch dir}; mkdir -p $SC/tmp $SC/uvcache $SC/hf $HOME/tt/wheels $HOME/tt/results
export UV_CACHE_DIR=$SC/uvcache TMPDIR=$SC/tmp UV_LINK_MODE=hardlink UV_HTTP_TIMEOUT=300 UV_INDEX_STRATEGY=unsafe-best-match
mkdir -p $SC/bin; curl -sSL -m 120 https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-musl.tar.gz | tar xz --strip-components=1 -C $SC/bin || cp $HOME/bin/uv $SC/bin/uv   # NFS cold reads are ~1 MB/s: take uv from the internet to the SSD
export PATH=$SC/bin:$PATH
T0=$(date +%s); lap() { echo "LAP $1 $(( $(date +%s) - T0 ))s"; }
V=$SC/venv; P=$V/bin/python; CU="--extra-index-url https://download.pytorch.org/whl/cu129"
CON=$HOME/tt/ada_scripts/vllm_constraints.txt
uv venv -q -p /usr/local/apps/python-3.12.4/bin/python3.12 $V; lap venv
# background: weights the tasks need (HF_HOME on the SSD).  hf comes from the venv once huggingface-hub is installed (below).
VW="https://wheels.vllm.ai/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm-0.31.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
uv pip install -q -p $P --index-url https://download.pytorch.org/whl/cu129 --extra-index-url https://pypi.org/simple torch==2.13.0 torchaudio==2.11.0 torchvision==0.28.0 || exit 11; lap torch
( export HF_HOME=$SC/hf HF_HUB_DISABLE_PROGRESS_BARS=1
  uv pip install -q -p $P "huggingface_hub[hf_xet]" && for m in "Qwen/Qwen3-4B-AWQ 74d4bd2bd4bff9cafc9345221320bffb08b406a3" "google/gemma-4-12B-it-qat-w4a16-ct 1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee" "Qwen/Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218"; do
    set -- $m; $V/bin/hf download $1 --revision $2 >> $HOME/tt/results/prefetch.log 2>&1; echo "$(date -Is) prefetched $1" >> $HOME/tt/results/prefetch.log; done ) > $SC/prefetch.out 2>&1 &
# PyG + build deps, then find out whether the temporal sampler needs the compiled extensions
uv pip install -q -p $P torch_geometric==2.7.0 setuptools wheel ninja cmake pybind11 scipy numpy || exit 12; lap pyg
cat > $SC/tmp/temporal_check.py <<'PY'
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE
print("WITH_PYG_LIB", WITH_PYG_LIB, "WITH_TORCH_SPARSE", WITH_TORCH_SPARSE)
d = HeteroData(); n = 50; E = 400
ei = torch.randint(0, n, (2, E)); t = torch.arange(E)
d["node"].x = torch.zeros(n, 1); d["node", "to", "node"].edge_index = ei; d["node", "to", "node"].t_sample = t; d["node", "to", "node"].edge_attr = torch.randn(E, 3)
d["node", "rev_to", "node"].edge_index = ei.flip(0); d["node", "rev_to", "node"].t_sample = t; d["node", "rev_to", "node"].edge_attr = torch.randn(E, 3)
pos = torch.arange(100, 200)
lo = LinkNeighborLoader(d, num_neighbors=[10, 10], batch_size=32, edge_label_index=(("node", "to", "node"), ei[:, pos]), edge_label=torch.zeros(100), time_attr="t_sample", edge_label_time=t[pos], temporal_strategy="uniform")
b = next(iter(lo)); print("TEMPORAL_OK", max(int(b[k].t_sample.max()) for k in b.edge_types if "t_sample" in b[k]), "<=", int(t[pos].max()))
PY
$P $SC/tmp/temporal_check.py > $SC/tmp/temporal_noext.txt 2>&1; cat $SC/tmp/temporal_noext.txt; lap temporal_check_noext
# compiled extensions: reuse cached wheels, else build (CPU kernels; MEGA-GNN imports torch_scatter directly; sampler per the check above)
W=$HOME/tt/wheels
build_ext() {
  export FORCE_ONLY_CPU=1 MAX_JOBS=${MAX_JOBS:-10} FORCE_CUDA=0 CUDAARCHS=75 TORCH_CUDA_ARCH_LIST=7.5 PATH=$V/bin:/usr/local/apps/cmake-3.31.7/bin:/usr/local/cuda/bin:$PATH   # venv python first on PATH (pyg_lib cmake asks `python` for the C++ ABI; wrong ABI = undefined symbol Node::name at import); pyg_lib: torch cmake config needs CUDAARCHS+nvcc even for a CPU build; ~30 GB RAM
  mkdir -p $SC/wheelbuild; cd $SC/wheelbuild
  for pkg in torch_scatter torch_sparse; do ls $W/$pkg-*.whl >/dev/null 2>&1 || { $P -m pip --version >/dev/null 2>&1 || uv pip install -q -p $P pip; $P -m pip wheel -q --no-deps --no-build-isolation -w $SC/wheelbuild $pkg && cp -n $SC/wheelbuild/$pkg-*.whl $W/; }; lap built_$pkg; done
  if ! grep -q TEMPORAL_OK $SC/tmp/temporal_noext.txt; then ls $W/pyg_lib-*.whl >/dev/null 2>&1 || { rm -rf pyg-lib; git clone -q --depth 1 --recursive --shallow-submodules https://github.com/pyg-team/pyg-lib.git && $P -m pip wheel -q --no-deps --no-build-isolation -w $SC/wheelbuild ./pyg-lib && cp -n $SC/wheelbuild/pyg_lib-*.whl $W/; }; lap built_pyg_lib; fi
  cd $SC
}
( build_ext ) > $SC/tmp/build_ext.log 2>&1 &
BPID=$!
uv pip install -q -p $P -c $CON "$VW" torch==2.13.0 torchaudio==2.11.0 torchvision==0.28.0 --extra-index-url https://download.pytorch.org/whl/cu129 || exit 13; lap vllm
# torchcodec: the PyPI linux wheel (0.17.0) is the CUDA 13 build; its libtorchcodec_image.so needs libnvrtc.so.13, and vLLM imports it unconditionally from vllm/multimodal/media/audio.py, so
# `vllm serve` died with "OSError: libnvrtc.so.13".  The driver here is 570 (CUDA 12.8), which cannot run cu13 anyway, and vLLM only uses torchcodec to decode audio (disabled: --limit-mm-per-prompt audio 0).  Take the CPU build.
uv pip install -q -p $P --reinstall --no-deps --index-url https://download.pytorch.org/whl/cpu "torchcodec==0.17.0+cpu" || exit 15; lap torchcodec_cpu
$P -c "import torchcodec; print('torchcodec', torchcodec.__version__)" || exit 16
$P $HOME/tt/ada_scripts/patch_vllm_turing.py $V || exit 17; lap vllm_t75_patch   # Gemma 4 head_size-512 layers: Triton attention needs 96 KB shared memory, an sm_75 GPU has 64 KB
uv pip install -q -p $P polars==2.0.0 pyarrow pandas scikit-learn lightgbm tqdm pyyaml pytest omegaconf requests wandb langgraph==1.2.19 lark simpy duckdb hydra-core rank-bm25 "dspy[optuna]>=3.4,<3.5" ag2 || exit 14   # langgraph..rank-bm25: ftm/ (the planner graph, run_llm_agent.py)
bash $HOME/tt/ada_scripts/build_pcst.sh $SC > $SC/tmp/pcst.log 2>&1 || { tail -5 $SC/tmp/pcst.log; exit 18; }; lap pcst_fast_patched   # the PyPI wheel returns wrong index arrays; sdist + ftm/retrieval/pcst_fast_binding.patch
uv pip install -q -p $P peft==0.21.2 trl==1.16.0 bitsandbytes==0.50.2 accelerate==1.15.0 datasets==5.1.0 || uv pip install -q -p $P peft trl bitsandbytes accelerate datasets
uv pip install -q -p $P sentence-transformers gliner2==2.1.0 laya==0.3.20 pydantic; lap rest
wait $BPID; cat $SC/tmp/build_ext.log; lap ext_build_done
uv pip install -q -p $P --no-deps $W/*.whl; lap ext_installed
$P $SC/tmp/temporal_check.py; $P -c "import torch_scatter,torch_sparse;print('scatter/sparse ok')"
$P -c "import torch,torch_geometric,vllm,transformers,peft,bitsandbytes,polars,lightgbm;print('IMPORT_OK',torch.__version__,torch.version.cuda,torch_geometric.__version__,vllm.__version__,transformers.__version__)"
cd $HOME/tt/t1_external/MEGA-GNN
for X in ports_cpp negative_sampling; do rm -f $X.cpython-*.so; g++ -O3 -shared -std=c++17 -fPIC $($P -m pybind11 --includes) $X.cpp -o $X$($P -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))"); done; ls *.so; lap mega_ext
uv pip freeze -p $P > $HOME/tt/venv_ssd_freeze.txt; lap DONE
