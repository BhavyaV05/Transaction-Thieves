#!/bin/bash
# GNN time-respecting re-score on ada (T1 audit finding 19): one job step, one GPU.  Usage: gnn_causal.sh {multi_gnn|mega_gnn} [batch_size]
# srun --jobid=$JID --overlap -n1 --gres=gpu:1 -c 8 --mem=44G bash ~/tt/ada_scripts/gnn_causal.sh multi_gnn 2048
# Owner: T3 (Akshat).  Reads train+val rows only (the staged transactions.parquet has no test rows); writes data/scores/hi_medium/p0_<model>_causal*.parquet.
set -x
MODEL=$1; BS=${2:-2048}
START=$(date +%s)
L=/ssd_scratch/$USER; [ -d $L/venv312 ] || { echo "venv missing: run setup_envs.sh gnn"; exit 1; }
PY=$L/venv312/bin/python
cd ~/tt/tt-int
export PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
if [ "$MODEL" = mega_gnn ]; then
  CLONE=~/tt/t1_external/MEGA-GNN; cd $CLONE
  for X in ports_cpp negative_sampling; do
    ls $X*.so >/dev/null 2>&1 || g++ -O3 -shared -std=c++17 -fPIC $($PY -m pybind11 --includes) $X.cpp -o $X$($PY -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")
  done; ls *.so; cd ~/tt/tt-int
else CLONE=~/tt/t1_external/Multi-GNN; fi
nvidia-smi --query-gpu=name,memory.total --format=csv
$PY scripts/train_gnn.py --our-split --data hi_medium --mode see_everything --model $MODEL --clone $CLONE --causal --resume --max-epochs 0 --batch-size $BS --workers 0
echo "WALL_SECONDS $(( $(date +%s) - START )) model=$MODEL bs=$BS gpu=RTX2080Ti"
