#!/bin/bash
# GNN time-respecting re-score (T1 audit finding 19).  usage: gnn.sh {multi_gnn|mega_gnn}; 1 GPU.  Resumable: nothing is trained (--max-epochs 0), the checkpoint is loaded.
source $(dirname $0)/common.sh; M=$1; START=$(date +%s)
cd $HOME/tt/tt-int; export PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6
CLONE=$HOME/tt/t1_external/$([ $M = mega_gnn ] && echo MEGA-GNN || echo Multi-GNN)
echo "gpu: $GPU_NAME"
for bs in 2048 1024; do   # the reproduction batch size first; 1024 only if 2048 runs out of the 11 GB
  $PY scripts/train_gnn.py --our-split --data hi_medium --mode see_everything --model $M --clone $CLONE --causal --resume --max-epochs 0 --batch-size $bs --workers 0 && { USED=$bs; break; }
  echo "batch $bs failed"
done
[ -n "$USED" ] || exit 1
mkdir -p $RES/gnn_$M; cp data/scores/hi_medium/p0_${M}_causal* data/runs/p0_hi_medium_${M}_causal_see_everything.json $RES/gnn_$M/
echo "WALL_SECONDS $(( $(date +%s) - START )) model=$M bs=$USED gpu=$GPU_NAME" | tee $RES/gnn_$M/wall.txt
