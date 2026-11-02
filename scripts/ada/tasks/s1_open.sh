#!/bin/bash
# open System One models on OUR decisions through their own typed interfaces.  usage: s1_open.sh gpu|cpu
source $(dirname $0)/common.sh; cd $S1; export PYTHONPATH=$PWD; OUT=$RES/s1bench/partb; mkdir -p $OUT
export HF_HOME=${HF_HOME}; [ $1 = cpu ] && export CUDA_VISIBLE_DEVICES=""
if [ $1 = gpu ]; then
  # JevK5 needs its source tree (S1BENCH_CACHE/ext/jevk5) on the path; Tev1 loads through transformers
  for m in jevk5 tev1-4b; do $PY scripts/s1bench_partb.py open --model $m --out $OUT/$m/scores.jsonl > $OUT/$m.log 2>&1 || echo "$m failed: $OUT/$m.log"; done
else
  for m in laya laya-typed-decisions; do $PY scripts/s1bench_partb.py open --model $m --cap 500 --out $OUT/$m/scores.jsonl > $OUT/$m.log 2>&1 || echo "$m failed"; done
  $PY scripts/s1bench_partb.py open --model gliner25-decide --adapter $S1/outputs/s1bench/ft/gliner25-ft/final --cap 500 --out $OUT/gliner25-ft/scores.jsonl > $OUT/gliner25-ft.log 2>&1 || echo "gliner ft failed"
fi
