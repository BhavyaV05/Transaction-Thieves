#!/bin/bash
# Gemma 4 12B large tier on Part A with tensor parallel 2 (only meaningful if smoke_gemma_tp2 served).  2 GPUs.
source $(dirname $0)/common.sh; START=$(date +%s); cd $S1; export PYTHONPATH=$PWD; OUT=$RES/s1bench; mkdir -p $OUT/logs; SERVELOG=$OUT/logs/serve_gemma.log
setsid $PY scripts/serve_llm.py --model gemma4-12b --turing --tp 2 --gpu-util 0.88 --port $PORT > $SERVELOG 2>&1 &
pid=$!; trap 'kill -- -$pid 2>/dev/null' EXIT; wait_server gemma4-12b $pid || exit 1
for c in ext s1mb; do
  if [ $c = ext ]; then $PY scripts/s1bench_run.py ext --tag gemma4-12b --model gemma4-12b --tasks all --perm --workers 8 --resume --out $OUT > $OUT/logs/gemma.ext.log 2>&1 || fail "gemma ext"
  else $PY scripts/s1bench_run.py s1mb --tag gemma4-12b --model gemma4-12b --benchmarks all --perm --workers 8 --resume --out $OUT > $OUT/logs/gemma.s1mb.log 2>&1 || fail "gemma s1mb"; fi
done
$PY scripts/s1bench_run.py latency --tag gemma4-12b --model gemma4-12b --tasks all --latency-n 100 --out $OUT > $OUT/logs/gemma.lat.log 2>&1 || fail "gemma latency"
echo "WALL_SECONDS $(( $(date +%s) - START )) gemma tp2 gpu=$GPU_NAME" | tee -a $OUT/wall_s1_gemma.txt
