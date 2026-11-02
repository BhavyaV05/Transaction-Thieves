#!/bin/bash
# S1BENCH on one 2080 Ti: vLLM serves Qwen3-4B-AWQ + the LoRA adapters; usage: s1_server.sh {sys1|base|soft}.  1 GPU.
source $(dirname $0)/common.sh; ARM=$1; START=$(date +%s)
cd $S1; export PYTHONPATH=$PWD; OUT=$RES/s1bench; mkdir -p $OUT/logs $RES/s1v1; SERVELOG=$OUT/logs/serve_$ARM.log
echo "gpu: $GPU_NAME"
setsid $PY scripts/serve_llm.py --model qwen3-4b-fast --turing --gpu-util 0.85 --lora-only fast3-s0,fast3soft-s0 --port $PORT > $SERVELOG 2>&1 &
pid=$!; trap 'kill -- -$pid 2>/dev/null' EXIT
wait_server qwen3-4b-fast $pid || exit 1
part_a() { local tag=$1 m=$2
  $PY scripts/s1bench_run.py ext   --tag $tag --model $m --tasks all --perm --workers 32 --resume --out $OUT > $OUT/logs/$tag.ext.log 2>&1
  $PY scripts/s1bench_run.py s1mb  --tag $tag --model $m --benchmarks all --perm --workers 32 --resume --out $OUT > $OUT/logs/$tag.s1mb.log 2>&1
  $PY scripts/s1bench_run.py latency --tag $tag --model $m --tasks all --latency-n 200 --out $OUT > $OUT/logs/$tag.lat.log 2>&1
  echo "$(date -Is) part A done: $tag ($(( $(date +%s) - START ))s)"; }
part_b() { local arm=$1 m=$2 d=$RES/s1v1/$1; mkdir -p $d
  [ -s $d/fast.jsonl ] || $PY scripts/score_fast_baseline.py score --tier fast --model $m --workers 16 --out $d/fast.jsonl > $d/score.log 2>&1
  [ -s $d/latency_rows.jsonl ] || $PY scripts/score_fast_baseline.py score --tier fast --model $m --workers 1 --limit 300 --out $d/latency_rows.jsonl > $d/latency.log 2>&1
  $PY - $d <<'PY'
import json, sys
d = sys.argv[1]; last = [l for l in open(f"{d}/latency.log") if l.startswith("{")][-1]; r = json.loads(last)
json.dump({"ms_per_decision": 1000 * r["latency_median_s"], "rows": r["rows_scored"], "workers": 1}, open(f"{d}/latency.json", "w"))
PY
  echo "$(date -Is) part B done: $arm"; }
case $ARM in
  sys1) part_b untuned_4b qwen3-4b-fast; part_b tuned_s0 fast3-s0; part_b tuned_soft_s0 fast3soft-s0; part_a system1 fast3-s0
        $PY scripts/s1bench_run.py ext --tag system1_listed --model fast3-s0 --variant listed --tasks banking77 clinc150_oos hwu64 --perm --workers 32 --resume --out $OUT > $OUT/logs/system1_listed.ext.log 2>&1 ;;
  base) part_a qwen3-4b qwen3-4b-fast ;;
  soft) part_a system1_soft fast3soft-s0 ;;
esac
echo "WALL_SECONDS $(( $(date +%s) - START )) arm=$ARM gpu=$GPU_NAME" | tee -a $OUT/wall_s1_$ARM.txt
