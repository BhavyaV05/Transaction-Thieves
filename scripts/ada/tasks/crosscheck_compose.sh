#!/bin/bash
# Cross-machine check: the recorded PC run outputs/t3_val_v3/fit/ours_compose/c000 (commit f949bf3, Gemma 4 12B on the 5070 Ti) reproduced on ada, first 20 alerts of val `fit` (their ids: ~/tt/crosscheck_ids.txt = scripts/ada/crosscheck_ids.txt; `--n 20 --offset 0` draws a different 20 than the first 20 of the n=80 run),
# same arguments (scripts/t3_rerun.sh run_chunks: --profile tuned, planner compose, library shape_gemma_train.json), Gemma served with tensor parallel 2 in fp16 on two 2080 Ti.
# Two runs on the same server: serial (concurrency 1) and concurrency 4 (the recorded run's setting).  2 GPUs.  Code: ~/tt/tt-f949 (git archive of f949bf3), data copied there.
source $(dirname $0)/common.sh; D=$RES/crosscheck_compose; mkdir -p $D; cd $S1; export PYTHONPATH=$PWD; SERVELOG=$D/serve.log
exec > >(tee -a $D/task.log) 2>&1
setsid $PY scripts/serve_llm.py --model gemma4-12b --turing --tp 2 --gpu-util 0.88 --port $PORT > $SERVELOG 2>&1 &
pid=$!; trap 'kill -- -$pid 2>/dev/null' EXIT
wait_server gemma4-12b $pid || fail "gemma tp2 did not serve (see $SERVELOG)"
grep -E "KV cache|concurrency|Loading weights took|Model loading took|backend" $SERVELOG | head -12
cd $HOME/tt/tt-f949; export PYTHONPATH=$PWD FTM_P0_VARIANT=v2 FTM_EDGE_SCORER=v2
for conc in 1 4; do
  [ -f $D/conc$conc/summary.json ] && continue
  rm -rf $D/conc$conc
  $PY scripts/run_llm_agent.py --alerts val --val-part fit --data hi_medium --n 20 --alert-ids $HOME/tt/crosscheck_ids.txt --llm endpoint --toolbox harness --serving-stack vllm --machine ada-2080ti-tp2 \
      --seed 0 --profile tuned --concurrency $conc --model gemma4-12b --run-id conc$conc --run-dir $D/conc$conc --system planner --mode compose \
      --library outputs/t3_library/frozen_v3/shape_gemma_train.json > $D/conc$conc.log 2>&1 || fail "run_llm_agent concurrency $conc (see $D/conc$conc.log)"
  [ -f $D/conc$conc/summary.json ] || fail "no summary.json for concurrency $conc"
  echo "done concurrency $conc"
done
nvidia-smi --query-gpu=memory.used --format=csv
