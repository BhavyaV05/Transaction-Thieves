#!/bin/bash
# Real-model parity of the pre-graph (legacy) loops against the graph versions, Gemma 4 12B TP2 fp16, 2 GPUs.  Code: ~/tt/tt-rw (git archive of t3/rewrite 49e62de, == main fab5b47 for these files).
# Every arm runs twice as graph (g1, g2: the run-to-run noise floor of the serving stack) and once as legacy (l1); parity = legacy-vs-graph agreement is not lower than graph-vs-graph.
# usage: parity_real.sh planner | <external system names ...>
source $(dirname $0)/common.sh; D=$RES/parity_real; mkdir -p $D; cd $S1; export PYTHONPATH=$PWD; SERVELOG=$D/serve.log
exec > >(tee -a $D/task.log) 2>&1
setsid $PY scripts/serve_llm.py --model gemma4-12b --turing --tp 2 --gpu-util 0.88 --port $PORT > $SERVELOG 2>&1 &
pid=$!; trap 'kill -- -$pid 2>/dev/null' EXIT
wait_server gemma4-12b $pid || fail "gemma tp2 did not serve (see $SERVELOG)"
cd $HOME/tt/tt-rw; export PYTHONPATH=$PWD FTM_P0_VARIANT=v2 FTM_EDGE_SCORER=v2
# legacy-equivalent configuration for the planner arms (--no-select-menu: the pre-graph loop shows the WHOLE routed menu, ~204 questions, which overflows the 8k context by itself; that overflow is why the ranked top-8 menu exists, docs/t3_prompt_budget.md): the profile "none" with the pieces of the tuned profile that the pre-graph loop has (evidence view, k, p0 v2 via env, realistic back-test table, fitted fusion, decision threshold, prompt cap); NOT the graph-only options (alert-aware rules, ranked top-8 menu, cannot-cross stop, evidence floor)
BT=configs/fusion/headroom_v1_table.jsonl
LEG="--profile none --rules fixed --evidence-view scorer_pcst --edge-scorer v2 --k 40 --backtest $BT --backtest-mode realistic --fusion-params configs/fusion/headroom_v1.json --decision configs/decision/headroom_v1.yaml --max-prompt-tokens 6750 --no-select-menu"
COMMON="--alerts val --val-part fit --data hi_medium --llm endpoint --toolbox harness --serving-stack vllm --machine ada-2080ti-tp2 --seed 0 --concurrency 1 --model gemma4-12b"
for sys in "$@"; do
  if [ $sys = planner ]; then ARGS="$LEG --n 20 --alert-ids $HOME/tt/crosscheck_ids.txt --system planner --mode compose --library outputs/t3_library/frozen_v3/shape_gemma_train.json"
  else ARGS="--profile tuned --concurrency 4 --n 8 --alert-ids $HOME/tt/parity_ids8.txt --system $sys --library outputs/t3_library/frozen_v3/shape_gemma_train.json"; fi
  for arm in g1 g2 l1; do
    [ -f $D/$sys/$arm/summary.json ] && continue
    rm -rf $D/$sys/$arm; mkdir -p $D/$sys
    if [ $arm = l1 ]; then
      if [ $sys = planner ]; then RUNNER="scripts/ada/legacy_parity_planner.py"; else RUNNER="scripts/run_llm_agent.py"; export FTM_LEGACY_AGENT_LOOPS=1; fi
    else RUNNER="scripts/run_llm_agent.py"; unset FTM_LEGACY_AGENT_LOOPS; fi
    $PY $RUNNER $COMMON $ARGS --run-id $arm --run-dir $D/$sys/$arm > $D/$sys/$arm.log 2>&1 || echo "FAILED $sys $arm (see $D/$sys/$arm.log)"
    unset FTM_LEGACY_AGENT_LOOPS
    [ -f $D/$sys/$arm/summary.json ] && echo "done $sys $arm"
  done
done
echo "PARITY_TASK_END"
