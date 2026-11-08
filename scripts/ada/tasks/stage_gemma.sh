#!/bin/bash
# Run a scripts/t3_rerun.sh stage (or scripts/t3_rag_bench.sh stage) on ada against Gemma 4 12B served here with tensor parallel 2 (2 GPUs, fp16).  Owner: T3 (Akshat).
#   usage: stage_gemma.sh rerun <stage> | rag <stage>        env: E15_N_PLANNER, E15_N_REACT, ARMS, ... pass through to the stage script
# The checkout is ~/tt/tt-main-git (origin/main + the merged stage PRs), the machine label is ada-2080ti-tp2, the stage script starts nothing itself (EXTERNAL_SERVER=1).
# A stage is resumable: finished chunks (summary.json) are skipped.  Results stay under ~/tt/tt-main/outputs/ ; rsync them to the PC with the machine label kept in the summaries.
source $(dirname $0)/common.sh; KIND=$1; STAGE=$2; D=$RES/stage_${KIND}_${STAGE}; mkdir -p $D; cd $S1; export PYTHONPATH=$PWD; SERVELOG=$D/serve.log
exec > >(tee -a $D/task.log) 2>&1
setsid $PY scripts/serve_llm.py --model gemma4-12b --turing --tp 2 --gpu-util 0.88 --port $PORT > $SERVELOG 2>&1 &
pid=$!; trap 'kill -- -$pid 2>/dev/null' EXIT
wait_server gemma4-12b $pid || fail "gemma tp2 did not serve (see $SERVELOG)"
cd $HOME/tt/tt-main-git; mkdir -p ${V:-outputs/t3_val_v3} outputs/injection_v3; export PYTHONPATH=$PWD PY=$PY MODEL=gemma4-12b MACHINE=ada-2080ti-tp2 EXTERNAL_SERVER=1 FTM_P0_VARIANT=v2 FTM_EDGE_SCORER=v2
export FTM_LLM_ENDPOINT FTM_LLM_API_KEY
if [ $KIND = rerun ]; then bash scripts/t3_rerun.sh $STAGE; else bash scripts/t3_rag_bench.sh $STAGE; fi || fail "stage $KIND $STAGE failed (rc $?)"
echo "STAGE_DONE $KIND $STAGE"
