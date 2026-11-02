#!/bin/bash
# Smoke test on ada: can vLLM 0.31.0 serve google/gemma-4-12B-it-qat-w4a16-ct with tensor-parallel 2 in fp16 on two RTX 2080 Ti (sm_75, 11 GB each)?
# Needs 2 GPUs in the step:  srun --jobid=$JID --overlap --gres=gpu:2 -c 8 --mem=40G bash scripts/ada/smoke_gemma_tp2.sh
# Starts the server with the repo's gemma4-12b profile flags minus fp8 KV (turing_args), sends 3 chat requests with one-token logprobs, prints RESULT lines. Owner: T3 (Akshat).
cd ~/tt/tt-s1bench
export HF_HOME=/ssd_scratch/$USER/hf HF_HUB_OFFLINE=1 PYTHONPATH=$PWD PATH=/ssd_scratch/$USER/venv312_vllm/bin:$PATH
export VLLM_API_KEY=smoke FTM_LLM_API_KEY=smoke
LOG=/ssd_scratch/$USER/gemma_tp2.log
setsid python scripts/serve_llm.py --model gemma4-12b --turing --tp 2 --gpu-util 0.88 --port 8010 > $LOG 2>&1 &
pid=$!; t0=$(date +%s)
until curl -sf -H "Authorization: Bearer smoke" localhost:8010/v1/models >/dev/null; do
  kill -0 $pid 2>/dev/null || { echo "RESULT gemma_tp2 server_died"; tail -60 $LOG; exit 1; }
  (( $(date +%s) - t0 > 1500 )) && { echo "RESULT gemma_tp2 timeout"; tail -40 $LOG; kill -- -$pid; exit 1; }
  sleep 5
done
echo "RESULT gemma_tp2 up_after_s=$(( $(date +%s) - t0 ))"
nvidia-smi --query-gpu=memory.used --format=csv
for i in 1 2 3; do
  /usr/bin/time -f "RESULT req_$i wall_s=%e" curl -s -H "Authorization: Bearer smoke" -H "Content-Type: application/json" localhost:8010/v1/chat/completions \
    -d '{"model":"gemma4-12b","messages":[{"role":"user","content":"Answer with exactly one letter: A = yes, B = no. Is Paris in France?"}],"max_tokens":1,"logprobs":true,"top_logprobs":5,"temperature":0}' | head -c 400; echo
done
grep -i -E "backend|marlin|kernel|fp16|float16|dtype" $LOG | head -8
kill -- -$pid 2>/dev/null
