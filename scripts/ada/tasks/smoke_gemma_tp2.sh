#!/bin/bash
# can vLLM 0.31.0 serve google/gemma-4-12B-it-qat-w4a16-ct with tensor-parallel 2 in fp16 on 2 RTX 2080 Ti?  2 GPUs.  Writes $RES/smoke_gemma_tp2/result.txt
source $(dirname $0)/common.sh; D=$RES/smoke_gemma_tp2; mkdir -p $D; cd $S1; export PYTHONPATH=$PWD; SERVELOG=$D/serve.log
exec > >(tee $D/result.txt) 2>&1
echo "gpus: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tr '\n' ';')"
setsid $PY scripts/serve_llm.py --model gemma4-12b --turing --tp 2 --gpu-util 0.88 --port $PORT > $SERVELOG 2>&1 &
pid=$!
if wait_server gemma4-12b $pid; then
  echo "RESULT gemma_tp2 SERVES"
  nvidia-smi --query-gpu=memory.used --format=csv
  for i in 1 2 3; do
    s=$(date +%s.%N)
    curl -s -H "Authorization: Bearer $VLLM_API_KEY" -H "Content-Type: application/json" $FTM_LLM_ENDPOINT/chat/completions \
      -d '{"model":"gemma4-12b","messages":[{"role":"user","content":"Answer with exactly one letter: A = yes, B = no. Is Paris in France?"}],"max_tokens":1,"logprobs":true,"top_logprobs":5,"temperature":0}' | head -c 500; echo
    echo "RESULT request_$i seconds $(echo "$(date +%s.%N) - $s" | bc)"
  done
  # a 6,000-token prompt (the planner's cap) to see prefill time and that the KV cache holds it
  $PY - <<'PY'
import json, os, time, urllib.request
body = {"model": "gemma4-12b", "messages": [{"role": "user", "content": ("transfer 17 amount 120.5 USD wire; " * 600) + "Answer with exactly one letter: A = yes, B = no. Is that a lot?"}], "max_tokens": 1, "logprobs": True, "top_logprobs": 5, "temperature": 0}
rq = urllib.request.Request(os.environ["FTM_LLM_ENDPOINT"] + "/chat/completions", json.dumps(body).encode(), {"Authorization": "Bearer " + os.environ["VLLM_API_KEY"], "Content-Type": "application/json"})
t = time.time(); r = json.load(urllib.request.urlopen(rq, timeout=600)); print("RESULT long_prompt tokens", r["usage"], "seconds", round(time.time() - t, 2))
PY
else echo "RESULT gemma_tp2 DOES_NOT_SERVE"; RC=1; fi
grep -i -E "marlin|kernel|backend|dtype|error|KV cache|cuda graph" $SERVELOG | head -20
kill -- -$pid 2>/dev/null; wait $pid 2>/dev/null
[ "${RC:-0}" = 0 ] || fail "gemma tp2 did not serve (see $SERVELOG)"
