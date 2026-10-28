#!/usr/bin/env bash
# Proof that the QLoRA adapters load into the AWQ-base vLLM server (LoRA on AWQ) and answer (E16 serving): serve <profile> (qwen3-8b-lora or
# qwen3-4b-lora), send one planner-style request to the base and to each adapter, print the first 200 characters of each answer.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/t3_val
MODEL=${1:-qwen3-8b-lora}; shift || true; ADAPTERS=${*:-planner-sft-s0 planner-sft-s1 planner-sft-s2}
export PATH=$HOME/projects/vllm-venv/bin:$PATH FTM_HF_HOME=${FTM_HF_HOME:-$HOME/hf_cache} HF_HUB_OFFLINE=1
set -a; source .env; set +a
export FTM_LLM_ENDPOINT=http://127.0.0.1:8000/v1
setsid ${PY:-.venv/bin/python} scripts/serve_llm.py --model "$MODEL" > "outputs/t3_val/serve_check_$MODEL.log" 2>&1 &
spid=$!; trap 'kill -- -$spid 2>/dev/null; wait $spid 2>/dev/null' EXIT
t0=$(date +%s)
until curl -sf -H "Authorization: Bearer $FTM_LLM_API_KEY" "$FTM_LLM_ENDPOINT/models" | grep -q "\"$MODEL\""; do
  kill -0 $spid 2>/dev/null || { echo "server died"; tail -30 "outputs/t3_val/serve_check_$MODEL.log"; exit 1; }
  (( $(date +%s) - t0 > 1500 )) && { echo "no load"; exit 1; }
  sleep 5
done
echo "loaded in $(( $(date +%s) - t0 ))s"
curl -s -H "Authorization: Bearer $FTM_LLM_API_KEY" "$FTM_LLM_ENDPOINT/models" | python3 -c "import json,sys;print([m['id'] for m in json.load(sys.stdin)['data']])"
for m in "$MODEL" $ADAPTERS; do
  echo "== $m"
  curl -s -H "Authorization: Bearer $FTM_LLM_API_KEY" -H "Content-Type: application/json" "$FTM_LLM_ENDPOINT/chat/completions" \
    -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with one JSON object {\\\"action\\\": \\\"stop\\\"}.\"}],\"max_tokens\":40,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    | python3 -c "import json,sys;d=json.load(sys.stdin);print(str(d['choices'][0]['message']['content'])[:200] if 'choices' in d else d)"
done
