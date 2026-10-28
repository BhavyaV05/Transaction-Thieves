#!/usr/bin/env bash
# Serve one model and run scripts/run_llm_agent.py over a split in chunks of N alerts (T3; teacher runs, library builds).
# Finished chunks (summary.json present) are skipped, so a stopped run resumes where it left off. The server is stopped
# at the end. Same environment as scripts/e18_queue.sh (vLLM venv, FTM_HF_HOME, .env key, loopback endpoint).
#
#   scripts/chunked_runs.sh gemma4-12b outputs/t3_teacher/runs train 1983 50 --keep-prompts
#
# Args: model profile, output dir, split, total alerts, chunk size, then extra run_llm_agent.py arguments.
set -uo pipefail
model=$1; out=$2; split=$3; total=$4; chunk=$5; shift 5
cd "$(dirname "$0")/.."
export PATH=$HOME/projects/vllm-venv/bin:$PATH FTM_HF_HOME=${FTM_HF_HOME:-$HOME/hf_cache} HF_HUB_OFFLINE=1
set -a; source .env; set +a
export FTM_LLM_ENDPOINT=http://127.0.0.1:8000/v1
PY=.venv/bin/python
mkdir -p "$out"

setsid $PY scripts/serve_llm.py --model "$model" > "$out/serve.log" 2>&1 &
spid=$!
t0=$(date +%s)
until curl -sf -H "Authorization: Bearer $FTM_LLM_API_KEY" "$FTM_LLM_ENDPOINT/models" | grep -q "\"$model\""; do
  kill -0 $spid 2>/dev/null || { echo "$(date -Is) $model: server died (see $out/serve.log)"; exit 1; }
  (( $(date +%s) - t0 > 1200 )) && { echo "$(date -Is) $model: did not load"; kill -- -$spid; exit 1; }
  sleep 5
done
echo "$(date -Is) $model: loaded in $(( $(date +%s) - t0 ))s"

i=0
for (( off=0; off<total; off+=chunk )); do
  c=$(printf "c%03d" $i); i=$((i + 1))
  if [[ -f $out/$c/summary.json ]]; then continue; fi
  rm -rf "${out:?}/$c"
  $PY scripts/run_llm_agent.py --system planner --mode compose --alerts "$split" --data hi_medium --n "$chunk" \
      --offset "$off" --llm endpoint --toolbox harness --model "$model" --machine t3-5070ti \
      --run-id "$c" --run-dir "$out/$c" "$@" > "$out/$c.log" 2>&1
  echo "$(date -Is) $c (offset $off): exit $?"
done
kill -- -$spid 2>/dev/null; wait $spid 2>/dev/null
echo "$(date -Is) $model: done"
