#!/usr/bin/env bash
# E18 queue on the reference machine (T3, docs/t3_e18.md "Per-candidate procedure"): for each candidate in turn, serve
# it, record load time (launch to the endpoint listing the model) and peak GPU memory, run its stage, stop the server.
# The 16 GB GPU holds one candidate at a time, so this is strictly sequential. Weights must already be in FTM_HF_HOME.
#
#   scripts/e18_queue.sh stage1 eval qwen3.8-27b-q3 gpt-oss-20b qwen3-14b
#   scripts/e18_queue.sh stage2 all  <top-3 names>
#   E18_OUT=outputs/e18_sensitivity/gptoss_out1024 E18_ARGS='--max-output-tokens 1024' scripts/e18_queue.sh stage1 eval gpt-oss-20b
#
# Writes outputs/e18/<stage>/load_<name>.json and outputs/e18/<stage>/{serve,run}_<name>.log. A candidate that does not
# load is recorded as failed (load_<name>.json, the server log) and skipped, as the procedure says.
set -uo pipefail
stage=$1; part=$2; shift 2
cd "$(dirname "$0")/.."
export PATH=$HOME/projects/vllm-venv/bin:$PATH FTM_HF_HOME=${FTM_HF_HOME:-$HOME/hf_cache} HF_HUB_OFFLINE=1
set -a; source .env; set +a
export FTM_LLM_ENDPOINT=http://127.0.0.1:8000/v1
root=${E18_OUT:-outputs/e18}; out=$root/$stage; mkdir -p "$out"
PY=.venv/bin/python
LOAD_TIMEOUT_S=${LOAD_TIMEOUT_S:-1200}

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

for name in "$@"; do
  dl=outputs/e18/dl_$name.log  # a download still running in another shell: wait for it, so load time excludes it
  while [[ -f $dl ]] && ! grep -q "^exit " "$dl"; do sleep 30; done
  if [[ -f $dl ]] && ! grep -q "^exit 0" "$dl"; then echo "$(date -Is) $name: download failed ($dl), skipped"; continue; fi
  base=$(gpu_used); t0=$(date +%s)
  setsid $PY scripts/serve_llm.py --model "$name" > "$out/serve_$name.log" 2>&1 &
  spid=$!
  ok=0; peak=$base
  while (( $(date +%s) - t0 < LOAD_TIMEOUT_S )); do
    kill -0 $spid 2>/dev/null || break
    if curl -sf -H "Authorization: Bearer $FTM_LLM_API_KEY" "$FTM_LLM_ENDPOINT/models" | grep -q "\"$name\""; then ok=1; break; fi
    u=$(gpu_used); (( u > peak )) && peak=$u
    sleep 5
  done
  load_s=$(( $(date +%s) - t0 ))
  if (( ok == 0 )); then
    echo "{\"candidate\": \"$name\", \"loaded\": false, \"load_s\": $load_s, \"peak_mib\": $peak, \"baseline_mib\": $base}" > "$out/load_$name.json"
    echo "$(date -Is) $name: did not load in ${load_s}s (see $out/serve_$name.log)"
    kill -- -$spid 2>/dev/null; wait $spid 2>/dev/null; sleep 10
    continue
  fi
  echo "$(date -Is) $name: loaded in ${load_s}s"
  ( while kill -0 $spid 2>/dev/null; do gpu_used; sleep 10; done ) > "$out/vram_$name.txt" &
  vpid=$!
  # shellcheck disable=SC2086  # E18_ARGS is a word list on purpose
  $PY scripts/run_e18.py "$stage" --val-part "$part" --only "$name" --machine t3-5070ti --out "$root" \
      --clusters outputs/e18/clusters.json ${E18_ARGS:-} > "$out/run_$name.log" 2>&1
  rc=$?
  kill $vpid 2>/dev/null
  run_peak=$(sort -n "$out/vram_$name.txt" | tail -1)
  (( ${run_peak:-0} > peak )) && peak=$run_peak
  echo "{\"candidate\": \"$name\", \"loaded\": true, \"load_s\": $load_s, \"peak_mib\": $peak, \"baseline_mib\": $base, \"run_exit\": $rc}" > "$out/load_$name.json"
  echo "$(date -Is) $name: $stage run exit $rc, peak GPU ${peak} MiB (desktop baseline ${base} MiB)"
  kill -- -$spid 2>/dev/null; wait $spid 2>/dev/null; sleep 10  # let the GPU memory free before the next load
done
