#!/usr/bin/env bash
# Qwen servers for T3's fine-tuned models, one server at a time (the 16 GB card): E16 val runs and the E10 fast-tier scoring.
#   scripts/t3_qwen_session.sh e16        qwen3-8b-lora: prompted base + the 3 QLoRA students on the 160 val `eval` alerts -> outputs/t3_val/eval/e16_*
#   scripts/t3_qwen_session.sh fast       qwen3-4b-lora: untuned 4B + tuned seeds 0-2 score every val decision row   -> outputs/e10/<arm>/fast.jsonl
#   scripts/t3_qwen_session.sh fast3 qwen3-4b-fast    wave 4: base + adapters fast3-s0..2 score the NEW decision set (data/decisions/hi_medium_s1v1) -> outputs/s1v1/<arm>/fast.jsonl
#   scripts/t3_qwen_session.sh large3 gemma4-12b      wave 4: Gemma 4 12B large tier by token logprobs on every val row            -> outputs/s1v1/large/large_lp.jsonl
#   scripts/t3_qwen_session.sh small      qwen3-1.7b: untuned small general LLM, same rows                          -> outputs/e10/untuned_1.7b/fast.jsonl
# Latency per decision (one request at a time, 300 rows) is measured for every scored arm: outputs/e10/<arm>/latency.json.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/projects/vllm-venv/bin:$PATH FTM_HF_HOME=${FTM_HF_HOME:-$HOME/hf_cache} HF_HUB_OFFLINE=1
set -a; source .env; set +a
export FTM_LLM_ENDPOINT=http://127.0.0.1:8000/v1
PY=${PY:-.venv/bin/python}
OUT=${T3_VAL_OUT:-outputs/t3_val}
PROFILE="--profile tuned"   # production components (configs/llm/tuned.yaml)
SHAPE=outputs/t3_library/frozen/shape_gemma.json
which=$1; MODEL=$2

serve_up() {
  setsid $PY scripts/serve_llm.py --model "$MODEL" > "$OUT/serve_$MODEL.log" 2>&1 &
  spid=$!; t0=$(date +%s)
  until curl -sf -H "Authorization: Bearer $FTM_LLM_API_KEY" "$FTM_LLM_ENDPOINT/models" | grep -q "\"$MODEL\""; do
    kill -0 $spid 2>/dev/null || { echo "$(date -Is) server died (see $OUT/serve_$MODEL.log)"; exit 1; }
    (( $(date +%s) - t0 > 1500 )) && { echo "server did not load"; kill -- -$spid; exit 1; }
    sleep 5
  done
  echo "$(date -Is) $MODEL loaded in $(( $(date +%s) - t0 ))s"
}
trap 'kill -- -$spid 2>/dev/null; wait $spid 2>/dev/null' EXIT
mkdir -p "$OUT" outputs/e10 outputs/s1v1
serve_up

score_arm() {  # score_arm <arm dir name> <served model name> [root, default outputs/e10]
  local arm=$1 m=$2 d=${3:-outputs/e10}/$1
  mkdir -p "$d"
  $PY scripts/score_fast_baseline.py score --tier fast --model "$m" --workers 16 --out "$d/fast.jsonl" > "$d/score.log" 2>&1
  # latency: one request at a time on a fixed sample (the rows are the first 300 of the file's order)
  $PY scripts/score_fast_baseline.py score --tier fast --model "$m" --workers 1 --limit 300 --out "$d/latency_rows.jsonl" > "$d/latency.log" 2>&1
  $PY - "$d" <<'PYEOF'
import json, sys
d = sys.argv[1]
last = [l for l in open(f"{d}/latency.log") if l.startswith("{")][-1]
r = json.loads(last)
json.dump({"ms_per_decision": 1000 * r["latency_median_s"], "rows": r["rows_scored"], "workers": 1}, open(f"{d}/latency.json", "w"))
PYEOF
  echo "$(date -Is) scored $arm: $(cat "$d/latency.json")"
}

case $which in
  e16)
    for spec in "e16_prompted|qwen3-8b-lora" "e16_student_s0|planner-sft-s0" "e16_student_s1|planner-sft-s1" "e16_student_s2|planner-sft-s2"; do
      name=${spec%%|*}; m=${spec#*|}; dir=$OUT/eval/$name; mkdir -p "$dir"
      for off in 0 80; do
        c=$(printf "c%03d" $((off / 80))); [[ -f $dir/$c/summary.json ]] && continue
        rm -rf "${dir:?}/$c"
        $PY scripts/run_llm_agent.py --system planner --mode compose --alerts val --val-part eval --data hi_medium --n 80 --offset $off \
            --llm endpoint --toolbox harness --model "$m" --serving-stack vllm --machine t3-5070ti --seed 0 $PROFILE \
            --library "$SHAPE" --run-id "$c" --run-dir "$dir/$c" > "$dir/$c.log" 2>&1
        echo "$(date -Is) $name $c: exit $?"
      done
    done ;;
  fast)
    score_arm untuned_4b qwen3-4b-lora
    for s in 0 1 2; do score_arm tuned_s$s fast-s$s; done ;;
  fast3)
    score_arm untuned_4b qwen3-4b-fast outputs/s1v1
    for s in 0 1 2; do score_arm tuned_s$s fast3-s$s outputs/s1v1; done ;;
  large3)
    mkdir -p outputs/s1v1/large
    $PY scripts/score_fast_baseline.py score --tier large-logprob --model "$MODEL" --workers 8 --out outputs/s1v1/large/large_lp.jsonl > outputs/s1v1/large/score.log 2>&1
    echo "$(date -Is) scored large tier: $(tail -1 outputs/s1v1/large/score.log)" ;;
  small) score_arm untuned_1.7b qwen3-1.7b ;;
esac
echo "$(date -Is) done: $which"
