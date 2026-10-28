#!/usr/bin/env bash
# Qwen servers for T3's fine-tuned models, one server at a time (the 16 GB card): E16 val runs and the E10 fast-tier scoring.
#   scripts/t3_qwen_session.sh e16        qwen3-8b-lora: prompted base + the 3 QLoRA students on the 160 val `eval` alerts -> outputs/t3_val/eval/e16_*
#   scripts/t3_qwen_session.sh fast       qwen3-4b-lora: untuned 4B + tuned seeds 0-2 score every val decision row   -> outputs/e10/<arm>/fast.jsonl
#   scripts/t3_qwen_session.sh small      qwen3-1.7b: untuned small general LLM, same rows                          -> outputs/e10/untuned_1.7b/fast.jsonl
# Latency per decision (one request at a time, 300 rows) is measured for every scored arm: outputs/e10/<arm>/latency.json.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/projects/vllm-venv/bin:$PATH FTM_HF_HOME=${FTM_HF_HOME:-$HOME/hf_cache} HF_HUB_OFFLINE=1
set -a; source .env; set +a
export FTM_LLM_ENDPOINT=http://127.0.0.1:8000/v1
PY=${PY:-.venv/bin/python}
OUT=${T3_VAL_OUT:-outputs/t3_val}
BT=data/backtest/hi_medium/7df6bad5337d/lr_table.jsonl
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
mkdir -p "$OUT" outputs/e10
serve_up

score_arm() {  # score_arm <arm dir name> <served model name>
  local arm=$1 m=$2 d=outputs/e10/$1
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
            --llm endpoint --toolbox harness --model "$m" --serving-stack vllm --machine t3-5070ti --seed 0 --backtest "$BT" \
            --library "$SHAPE" --run-id "$c" --run-dir "$dir/$c" > "$dir/$c.log" 2>&1
        echo "$(date -Is) $name $c: exit $?"
      done
    done ;;
  fast)
    score_arm untuned_4b qwen3-4b-lora
    for s in 0 1 2; do score_arm tuned_s$s fast-s$s; done ;;
  small) score_arm untuned_1.7b qwen3-1.7b ;;
esac
echo "$(date -Is) done: $which"
