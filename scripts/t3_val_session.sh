#!/usr/bin/env bash
# One Gemma 4 12B server session for the T3 validation work (G4): serve once, then, resumably and in this order,
#   smoke   (1) every system on 6 TRAIN alerts that overflowed the prompt cap in the teacher runs (outputs/t3_teacher/prompt_overflow_ids.txt):
#           no alert may end on PromptTooLong; (2) the four E2 eps arms (0.5, 1, 4, inf) of our planner on 10 val `eval` alerts
#   fitcal  our planner (compose, no library yet) on val `fit` and `cal`: the library's training runs (never `eval`)
#   library build and freeze (shape + account keying) from the teacher runs (train) + those fit/cal runs  -> outputs/t3_library/frozen
#   dspy    MIPROv2 on val `fit` (scripts/optimize_dspy.py)                                              -> outputs/t3_dspy/react_miprov2.json
#   e10large  latency per decision of the large tier (Gemma JSON decisions, one request at a time, 300 rows) -> outputs/e10/large/latency.json
#   eval    every system on the 160 val `eval` askable alerts (seed 0, harness, realistic back-test table, thinking off)
# Outputs: outputs/t3_val/fitcal/{fit,cal}/c###, outputs/t3_val/eval/<spec>/c### (a chunk is done when its summary.json exists).
# The table: scripts/val_runs_table.py -> docs/t3_val_runs.md.
#
#   scripts/t3_val_session.sh [stage ...]       # stages: smoke fitcal library dspy eval e10large (default: all, in order); SPECS="react lats" limits eval
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/projects/vllm-venv/bin:$PATH FTM_HF_HOME=${FTM_HF_HOME:-$HOME/hf_cache} HF_HUB_OFFLINE=1
set -a; source .env; set +a
export FTM_LLM_ENDPOINT=http://127.0.0.1:8000/v1
PY=${PY:-.venv/bin/python}
MODEL=gemma4-12b
OUT=${T3_VAL_OUT:-outputs/t3_val}
BT=data/backtest/hi_medium/7df6bad5337d/lr_table.jsonl
LIBDIR=outputs/t3_library/frozen
SHAPE=$LIBDIR/shape_gemma.json; ACCOUNT=$LIBDIR/account_gemma.json
DSPY=outputs/t3_dspy/react_miprov2.json
stages=${*:-smoke fitcal library dspy eval e10large}
mkdir -p "$OUT" "$LIBDIR"

serve_up() {
  setsid $PY scripts/serve_llm.py --model "$MODEL" > "$OUT/serve.log" 2>&1 &
  spid=$!; t0=$(date +%s)
  until curl -sf -H "Authorization: Bearer $FTM_LLM_API_KEY" "$FTM_LLM_ENDPOINT/models" | grep -q "\"$MODEL\""; do
    kill -0 $spid 2>/dev/null || { echo "$(date -Is) server died (see $OUT/serve.log)"; exit 1; }
    (( $(date +%s) - t0 > 1200 )) && { echo "$(date -Is) server did not load"; kill -- -$spid; exit 1; }
    sleep 5
  done
  echo "$(date -Is) $MODEL loaded in $(( $(date +%s) - t0 ))s"
}
serve_down() { kill -- -$spid 2>/dev/null; wait $spid 2>/dev/null; echo "$(date -Is) server stopped"; }
trap serve_down EXIT

# run_chunks <dir> <val-part> <total> <chunk> <run_llm_agent args ...>
run_chunks() {
  local dir=$1 part=$2 total=$3 chunk=$4; shift 4
  mkdir -p "$dir"; local i=0 off c
  for (( off=0; off<total; off+=chunk )); do
    c=$(printf "c%03d" $i); i=$((i + 1))
    [[ -f $dir/$c/summary.json ]] && continue
    rm -rf "${dir:?}/$c"
    $PY scripts/run_llm_agent.py --alerts val --val-part "$part" --data hi_medium --n "$chunk" --offset "$off" \
        --llm endpoint --toolbox harness --model "$MODEL" --serving-stack vllm --machine t3-5070ti --seed 0 \
        --backtest "$BT" --run-id "$c" --run-dir "$dir/$c" "$@" > "$dir/$c.log" 2>&1
    echo "$(date -Is) $dir/$c (offset $off): exit $?"
  done
}

serve_up
for st in $stages; do
  case $st in
  smoke)
    mkdir -p "$OUT/smoke"; head -60 outputs/t3_teacher/prompt_overflow_ids.txt > "$OUT/overflow_ids.txt"
    for sys in planner react reflexion rewoo lats llmcompiler dspy_react debate uot bed_llm sharma_native sharma_ported torres_native \
               torres_ported co_investigator_native co_investigator_ported finfre_rag_native finfre_rag_ported; do
      d=$OUT/smoke/overflow_$sys; [[ -f $d/summary.json ]] && continue
      $PY scripts/run_llm_agent.py --system "$sys" --alerts train --data hi_medium --n 6 --alert-ids "$OUT/overflow_ids.txt" \
          --llm endpoint --toolbox harness --model "$MODEL" --serving-stack vllm --machine t3-5070ti --seed 0 --backtest "$BT" \
          --run-id "overflow_$sys" --run-dir "$d" > "$d.log" 2>&1
      echo "$(date -Is) smoke overflow $sys: exit $? ($(grep -c '^PromptTooLong' "$d.log") PromptTooLong lines, $(grep -c '^PromptRefit' "$d.log") refits)"
    done
    for arm in "eps05|--eps-per-alert 0.5" "eps1|--eps-per-alert 1" "eps4|--eps-per-alert 4" "epsinf|--no-noise"; do
      d=$OUT/smoke/${arm%%|*}; [[ -f $d/summary.json ]] && continue
      # shellcheck disable=SC2086
      $PY scripts/run_llm_agent.py --system planner --mode compose --alerts val --val-part eval --data hi_medium --n 10 \
          --llm endpoint --toolbox harness --model "$MODEL" --serving-stack vllm --machine t3-5070ti --seed 0 --backtest "$BT" \
          --run-id "${arm%%|*}" --run-dir "$d" ${arm#*|} > "$d.log" 2>&1
      echo "$(date -Is) smoke ${arm%%|*}: exit $?"
    done ;;
  fitcal)  # askable alerts: fit 191, cal 109 (docs/t3_e18.md); 100 per chunk
    run_chunks "$OUT/fitcal/fit" fit 191 100 --system planner --mode compose
    run_chunks "$OUT/fitcal/cal" cal 109 109 --system planner --mode compose ;;
  library)
    runs=$(ls -d outputs/t3_teacher/runs/c0?? "$OUT"/fitcal/fit/c0?? "$OUT"/fitcal/cal/c0?? | tr '\n' ' ')
    for k in shape account; do
      f=$LIBDIR/${k}_gemma.json
      [[ -f $f ]] || $PY scripts/build_library.py --runs $runs --data hi_medium --backtest "$BT" --keying $k --out "$f" --freeze \
          > "$OUT/library_$k.json" 2> "$OUT/library_$k.log"
      echo "$(date -Is) library $k: $(cat "$OUT/library_$k.json" | tr -d '\n ')"
    done ;;
  dspy)
    [[ -f $DSPY ]] || $PY scripts/optimize_dspy.py --data hi_medium --n "${DSPY_N:-40}" --model "$MODEL" > "$OUT/dspy.log" 2>&1
    if [[ -f $DSPY ]]; then echo "$(date -Is) dspy: program $(sha256sum "$DSPY" | cut -c1-16)"
    else echo "$(date -Is) dspy: FAILED, no program saved (see $OUT/dspy.log)"; fi ;;
  eval)
    # name|args   (name = output dir; every system gets the frozen shape library, as the final run's common_args do)
    specs=(
      "ours_compose|--system planner --mode compose"
      "ours_fixed_list|--system planner --mode fixed_list"
      "ours_act_only|--system planner --mode act_only"
      "ours_no_backtest|--system planner --mode no_backtest"
      "ours_llm_decides|--system planner --mode llm_decides"
      "ours_batched_2_1|--system planner --mode compose --batch-schedule 2,1"
      "ours_library_account|--system planner --mode compose --library $ACCOUNT"
      "react|--system react" "rewoo|--system rewoo" "llmcompiler|--system llmcompiler" "reflexion|--system reflexion"
      "uot|--system uot" "bed_llm|--system bed_llm" "dspy_react|--system dspy_react --dspy-program $DSPY"
      "debate|--system debate" "lats|--system lats"
      "torres_native|--system torres_native" "torres_ported|--system torres_ported"
      "sharma_native|--system sharma_native" "sharma_ported|--system sharma_ported"
      "finfre_rag_native|--system finfre_rag_native" "finfre_rag_ported|--system finfre_rag_ported"
      "co_investigator_native|--system co_investigator_native" "co_investigator_ported|--system co_investigator_ported"
    )
    for s in "${specs[@]}"; do
      name=${s%%|*}; args=${s#*|}
      [[ -n ${SPECS:-} && " $SPECS " != *" $name "* ]] && continue
      [[ $name == dspy_react && ! -f $DSPY ]] && { echo "$(date -Is) skip dspy_react: no program at $DSPY"; continue; }
      # shellcheck disable=SC2086
      run_chunks "$OUT/eval/$name" eval 160 80 --library "$SHAPE" $args
    done ;;
  e10large)
    d=outputs/e10/large; mkdir -p "$d"
    $PY scripts/score_fast_baseline.py score --tier large --model "$MODEL" --workers 1 --limit 300 --out "$d/latency_rows.jsonl" > "$d/latency.log" 2>&1
    $PY - "$d" <<'PYEOF'
import json, sys
d = sys.argv[1]
r = json.loads([l for l in open(f"{d}/latency.log") if l.startswith("{")][-1])
json.dump({"ms_per_decision": 1000 * r["latency_median_s"], "rows": r["rows_scored"], "workers": 1}, open(f"{d}/latency.json", "w"))
PYEOF
    echo "$(date -Is) e10 large latency: $(cat $d/latency.json)" ;;
  esac
done
echo "$(date -Is) session done: $stages"
