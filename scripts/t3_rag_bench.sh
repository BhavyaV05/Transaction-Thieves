#!/usr/bin/env bash
# GPU matrix of the five RAG components with WITH vs WITHOUT arms (docs/t3_rag_bench.md; arm list: scripts/t3_rag_bench_report.py).
# One Gemma 4 12B server session, strictly the tuned production profile (`--profile tuned`: p0 v2, scorer_pcst + edge scorer v2, headroom_v1
# ConditionalLR, threshold + cannot_cross stop, concurrency 4) plus the frozen shape library, and ONE factor changed per arm. Why one factor at a
# time and not the cross product: the factors act on different parts of the prompt, a full product is 4 x 4 x 4 x 3 x ... arms of 0.2 GPU-h
# each (about 100 h) and a 300-alert paired comparison has no power for 3-way interactions; interactions are checked instead by ONE final
# "all chosen winners combined" arm against the default (stage `combined`).
#
# Validation `fit` (191) + `cal` (109) askable alerts, resumable chunks (a chunk is done when its summary.json exists). The eval part is only
# touched by stage `eval`, once, for the frozen winners (guarded by outputs/rag_bench_eval/.done). Test is never touched.
#
#   scripts/t3_rag_bench.sh plan                   # print the arm table with GPU-hour estimates and blocked arms (no GPU)
#   scripts/t3_rag_bench.sh arms                   # every ready arm on fit + cal (ARMS="a b" limits; DRY_RUN=1 prints the commands)
#   scripts/t3_rag_bench.sh report                 # CPU: metrics, paired cluster-bootstrap CIs, Holm, selection -> outputs/rag_bench/selection.json
#   scripts/t3_rag_bench.sh combined               # the all-winners arm (needs selection.json), 2 seeds, then report again
#   scripts/t3_rag_bench.sh eval                   # ONCE: default + chosen winners (+ combined) on the 160 val eval alerts
# Libraries (fixed 7 Nov, before any GPU run): fit and cal arms use the TEACHER-ONLY libraries outputs/t3_library/frozen_v3/{shape,account}_gemma_train.json, eval uses
# frozen_v3/{shape,account}_gemma.json (same rule as scripts/t3_rerun.sh); the choice is made by the report script (t3_rag_bench_report.py lib_path), never here.
# The bench no longer rebuilds libraries: frozen_v3 is built from the teacher_v2 runs (1,983 train alerts, the same tuned stack), so the old `library` stage would only repeat them.
# Env: MODEL (gemma4-12b), OUT (outputs/rag_bench), CHUNK (100), ARMS, DRY_RUN=1, PY.
set -uo pipefail
cd "$(dirname "$0")/.."
DRY=${DRY_RUN:-0}
PY=${PY:-.venv/bin/python}
MODEL=${MODEL:-gemma4-12b}
OUT=${OUT:-outputs/rag_bench}
CHUNK=${CHUNK:-100}
REPORT="$PY scripts/t3_rag_bench_report.py"
export PYTHONPATH=${PYTHONPATH:-.}
exec 3>&1
run() { if [[ $DRY == 1 ]]; then echo "DRY: $*" >&3; else "$@"; fi; }
run_log() { local log=$1; shift; if [[ $DRY == 1 ]]; then echo "DRY: $* > $log" >&3; return 0; else "$@" > "$log" 2>&1; fi; }
stage=${1:-plan}

serve_up() {
  [[ $DRY == 1 ]] && { echo "DRY: serve $MODEL" >&3; return; }
  export PATH=$HOME/projects/vllm-venv/bin:$PATH FTM_HF_HOME=${FTM_HF_HOME:-$HOME/hf_cache} HF_HUB_OFFLINE=1
  set -a; source .env; set +a
  export FTM_LLM_ENDPOINT=http://127.0.0.1:8000/v1
  mkdir -p "$OUT"
  setsid $PY scripts/serve_llm.py --model "$MODEL" > "$OUT/serve.log" 2>&1 &
  spid=$!; t0=$(date +%s)
  until curl -sf -H "Authorization: Bearer $FTM_LLM_API_KEY" "$FTM_LLM_ENDPOINT/models" | grep -q "\"$MODEL\""; do
    kill -0 $spid 2>/dev/null || { echo "$(date -Is) server died (see $OUT/serve.log)"; exit 1; }
    (( $(date +%s) - t0 > 1200 )) && { echo "$(date -Is) server did not load"; kill -- -$spid; exit 1; }
    sleep 5
  done
  echo "$(date -Is) $MODEL loaded in $(( $(date +%s) - t0 ))s"
  trap 'kill -- -$spid 2>/dev/null; wait $spid 2>/dev/null' EXIT
}

# run_arm <root> <name> <system> <seed> <library-path|none> <part:total:chunk ...> -- <extra flags>
run_arm() {
  local root=$1 name=$2 sys=$3 seed=$4 lib=$5; shift 5
  local specs=() ; while [[ $1 != -- ]]; do specs+=("$1"); shift; done; shift
  local libarg=(); [[ $lib != none ]] && libarg=(--library "$lib")
  local spec part total chunk off i c d
  for spec in "${specs[@]}"; do
    IFS=: read -r part total chunk <<< "$spec"
    d=$root/$name/$part; i=0; [[ $DRY == 1 ]] || mkdir -p "$d"
    for (( off=0; off<total; off+=chunk )); do
      c=$(printf "c%03d" $i); i=$((i + 1))
      [[ -f $d/$c/summary.json ]] && continue
      [[ $DRY == 1 ]] || rm -rf "${d:?}/$c"
      run_log "$d/$c.log" $PY scripts/run_llm_agent.py --system "$sys" --mode compose --alerts val --val-part "$part" --data hi_medium --n "$chunk" --offset "$off" \
          --llm endpoint --toolbox harness --model "$MODEL" --serving-stack vllm --candidate "$name" --seed "$seed" --machine t3-5070ti \
          --profile tuned "${libarg[@]}" "$@" --run-id "$c" --run-dir "$d/$c"
      echo "$(date -Is) $name $part/$c (offset $off): exit $?"
    done
  done
}

case $stage in
plan)
  $REPORT plan ;;
arms)
  serve_up
  while IFS=$'\t' read -r name sys seed lib lib_eval est status flags; do
    [[ -n ${ARMS:-} && " $ARMS " != *" $name "* ]] && continue
    if [[ $status != ok ]]; then echo "$(date -Is) $name: BLOCKED ($status)"; continue; fi
    # shellcheck disable=SC2086
    run_arm "$OUT" "$name" "$sys" "$seed" "$lib" "fit:191:$CHUNK" "cal:109:109" -- $flags
  done < <($REPORT arms --tsv)
  run $REPORT report --root "$OUT" ;;
report)
  run $REPORT report --root "$OUT" ;;
combined)  # all chosen winners in one arm, seeds 0 and 1: the interaction check against the default
  [[ -f $OUT/selection.json ]] || { echo "no $OUT/selection.json: run the arms and the report first"; exit 1; }
  read -r lib flags < <($REPORT winner-flags --selection "$OUT/selection.json")
  libpath=none; [[ $lib != none ]] && libpath=$($PY -c "import sys; sys.path.insert(0,'scripts'); import t3_rag_bench_report as r; print(r.lib_path('$lib', 'fitcal'))")
  echo "combined: library=$lib ($libpath) flags=[$flags]"
  serve_up
  for seed in 0 1; do
    # shellcheck disable=SC2086
    run_arm "$OUT" "planner_combined$([[ $seed == 1 ]] && echo _seed1)" planner "$seed" "$libpath" "fit:191:$CHUNK" "cal:109:109" -- $flags
  done
  run $REPORT report --root "$OUT" ;;
eval)  # the once-only eval: refuse a second time
  EV=${EVAL_OUT:-outputs/rag_bench_eval}
  [[ -f $EV/.done ]] && { echo "eval already scored once ($EV/.done); it is never repeated"; exit 1; }
  [[ -f $OUT/selection.json ]] || { echo "no selection.json"; exit 1; }
  names=$($PY - <<'E'
import json, sys
sys.path.insert(0, "scripts")
import t3_rag_bench_report as r
sel = json.load(open("outputs/rag_bench/selection.json"))["selection"]
out = {"planner_default"} | {s["winner"] for s in sel.values()} | {s["default"] for s in sel.values()}
print(" ".join(sorted(out)))
E
)
  serve_up
  while IFS=$'\t' read -r name sys seed lib_fitcal lib est status flags; do
    [[ " $names " != *" $name "* ]] && continue
    [[ $status != ok ]] && continue
    # shellcheck disable=SC2086
    run_arm "$EV" "$name" "$sys" "$seed" "$lib" "eval:160:160" -- $flags
  done < <($REPORT arms --tsv)
  if [[ -d $OUT/planner_combined ]]; then  # the combined arm of stage `combined`, once, with the same flags
    read -r lib flags < <($REPORT winner-flags --selection "$OUT/selection.json")
    libpath=none; [[ $lib != none ]] && libpath=$($PY -c "import sys; sys.path.insert(0,'scripts'); import t3_rag_bench_report as r; print(r.lib_path('$lib', 'eval'))")
    # shellcheck disable=SC2086
    run_arm "$EV" planner_combined planner 0 "$libpath" "eval:160:160" -- $flags
  fi
  [[ $DRY == 1 ]] || touch "$EV/.done"
  run $REPORT report --root "$EV" --eval ;;
*) echo "usage: $0 plan|library|arms|report|combined|eval"; exit 2 ;;
esac
