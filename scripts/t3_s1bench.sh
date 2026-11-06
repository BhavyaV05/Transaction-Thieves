#!/usr/bin/env bash
# S1BENCH GPU steps (docs/t3_s1bench.md section 9).  One server at a time on the 16 GB card, like scripts/t3_qwen_session.sh.
#   scripts/t3_s1bench.sh fast   qwen3-4b-fast     System 1 (adapter ${S1_ADAPTER:-fast3-s0}) and the untuned base on Part A (public benchmarks + S1MB)
#   scripts/t3_s1bench.sh large  gemma4-12b        the Gemma 4 12B large tier on Part A
#   scripts/t3_s1bench.sh partb-gpu                JevK5 and Tev1-4B on our decisions through their own interfaces (no server: transformers, bf16)
# Part B scores of System 1 / base / Gemma come from the existing wave-4 scoring (scripts/t3_qwen_session.sh fast3 / large3): outputs/s1v1/*/fast.jsonl.
# Prerequisites (CPU, done): scripts/s1bench_prepare.py download|external|s1mb|published  ->  /home/akshat/projects/s1bench_data/prepared
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/projects/vllm-venv/bin:$PATH FTM_HF_HOME=${FTM_HF_HOME:-$HOME/hf_cache} HF_HUB_OFFLINE=1
set -a; source .env; set +a
export FTM_LLM_ENDPOINT=http://127.0.0.1:8000/v1
PY=${PY:-.venv/bin/python}
OUT=${S1B_OUT:-outputs/s1bench}
which=$1; MODEL=${2:-}
ADAPTER=${S1_ADAPTER:-fast3-s0}
mkdir -p "$OUT" outputs/t3_val

serve_up() {
  setsid $PY scripts/serve_llm.py --model "$MODEL" > "$OUT/serve_$MODEL.log" 2>&1 &
  spid=$!; t0=$(date +%s)
  until curl -sf -H "Authorization: Bearer ${FTM_LLM_API_KEY:-none}" "$FTM_LLM_ENDPOINT/models" | grep -q "\"$MODEL\""; do
    kill -0 $spid 2>/dev/null || { echo "$(date -Is) server died (see $OUT/serve_$MODEL.log)"; exit 1; }
    (( $(date +%s) - t0 > 1500 )) && { echo "server did not load"; kill -- -$spid; exit 1; }
    sleep 5
  done
  echo "$(date -Is) $MODEL loaded in $(( $(date +%s) - t0 ))s"
}

part_a() {  # part_a <tag> <served model name> [extra args for ext]
  local tag=$1 m=$2; shift 2
  $PY scripts/s1bench_run.py ext   --tag "$tag" --model "$m" --tasks all --perm --workers 32 --resume --out "$OUT" "$@" > "$OUT/$tag.ext.log" 2>&1
  $PY scripts/s1bench_run.py s1mb  --tag "$tag" --model "$m" --benchmarks all --perm --workers 32 --resume --out "$OUT" > "$OUT/$tag.s1mb.log" 2>&1
  # latency: one request at a time, 200 eval items per task (all passes of a two-stage decision)
  $PY scripts/s1bench_run.py latency --tag "$tag" --model "$m" --tasks all --latency-n 200 --out "$OUT" > "$OUT/$tag.lat.log" 2>&1
  echo "$(date -Is) part A done: $tag"
}

case $which in
  fast)    # MODEL=qwen3-4b-fast
    trap 'kill -- -$spid 2>/dev/null; wait $spid 2>/dev/null' EXIT
    serve_up
    part_a system1 "$ADAPTER"                                    # production two-stage (typed.py)
    $PY scripts/s1bench_run.py ext --tag system1_listed --model "$ADAPTER" --variant listed --tasks banking77 clinc150_oos hwu64 --perm --workers 32 --resume --out "$OUT" \
        > "$OUT/system1_listed.ext.log" 2>&1                      # benchmark-only stage 1 that lists the labels of each group
    part_a qwen3-4b qwen3-4b-fast                                # the untuned base (reference); the served base name is the profile name
    ;;
  large)   # MODEL=gemma4-12b
    trap 'kill -- -$spid 2>/dev/null; wait $spid 2>/dev/null' EXIT
    serve_up
    part_a gemma4-12b "$MODEL" --workers 8
    ;;
  partb-gpu)  # no server: the decoder-based open models run through transformers (bf16) on the free card
    for m in jevk5 tev1-4b; do
      $PY scripts/s1bench_partb.py open --model $m --out "$OUT/partb/$m/scores.jsonl" > "$OUT/partb/$m.log" 2>&1 || echo "$m failed: see $OUT/partb/$m.log"
    done ;;
esac
echo "$(date -Is) done: $which"
