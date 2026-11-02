#!/usr/bin/env bash
# E18 backbone refresh (2 Nov, docs/t3_e18.md "E18 refresh (2 Nov)"; a labelled deviation from prereg/e18_decision.yaml, T4 decides).
# Per candidate, strictly one at a time: download (FTM_HF_HOME, pinned revision) -> serve -> run the NEW pipeline (LangGraph planner, routed
# menu, ConditionalFusion headroom_v1, p0 v2, edge scorer v2) on val `fit` (191) and `cal` (109) in resumable chunks -> stop the server ->
# delete the weights unless this candidate is the current lowest-Brier one (weights that were already in the cache are never deleted).
# Validation fit+cal only; the `eval` part and test are never touched. The non-LLM EIG reference (CPU) is computed first.
# Finally: scripts/t3_e18_refresh_report.py report -> outputs/e18_refresh/report.{json,md} (the selection rule is fixed in the doc).
#
#   scripts/t3_e18_refresh.sh                                   # the six default candidates, in order
#   CANDIDATES="qwen35-9b gemma4-12b" scripts/t3_e18_refresh.sh  # a subset / order; CANDIDATES="... phi4-14b" adds the optional Phi-4
#   DRY_RUN=1 scripts/t3_e18_refresh.sh                         # print every command, run nothing
# The pipeline is `--profile tuned` (configs/llm/tuned.yaml: p0 v2, C11 view scorer_pcst + edge scorer v2, headroom_v1 table + ConditionalLR,
# threshold and cannot_cross of configs/decision/headroom_v1.yaml, concurrency 4); nothing is hand-passed. --timing-sample 8 only widens the latency sample.
# Env: EVIDENCE_VIEW (default: the profile's),
#      FTM_HF_HOME (default ~/hf_cache), OUT (default outputs/e18_refresh), CHUNK (default 100), KEEP_ALL=1 (delete nothing).
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/projects/vllm-venv/bin:$PATH FTM_HF_HOME=${FTM_HF_HOME:-$HOME/hf_cache}
DRY=${DRY_RUN:-0}
PY=${PY:-.venv/bin/python}
OUT=${OUT:-outputs/e18_refresh}
CHUNK=${CHUNK:-100}
VIEW_ARG=${EVIDENCE_VIEW:+--evidence-view $EVIDENCE_VIEW}  # empty = the profile's view
CANDIDATES=${CANDIDATES:-"gemma4-12b qwen35-9b qwen35-4b gemma4-e4b qwen3-14b gpt-oss-20b"}
SEEDING="--seed 0 --machine t3-5070ti"
# fit 191 + cal 109 askable alerts (docs/t3_e18.md); cal is one chunk
PARTS="fit:191:$CHUNK cal:109:109"
exec 3>&1  # DRY lines go to the original stdout even when the command is redirected
run() { if [[ $DRY == 1 ]]; then echo "DRY: $*" >&3; else "$@"; fi; }
if [[ $DRY != 1 ]]; then set -a; source .env; set +a; fi
export FTM_LLM_ENDPOINT=http://127.0.0.1:8000/v1
mkdir -p "$OUT"

# per-candidate run flags: thinking is off for all (the default); gpt-oss has no switch -> low effort and the 1,024-token output cap (old E18 finding)
extra_flags() { case $1 in gpt-oss-20b) echo "--reasoning-effort low --max-output-tokens 1024" ;; *) echo "" ;; esac; }
dl_excludes() { case $1 in gpt-oss-20b) echo "--exclude original/* --exclude metal/*" ;; *) echo "" ;; esac; }

# the non-LLM EIG reference on the same alerts (CPU, frozen headroom_v1 'primary', cross-part)
[[ -f $OUT/eig_ref.json ]] || run nice -n 10 $PY scripts/t3_e18_refresh_report.py eig-ref --out "$OUT/eig_ref.json"

spid=""
stop_server() { [[ -n $spid ]] && { kill -- -"$spid" 2>/dev/null; wait "$spid" 2>/dev/null; sleep 10; spid=""; }; }
trap stop_server EXIT

for name in $CANDIDATES; do
  if [[ $DRY == 1 ]]; then IFS=$'\t' read -r repo rev cachedir < <($PY scripts/t3_e18_refresh_report.py profile "$name"); else
    IFS=$'\t' read -r repo rev cachedir < <($PY scripts/t3_e18_refresh_report.py profile "$name") || { echo "$(date -Is) $name: no profile"; continue; }; fi
  d=$OUT/$name; mkdir -p "$d"
  if ls "$d"/cal/c000/summary.json >/dev/null 2>&1 && ls "$d"/fit/c001/summary.json >/dev/null 2>&1; then echo "$(date -Is) $name: already done"; continue; fi
  hub=$FTM_HF_HOME/hub/$cachedir
  preexisting=0; [[ -d $hub ]] && preexisting=1
  echo "$(date -Is) $name: $repo @ $rev (cache pre-existing: $preexisting)"
  # 1. download once, pinned revision (not offline for this step)
  # shellcheck disable=SC2046
  run env -u HF_HUB_OFFLINE hf download "$repo" --revision "$rev" $(dl_excludes "$name") > "$d/download.log" 2>&1 \
    || { echo "$(date -Is) $name: download failed (see $d/download.log), skipped"; echo "{\"candidate\": \"$name\", \"loaded\": false, \"stage\": \"download\"}" > "$d/load.json"; continue; }
  # 2. serve at the prompt cap (profile max_model_len 8192) and wait for the endpoint, recording load time and peak VRAM
  export HF_HUB_OFFLINE=1
  if [[ $DRY == 1 ]]; then echo "DRY: setsid $PY scripts/serve_llm.py --model $name > $d/serve.log; wait for /models to list $name (timeout 1200 s)"; else
    base=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); peak=$base; t0=$(date +%s); ok=0
    setsid $PY scripts/serve_llm.py --model "$name" > "$d/serve.log" 2>&1 &
    spid=$!
    while (( $(date +%s) - t0 < 1200 )); do
      kill -0 $spid 2>/dev/null || break
      curl -sf -H "Authorization: Bearer $FTM_LLM_API_KEY" "$FTM_LLM_ENDPOINT/models" | grep -q "\"$name\"" && { ok=1; break; }
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); (( u > peak )) && peak=$u
      sleep 5
    done
    load_s=$(( $(date +%s) - t0 ))
    if (( ok == 0 )); then
      echo "{\"candidate\": \"$name\", \"loaded\": false, \"stage\": \"load\", \"load_s\": $load_s, \"peak_mib\": $peak}" > "$d/load.json"
      echo "$(date -Is) $name: did not load in ${load_s}s (see $d/serve.log), skipped"
      stop_server
      [[ $preexisting == 0 && ${KEEP_ALL:-0} != 1 ]] && rm -rf "$hub"
      continue
    fi
    echo "$(date -Is) $name: loaded in ${load_s}s"
  fi
  # 3. val fit + cal, resumable chunks: a chunk is done when its summary.json exists
  for spec in $PARTS; do
    IFS=: read -r part total chunk <<< "$spec"
    i=0
    for (( off=0; off<total; off+=chunk )); do
      c=$(printf "c%03d" $i); i=$((i + 1))
      [[ -f $d/$part/$c/summary.json ]] && continue
      [[ $DRY == 1 ]] || rm -rf "${d:?}/$part/$c"
      mkdir -p "$d/$part"
      # shellcheck disable=SC2046
      run $PY scripts/run_llm_agent.py --system planner --mode compose --alerts val --val-part "$part" --data hi_medium \
          --n "$chunk" --offset "$off" --llm endpoint --toolbox harness --model "$name" --serving-stack vllm --candidate "$name" $SEEDING \
          --profile tuned $VIEW_ARG --timing-sample 8 $(extra_flags "$name") \
          --run-id "$c" --run-dir "$d/$part/$c" > "$d/$part/$c.log" 2>&1
      echo "$(date -Is) $name $part/$c (offset $off): exit $?"
    done
  done
  [[ $DRY == 1 ]] && { echo "DRY: stop server; report --best; delete $hub unless best or pre-existing"; continue; }
  [[ -n ${load_s:-} ]] && echo "{\"candidate\": \"$name\", \"loaded\": true, \"load_s\": $load_s, \"peak_mib\": $peak, \"baseline_mib\": $base, \"repo\": \"$repo\", \"revision\": \"$rev\"}" > "$d/load.json"
  # 4. stop the server, then delete the weights unless this is the current best (or was in the cache before we started)
  stop_server
  best=$($PY scripts/t3_e18_refresh_report.py report --root "$OUT" --best 2>/dev/null)
  echo "$(date -Is) $name done; current best by fit+cal Brier: ${best:-none}"
  if [[ ${KEEP_ALL:-0} != 1 ]]; then
    if [[ $preexisting == 0 && $best != "$name" ]]; then rm -rf "$hub"; echo "$(date -Is) $name: weights deleted"; fi
    # a previously kept best that this candidate has just beaten
    if [[ -n $best && $best == "$name" && -f $OUT/.kept ]]; then
      prev=$(cat "$OUT/.kept"); if [[ $prev != "$name" && -f $OUT/.kept_$prev ]]; then rm -rf "$(cat "$OUT/.kept_$prev")"; rm -f "$OUT/.kept_$prev"; fi
    fi
    [[ $preexisting == 0 && $best == "$name" ]] && { echo "$name" > "$OUT/.kept"; echo "$hub" > "$OUT/.kept_$name"; }
  fi
done

run $PY scripts/t3_e18_refresh_report.py report --root "$OUT"
