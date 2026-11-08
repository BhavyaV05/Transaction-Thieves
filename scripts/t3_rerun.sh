#!/usr/bin/env bash
# T3 rerun orchestration: every remaining GPU stage in dependency order, each with its CPU post-processing (docs/t3_rerun.md).
#
#   scripts/t3_rerun.sh <stage> [<stage> ...]     DRY_RUN=1 prints every command and runs nothing
#
# Stages (sub-stages in brackets are what the queue lines call; a bare stage runs its sub-stages in order):
#   plan        print the GPU-hour table (scripts/t3_rerun_tools.py hours) and the recommended queue lines
#   teacher     Gemma 4 12B planner, tuned profile, ALL 1,983 HI-Medium train alerts, chunks of 80 -> outputs/t3_teacher_v2/runs
#               + the library seeding runs (planner compose, no library) on val fit and cal         -> outputs/t3_val_v3/seed/{fit,cal}
#               REQUIRES the planner prompt-budget fix (PROMPTFIT) merged in the commit the queue line names: set PROMPTFIT_MERGED=1
#   library     (CPU) shape + account libraries -> outputs/t3_library/frozen_v3/: *_gemma.json from teacher_v2 + the seeding runs (used for
#               val eval and everything after), *_gemma_train.json from teacher_v2 only (used on val fit/cal, so no fit/cal label leaks)
#   distill     (CPU) train_planner.py distill, expected-value filter, teacher_v2 -> outputs/sft_v3/planner_train.jsonl, then the stop-example
#               rebalance -> planner_train_balanced.jsonl + stop_rebalance.json (counts)
#   student     [student-train] QLoRA 3 seeds (train venv) -> models/planner_sft_v3
#               [student-e16]   prompted Qwen3-8B vs the 3 students on fit, cal, eval (eval once) + reference Gemma ours_compose from `val`;
#                               report: t3_rerun_report.py e16-choose (fit, cal) and e16 (eval) = scripts/t3_e16_report.py
#   val         [val-dspy] MIPROv2 program  [val-fitcal] all systems on fit and cal  [val-calibrate] scripts/calibrate_external.py fit on fit,
#               checked on cal  [val-eval] all systems on eval (once)  [val-apply] the frozen maps applied to eval  [val-report] tables + drafts
#   e15         injection tests (planner + react) on the real endpoint, with/without the gate (+ CaMeL-style arms, + keyword screen)
#   e18         scripts/t3_e18_refresh.sh (ordering wrapper)
#   rag         scripts/t3_rag_bench.sh library arms report combined eval (ordering wrapper)
#   system1     docs/t3_system1.md section 8 steps 3-8 in order (fast scoring, CPU reports, dual-serve smoke, integration arms fit+cal,
#               selection, eval once, E15 with the fast-model screen)
#   all         teacher library distill student val e15 e18 rag system1
#
# Every LLM run is `--profile tuned --concurrency 4`, chunks of 80 alerts, val parts fit / cal / eval kept apart (eval only in the stages marked
# "once"). A chunk is done when its summary.json exists; finished chunks / outputs are skipped, so any stage can be killed and restarted.
# Test data are never touched (no --final, no test split). Env: PY, TRAIN_PY, V (val tree, default outputs/t3_val_v3), CHUNK (80), MODEL (gemma4-12b),
# FORCE_EVAL=1 (run eval before the calibration maps exist), S1_SEED (fast adapter seed, 0), E15_N_PLANNER (60), E15_N_REACT (40), DSPY (program path),
# E18_CANDIDATES, RAG_STAGES.
set -uo pipefail
cd "$(dirname "$0")/.."
DRY=${DRY_RUN:-0}
# MACHINE is recorded in every run summary (default: the PC).  EXTERNAL_SERVER=1: the endpoint is already served (e.g. on ada, scripts/ada/tasks) -> no server is started or stopped here;
# FTM_LLM_ENDPOINT and FTM_LLM_API_KEY must be exported by the caller.
MACHINE=${MACHINE:-t3-5070ti}
EXTERNAL_SERVER=${EXTERNAL_SERVER:-0}
PY=${PY:-.venv/bin/python}
[[ -x $PY ]] || PY=$HOME/projects/Transaction-Thieves/.venv/bin/python
TRAIN_PY=${TRAIN_PY:-$HOME/projects/train-venv/bin/python}
export PYTHONPATH=${PYTHONPATH:-$PWD}
TOOLS="$PY scripts/t3_rerun_tools.py"
# backbone: MODEL if set, else the E18-refresh pick (docs/t3_e18.md "E18 refresh (2 Nov)", selection fixed before the runs), else Gemma 4 12B
# (also when the refresh picked no LLM: the LLM systems still run on the incumbent so they can be compared with the EIG policy)
MODEL=${MODEL:-$($PY -c 'import json,sys; p=json.load(open("outputs/e18_refresh/report.json")).get("pick"); print(p or "gemma4-12b")' 2>/dev/null || echo gemma4-12b)}
CHUNK=${CHUNK:-80}
V=${V:-outputs/t3_val_v3}
TEACHER=${TEACHER:-outputs/t3_teacher_v2}
LIB=outputs/t3_library/frozen_v3
SFT=${SFT:-outputs/sft_v3}
STUDENTS=models/planner_sft_v3
DSPY=${DSPY:-outputs/t3_dspy/react_miprov2.json}   # 2 Nov: reuse the MIPROv2 program (sha 0697b613...; the DSPy-ReAct loop and prompts are unchanged by the rewrite, parity-tested); DSPY=outputs/t3_dspy_v3/react_miprov2.json re-optimises (2.7 h)
S1_SEED=${S1_SEED:-0}
FROZEN=outputs/s1v1/system1_frozen_s$S1_SEED.json
BT=configs/fusion/headroom_v1_table.jsonl
FUSION=configs/fusion/headroom_v1.json
THRESH=0.3227
PROFILE="--profile tuned --concurrency 4"
COMMON="--llm endpoint --toolbox harness --serving-stack vllm --machine $MACHINE --seed 0 $PROFILE"
FIT_N=191 CAL_N=109 EVAL_N=160 TRAIN_N=1983
export FTM_P0_VARIANT=${FTM_P0_VARIANT:-v2} FTM_EDGE_SCORER=${FTM_EDGE_SCORER:-v2}

exec 3>&1  # DRY lines go to the original stdout even when a command is redirected
say() { echo "$(date -Is) $*" >&3; }
run() { if [[ $DRY == 1 ]]; then echo "DRY: $*" >&3; else "$@"; fi; }
# run_log <log> cmd ... : stdout+stderr to the log (DRY: shown)
run_log() { local log=$1; shift; if [[ $DRY == 1 ]]; then echo "DRY: $* > $log" >&3; else "$@" > "$log" 2>&1; fi; }
mk() { [[ $DRY == 1 ]] || mkdir -p "$@"; }

# ---------------------------------------------------------------------------------------------------------------------------------
# servers
spid=""
env_vllm() {
  [[ $DRY == 1 || $EXTERNAL_SERVER == 1 ]] && return
  export PATH=$HOME/projects/vllm-venv/bin:$PATH FTM_HF_HOME=${FTM_HF_HOME:-$HOME/hf_cache} HF_HUB_OFFLINE=1
  set -a; source .env; set +a
  export FTM_LLM_ENDPOINT=http://127.0.0.1:8000/v1 FTM_FAST_ENDPOINT=http://127.0.0.1:8001/v1
}
serve_down() { [[ -n $spid ]] && { kill -- -"$spid" 2>/dev/null; wait "$spid" 2>/dev/null; sleep 10; say "server stopped"; spid=""; }; return 0; }
trap serve_down EXIT
# serve_up <profile> [serve_llm args...] : one server, waits until the endpoint lists the profile (or, for --dual, every profile named in WAIT_FOR)
serve_up() {
  local prof=$1; shift
  [[ $EXTERNAL_SERVER == 1 ]] && { say "external server at ${FTM_LLM_ENDPOINT:-unset} ($prof)"; return; }
  if [[ $DRY == 1 ]]; then echo "DRY: setsid $PY scripts/serve_llm.py --model $prof $* ; wait until /models lists $prof" >&3; return; fi
  mkdir -p "$V"
  setsid $PY scripts/serve_llm.py --model "$prof" "$@" > "$V/serve_$prof.log" 2>&1 &
  spid=$!; local t0=$(date +%s)
  until curl -sf -H "Authorization: Bearer $FTM_LLM_API_KEY" "$FTM_LLM_ENDPOINT/models" | grep -q "\"$prof\""; do
    kill -0 $spid 2>/dev/null || { say "$prof: server died (see $V/serve_$prof.log)"; exit 1; }
    (( $(date +%s) - t0 > 1500 )) && { say "$prof: did not load"; kill -- -$spid; exit 1; }
    sleep 5
  done
  say "$prof loaded in $(( $(date +%s) - t0 ))s"
}
serve_dual() {  # Gemma (8000) + the fast model (8001), docs/t3_system1.md section 5
  if [[ $DRY == 1 ]]; then echo "DRY: setsid $PY scripts/serve_llm.py --dual system1 ; wait for gemma4-12b-dual (8000) and qwen3-4b-fast (8001)" >&3; return; fi
  mkdir -p "$V"
  setsid $PY scripts/serve_llm.py --dual system1 > "$V/serve_dual.log" 2>&1 &
  spid=$!; local t0=$(date +%s)
  until curl -sf -H "Authorization: Bearer $FTM_LLM_API_KEY" "$FTM_LLM_ENDPOINT/models" | grep -q gemma4-12b-dual \
     && curl -sf -H "Authorization: Bearer $FTM_LLM_API_KEY" "$FTM_FAST_ENDPOINT/models" | grep -q qwen3-4b-fast; do
    kill -0 $spid 2>/dev/null || { say "dual server died (see $V/serve_dual.log)"; exit 1; }
    (( $(date +%s) - t0 > 2400 )) && { say "dual server did not load"; kill -- -$spid; exit 1; }
    sleep 5
  done
  say "dual loaded in $(( $(date +%s) - t0 ))s; $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -1)"
}

# ---------------------------------------------------------------------------------------------------------------------------------
# run_chunks <dir> <alerts> <val-part|-> <total> <model> <run_llm_agent args ...> : chunks of CHUNK alerts, finished chunks skipped
run_chunks() {
  local dir=$1 alerts=$2 part=$3 total=$4 model=$5; shift 5
  local pargs="--alerts $alerts"; [[ $part != - ]] && pargs="$pargs --val-part $part"
  mk "$dir"
  local off i c
  while read -r off i; do
    c=$(printf "c%03d" "$i")
    [[ -f $dir/$c/summary.json ]] && continue
    [[ $DRY != 1 ]] && rm -rf "${dir:?}/$c"
    # shellcheck disable=SC2086
    run_log "$dir/$c.log" $PY scripts/run_llm_agent.py $pargs --data hi_medium --n "$CHUNK" --offset "$off" $COMMON --model "$model" \
        --run-id "$c" --run-dir "$dir/$c" "$@"
    [[ $DRY == 1 ]] || say "$dir/$c (offset $off): exit $? ($(grep -c '^PromptTooLong' "$dir/$c.log") PromptTooLong, $(grep -c '^PromptRefit' "$dir/$c.log") refits)"
    # a chunk whose arm never called the model, or whose prompts were all too long, is not a result: stop (CHUNK_CHECK=0 to skip)
    if [[ $DRY != 1 && ${CHUNK_CHECK:-1} == 1 ]]; then $PY scripts/t3_chunk_check.py "$dir/$c" || { say "$dir/$c FAILED the chunk check: stopping, nothing recorded"; exit 4; }; fi
  done < <($TOOLS chunks "$total" "$CHUNK")
}
total_of() { case $1 in fit) echo $FIT_N ;; cal) echo $CAL_N ;; eval) echo $EVAL_N ;; esac; }
# the --n of a chunk is CHUNK; the last chunk of a part has fewer alerts, run_llm_agent takes what is left after --offset (n is an upper bound)

# comma list of the chunk run dirs of a path (system1_report.py NAME=dir,dir,...)
chunk_dirs() { ls -d "$1"/c[0-9][0-9][0-9] 2>/dev/null | paste -sd, -; }

# ---------------------------------------------------------------------------------------------------------------------------------
stage_teacher() {
  if [[ ${PROMPTFIT_MERGED:-0} != 1 && $DRY != 1 ]]; then
    say "teacher: refusing to start. The tuned prompt (~10k tokens) overflows the 6k cap without the PROMPTFIT planner prompt-budget fix;"
    say "teacher: the queue line must name a commit that contains it, and set PROMPTFIT_MERGED=1."; exit 2
  fi
  env_vllm; serve_up "$MODEL"
  # 1. teacher: every train alert, planner compose, prompts kept (the distillation input)
  run_chunks "$TEACHER/runs" train - $TRAIN_N "$MODEL" --system planner --mode compose --keep-prompts
  # 2. library seeding runs: planner compose on val fit and cal WITHOUT a library (labels of fit/cal enter the library only via these runs)
  run_chunks "$V/seed/fit" val fit $FIT_N "$MODEL" --system planner --mode compose
  run_chunks "$V/seed/cal" val cal $CAL_N "$MODEL" --system planner --mode compose
  serve_down
}

build_lib() {  # build_lib <keying> <out> <runs...>
  local k=$1 out=$2; shift 2
  [[ -f $out ]] && { say "library: $out exists, skipped"; return; }
  mk "$LIB"
  run_log "${out%.json}.log" $PY scripts/build_library.py --runs "$@" --data hi_medium --backtest "$BT" --keying "$k" --out "$out" --freeze
}
stage_library() {
  local tr full
  tr=$(ls -d $TEACHER/runs/c[0-9][0-9][0-9] 2>/dev/null | tr '\n' ' ')
  full="$tr $(ls -d $V/seed/fit/c[0-9][0-9][0-9] $V/seed/cal/c[0-9][0-9][0-9] 2>/dev/null | tr '\n' ' ')"
  if [[ $DRY == 1 && -z $tr ]]; then tr="$TEACHER/runs/c0*"; full="$tr $V/seed/fit/c0* $V/seed/cal/c0*"; fi
  if [[ $DRY != 1 && -z $tr ]]; then say "library: no teacher runs under $TEACHER/runs (run stage teacher first)"; exit 1; fi
  for k in shape account; do
    # shellcheck disable=SC2086
    build_lib $k "$LIB/${k}_gemma_train.json" $tr
    # shellcheck disable=SC2086
    build_lib $k "$LIB/${k}_gemma.json" $full
  done
  [[ $DRY == 1 ]] || for f in "$LIB"/*.json; do say "library $f: $(sha256sum "$f" | cut -c1-16)"; done
}

stage_distill() {
  local rev
  rev=$($TOOLS revision "$MODEL") || exit 1
  mk "$SFT"
  if [[ -f $SFT/planner_train.jsonl ]]; then say "distill: $SFT/planner_train.jsonl exists, skipped"; else
    local runs; runs=$(ls -d $TEACHER/runs/c[0-9][0-9][0-9] 2>/dev/null | tr '\n' ' '); [[ -z $runs ]] && runs="$TEACHER/runs/c0*"
    # shellcheck disable=SC2086
    run_log "$SFT/distill.log" $PY scripts/train_planner.py distill --runs $runs \
        --data hi_medium --splits train --backtest "$BT" --backtest-mode realistic --fusion-params "$FUSION" --filter expected \
        --stop-threshold $THRESH --teacher-model "$MODEL" --teacher-revision "$rev" --out "$SFT/planner_train.jsonl"
  fi
  if [[ -f $SFT/planner_train_balanced.jsonl ]]; then say "distill: balanced set exists, skipped"; else
    # stops are kept up to 25% of the examples (never upsampled); the counts before and after are in stop_rebalance.json
    run $TOOLS rebalance --in "$SFT/planner_train.jsonl" --out "$SFT/planner_train_balanced.jsonl" --report "$SFT/stop_rebalance.json" --max-stop-share 0.25
  fi
  [[ $DRY == 1 ]] || { say "distill: $(tail -c 600 "$SFT/summary.json" 2>/dev/null | tr -d '\n ')"; cat "$SFT/stop_rebalance.json" 2>/dev/null; }
}

stage_student_train() {
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  for s in 0 1 2; do
    run $TRAIN_PY scripts/train_planner.py train --config configs/llm/sft_planner_v3.yaml --dataset "$SFT/planner_train_balanced.jsonl" --seed $s --skip-existing || exit 1
  done
}
stage_student_eval() {
  # E16 rerun: prompted Qwen3-8B (AWQ base of the profile) vs the 3 students; fit, cal (choose), then eval once. The Gemma reference
  # (ours_compose) comes from stage val on the same parts. The library of fit/cal is the teacher-only one, eval's the full one.
  local prof=qwen3-8b-lora-v3 part lib arm model
  env_vllm; serve_up "$prof"
  for part in fit cal eval; do
    [[ $part == eval ]] && lib=$LIB/shape_gemma.json || lib=$LIB/shape_gemma_train.json
    if [[ $part == eval && -f $V/eval/.e16_done && ${FORCE_EVAL:-0} != 1 ]]; then say "e16: eval already run (once), skipped"; continue; fi
    for arm in "e16_prompted|$prof" "e16_student_s0|planner-sft-v3-s0" "e16_student_s1|planner-sft-v3-s1" "e16_student_s2|planner-sft-v3-s2"; do
      model=${arm#*|}
      run_chunks "$V/$part/${arm%%|*}" val $part "$(total_of $part)" "$model" --system planner --mode compose --library "$lib"
    done
    [[ $part == eval && $DRY != 1 ]] && touch "$V/eval/.e16_done"
  done
  serve_down
  for part in fit cal; do run_log "$V/analysis_e16_$part.log" $PY scripts/t3_rerun_report.py e16-choose --part $part --root "$V"; done
  run_log "$V/analysis_e16_eval.log" $PY scripts/t3_rerun_report.py e16 --root "$V"
}

# ---------------------------------------------------------------------------------------------------------------------------------
# the val systems: name|args  (@SHAPE@ / @ACCOUNT@ = the library of the part: teacher-only on fit and cal, full on eval)
val_specs() {
  cat <<EOF
ours_compose|--system planner --mode compose --library @SHAPE@
ours_llm_decides|--system planner --mode llm_decides --library @SHAPE@
ours_act_only|--system planner --mode act_only --library @SHAPE@
ours_fixed_list|--system planner --mode fixed_list --library @SHAPE@
ours_batched_2_1|--system planner --mode compose --batch-schedule 2,1 --library @SHAPE@
ours_library_account|--system planner --mode compose --library @ACCOUNT@
ours_no_backtest|--system planner --mode no_backtest --library @SHAPE@
react_adapted|--system react_adapted --variant-label react_adapted --library @SHAPE@
reflexion_adapted|--system reflexion_adapted --variant-label reflexion_adapted --library @SHAPE@
rewoo_adapted|--system rewoo_adapted --variant-label rewoo_adapted --library @SHAPE@
lats_adapted|--system lats_adapted --variant-label lats_adapted --library @SHAPE@
llmcompiler_adapted|--system llmcompiler_adapted --variant-label llmcompiler_adapted --library @SHAPE@
dspy_react_adapted|--system dspy_react_adapted --variant-label dspy_react_adapted --dspy-program @DSPY@ --library @SHAPE@
debate_adapted|--system debate_adapted --variant-label debate_adapted --library @SHAPE@
uot_adapted|--system uot_adapted --variant-label uot_adapted --library @SHAPE@
bed_llm_adapted|--system bed_llm_adapted --variant-label bed_llm_adapted --library @SHAPE@
mediq_adapted|--system mediq_adapted --variant-label mediq_adapted --library @SHAPE@
react|--system react --library @SHAPE@
reflexion|--system reflexion --library @SHAPE@
rewoo|--system rewoo --library @SHAPE@
lats|--system lats --library @SHAPE@
llmcompiler|--system llmcompiler --library @SHAPE@
dspy_react|--system dspy_react --dspy-program @DSPY@ --library @SHAPE@
debate|--system debate --library @SHAPE@
uot|--system uot --library @SHAPE@
bed_llm|--system bed_llm --library @SHAPE@
sharma_native|--system sharma_native --library @SHAPE@
sharma_ported|--system sharma_ported --library @SHAPE@
torres_native|--system torres_native --library @SHAPE@
torres_ported|--system torres_ported --library @SHAPE@
co_investigator_native|--system co_investigator_native --library @SHAPE@
co_investigator_ported|--system co_investigator_ported --library @SHAPE@
finfre_rag_native|--system finfre_rag_native --library @SHAPE@
finfre_rag_ported|--system finfre_rag_ported --library @SHAPE@
EOF
}
# case memory: Sharma / Torres / Co-Investigator build it from TRAIN alerts by default (--memory-n 500, stratified 0.5; --no-memory is the pre-fix arm and is not used here)
is_external() { [[ $1 != ours_* ]]; }

val_parts() {  # val_parts <part ...> : every spec on the given parts
  local part spec name args shape account
  for part in "$@"; do
    if [[ $part == eval ]]; then shape=$LIB/shape_gemma.json; account=$LIB/account_gemma.json
    else shape=$LIB/shape_gemma_train.json; account=$LIB/account_gemma_train.json; fi
    while IFS='|' read -r name args; do
      [[ -n ${SPECS:-} && " $SPECS " != *" $name "* ]] && continue
      args=${args//@SHAPE@/$shape}; args=${args//@ACCOUNT@/$account}; args=${args//@DSPY@/$DSPY}
      if [[ $args == *"$DSPY"* && ! -f $DSPY && $DRY != 1 ]]; then say "skip $name: no DSPy program at $DSPY (stage val-dspy)"; continue; fi
      # shellcheck disable=SC2086
      run_chunks "$V/$part/$name" val "$part" "$(total_of $part)" "$MODEL" $args
    done < <(val_specs)
  done
}

stage_val_dspy() {
  if [[ -f $DSPY ]]; then say "dspy: program exists ($DSPY), skipped"; return; fi
  env_vllm; serve_up "$MODEL"
  run_log "$V/dspy.log" $PY scripts/optimize_dspy.py --data hi_medium --n "${DSPY_N:-40}" --model "$MODEL" --out "$(dirname "$DSPY")"
  serve_down
}
stage_val_fitcal() { env_vllm; serve_up "$MODEL"; val_parts fit cal; serve_down; }
stage_val_calibrate() {  # fit on val fit, checked on val cal (a map that does not improve cal Brier is frozen as the identity), every external
  local name sys
  mk "$V/calibration"
  while IFS='|' read -r name _; do
    is_external "$name" || continue
    [[ -f $V/calibration/$name.json ]] && continue
    if [[ $DRY == 1 ]]; then sys="<run-log system of $name>"; else
      sys=$($TOOLS runlog-system $V/fit/$name/c[0-9][0-9][0-9]) || { say "calibrate $name: no run logs, skipped"; continue; }
    fi
    # shellcheck disable=SC2086
    run_log "$V/calibration/$name.log" $PY scripts/calibrate_external.py fit --system "$sys" --runs $V/fit/$name/c[0-9][0-9][0-9] $V/cal/$name/c[0-9][0-9][0-9] \
        --fit-parts fit --check-parts cal --out "$V/calibration/$name.json"
  done < <(val_specs)
}
stage_val_eval() {  # ONCE: every system on the 160 eval alerts (chunks that exist are skipped)
  if [[ ${FORCE_EVAL:-0} != 1 && $DRY != 1 ]]; then
    local missing=0 name
    while IFS='|' read -r name _; do is_external "$name" && [[ ! -f $V/calibration/$name.json ]] && missing=$((missing + 1)); done < <(val_specs)
    (( missing > 0 )) && { say "val-eval: $missing calibration maps missing under $V/calibration (run val-calibrate first; FORCE_EVAL=1 to override)"; exit 1; }
  fi
  env_vllm; serve_up "$MODEL"; val_parts eval; serve_down
}
stage_val_apply() {  # the frozen maps applied to the eval run logs (no label read; the raw value stays in probability_raw)
  local name c f
  while IFS='|' read -r name _; do
    is_external "$name" || continue
    [[ -f $V/calibration/$name.json ]] || continue
    for c in "$V"/eval/$name/c[0-9][0-9][0-9]; do
      [[ -f $c/runlog.parquet && ! -f $V/eval_calibrated/$name/$(basename "$c").parquet ]] || continue
      mk "$V/eval_calibrated/$name"
      run $PY scripts/calibrate_external.py apply --map "$V/calibration/$name.json" --runlog "$c/runlog.parquet" --out "$V/eval_calibrated/$name/$(basename "$c").parquet"
    done
  done < <(val_specs)
}
stage_val_report() {  # CPU: tables, analysis, STATUS line drafts
  run $PY scripts/val_runs_table.py --root "$V/eval" --csv "$V/table.csv" --md docs/t3_val_runs.md
  run $PY scripts/t3_rerun_report.py analysis --root "$V"
  if [[ $DRY != 1 ]]; then
    mkdir -p "$HOME/projects/rewrite_notes/drafts"
    { echo "# STATUS draft $(date -I) (generated by scripts/t3_rerun.sh val-report; paste, do not commit blindly)"
      echo "- val rerun tree: $V ; table: $V/table.csv ; analysis: $V/analysis/analysis.json"
      head -c 1500 "$V/table.csv"; } > "$HOME/projects/rewrite_notes/drafts/status_val_lines.md"
  fi
}
stage_val() { stage_val_dspy; stage_val_fitcal; stage_val_calibrate; stage_val_eval; stage_val_apply; stage_val_report; }

# ---------------------------------------------------------------------------------------------------------------------------------
stage_e15() {  # docs/t4_injection.md on the real endpoint; val, planner (100 alerts) and react (50), gate on/off, CaMeL-style arms, keyword screen
  local set=data/safety/injection_val_diverse.jsonl
  [[ -f $set ]] || run $PY -c "from ftm.safety import injection as inj; inj.write_jsonl(inj.generate_diverse(400, 400, 0, 'val'), '$set')"
  env_vllm; serve_up "$MODEL"
  # shellcheck disable=SC2086
  run_log "$V/e15_planner.log" $PY scripts/run_injection_tests.py --system planner --alerts val --val-part eval --data hi_medium --n "${E15_N_PLANNER:-60}" \
      --llm endpoint --model "$MODEL" --backtest "$BT" --backtest-mode realistic --fusion-params "$FUSION" --set "$set" --screen heuristic \
      --arms nogate,gate,gate+camel_flow,gate+camel_full,gate+screen --machine "$MACHINE" --run-id e15_planner --out outputs/injection_v3 \
      || { say "e15_planner: exit $? (3 = an arm was NOT TESTED / not exposed, see $V/e15_planner.log); nothing is recorded as a result"; exit 3; }
  # shellcheck disable=SC2086
  run_log "$V/e15_react.log" $PY scripts/run_injection_tests.py --system react --alerts val --val-part eval --data hi_medium --n "${E15_N_REACT:-40}" \
      --llm endpoint --model "$MODEL" --backtest "$BT" --backtest-mode realistic --fusion-params "$FUSION" --set "$set" --screen heuristic \
      --arms nogate,gate,gate+screen --machine "$MACHINE" --run-id e15_react --out outputs/injection_v3 \
      || { say "e15_react: exit $? (3 = an arm was NOT TESTED / not exposed, see $V/e15_react.log); nothing is recorded as a result"; exit 3; }
  serve_down
}

stage_e18() {
  # the stale gemma result (old prompt, 6k-cap confound) is moved aside once; the refresh reruns it on the new prompt
  local d=outputs/e18_refresh/gemma4-12b
  if [[ -d $d && ! -f $d/.prompt_budget_v2 ]]; then
    mk outputs/e18_refresh_stale
    run mv "$d" "outputs/e18_refresh_stale/gemma4-12b_$(date +%s)"
  fi
  env CANDIDATES="${E18_CANDIDATES:-gemma4-12b qwen35-9b qwen35-4b gemma4-e4b qwen3-14b gpt-oss-20b phi4-14b}" bash scripts/t3_e18_refresh.sh  # honours DRY_RUN itself
  [[ $DRY == 1 ]] || { mkdir -p outputs/e18_refresh/gemma4-12b; touch outputs/e18_refresh/gemma4-12b/.prompt_budget_v2; }
}

stage_rag() {
  local st
  for st in ${RAG_STAGES:-library arms report combined eval}; do bash scripts/t3_rag_bench.sh "$st"; done  # honours DRY_RUN itself
}

stage_system1() {  # docs/t3_system1.md section 8, steps 3-8 (steps 0-2 are queue lines s1_*)
  local S=outputs/s1v1 arm part n off1 sel funcs tiers
  # 3. fast scoring of every val row (one server of its own)
  if [[ -f $S/tuned_s2/latency.json && -f $S/tuned_s0/latency.json && -f $S/tuned_s1/latency.json ]]; then say "system1 step 3: fast scoring present, skipped"; else
    run bash scripts/t3_qwen_session.sh fast3 qwen3-4b-fast; fi
  # 4. CPU: consistency + the offline report and the frozen file
  [[ -f $S/perturbed.jsonl ]] || run $PY scripts/system1_consistency.py prepare --out $S/perturbed.jsonl
  if [[ ! -f $FROZEN ]]; then
    env_vllm; serve_up qwen3-4b-fast   # the perturbed twins are scored by the same adapter
    run_log $S/tuned_s$S1_SEED/perturbed.log $PY scripts/score_fast_baseline.py --decisions $S/perturbed.jsonl score --tier fast --model fast3-s$S1_SEED \
        --out $S/tuned_s$S1_SEED/perturbed_fast.jsonl
    serve_down
    run $PY scripts/score_fast_baseline.py report --fast $S/tuned_s$S1_SEED/fast.jsonl --large $S/large/large_lp.jsonl --fast-model fast3-s$S1_SEED \
        --large-model gemma4-12b --latency $S/tuned_s$S1_SEED/latency.json --freeze $FROZEN --out $S/tuned_s$S1_SEED/report.json
    run $PY scripts/system1_consistency.py report --perturbed $S/perturbed.jsonl --orig-scores $S/tuned_s$S1_SEED/fast.jsonl \
        --pert-scores $S/tuned_s$S1_SEED/perturbed_fast.jsonl --temperatures $FROZEN --out $S/tuned_s$S1_SEED/consistency.json
  fi
  # 5. dual-serve smoke test: must answer both models, per_alert has s1_decisions > 0, no OOM
  env_vllm; serve_dual
  local s1="--system1-model fast3-s$S1_SEED --system1-endpoint ${FTM_FAST_ENDPOINT:-http://127.0.0.1:8001/v1}"
  mk outputs/s1   # the smoke log is written here before any run dir exists
  if [[ ! -f outputs/s1/smoke/summary.json ]]; then
    # shellcheck disable=SC2086
    run_log outputs/s1/smoke.log $PY scripts/run_llm_agent.py --system planner --alerts val --val-part fit --data hi_medium --n 8 $COMMON --model gemma4-12b-dual \
        $s1 --system1 $FROZEN --library $LIB/shape_gemma_train.json --run-dir outputs/s1/smoke
    [[ $DRY == 1 ]] || { grep -q '"s1_decisions": *[1-9]' outputs/s1/smoke/per_alert.jsonl || { say "system1 smoke: no System 1 decision in per_alert.jsonl, stopping"; exit 1; }; }
  fi
  # 6. integration arms on fit + cal (the `off` arm runs on the SAME dual server so only System 1 differs)
  for arm in off worth inject typology all_fast all; do
    local a
    case $arm in off) a="" ;; worth) a="--system1 $FROZEN --system1-functions worth" ;; inject) a="--system1 $FROZEN --system1-functions inject" ;;
      typology) a="--system1 $FROZEN --system1-functions typology" ;; all_fast) a="--system1 $FROZEN --system1-tiers fast" ;; all) a="--system1 $FROZEN" ;; esac
    for part in fit cal; do
      # shellcheck disable=SC2086
      run_chunks outputs/s1/fitcal/$arm/$part val $part "$(total_of $part)" gemma4-12b-dual --system planner --mode compose --library $LIB/shape_gemma_train.json $s1 $a
    done
  done
  # 7. selection on fit + cal; eval once with the frozen selection
  if [[ ! -f outputs/s1/fitcal_report/selection.json ]]; then
    local armargs=""
    for arm in worth inject typology all_fast all; do
      [[ $DRY == 1 ]] && armargs="$armargs --arm $arm=outputs/s1/fitcal/$arm/fit/c*,outputs/s1/fitcal/$arm/cal/c*" \
        || armargs="$armargs --arm $arm=$(chunk_dirs outputs/s1/fitcal/$arm/fit),$(chunk_dirs outputs/s1/fitcal/$arm/cal)"
    done
    # shellcheck disable=SC2086
    run $PY scripts/system1_report.py report --part fitcal \
        --off "off=$([[ $DRY == 1 ]] && echo 'outputs/s1/fitcal/off/{fit,cal}/c*' || echo "$(chunk_dirs outputs/s1/fitcal/off/fit),$(chunk_dirs outputs/s1/fitcal/off/cal)")" \
        $armargs --offline $S/tuned_s$S1_SEED/report.json --out outputs/s1/fitcal_report
  fi
  if [[ -f outputs/s1/eval/.done && ${FORCE_EVAL:-0} != 1 ]]; then say "system1: eval already run (once), skipped"; else
    local ea=""
    if [[ $DRY == 1 ]]; then funcs="<adopted functions>"; tiers="<fast|cascade>"; else
      local sel_args; sel_args=$($TOOLS s1-eval-args outputs/s1/fitcal_report/selection.json) || exit 1
      funcs=${sel_args%%$'\t'*}; tiers=${sel_args#*$'\t'}; fi  # not `IFS=$'\t' read`: tab is IFS whitespace, so an empty function list would shift into funcs
    [[ -n $funcs ]] && { ea="--system1 $FROZEN --system1-functions $funcs"; [[ $tiers == fast ]] && ea="$ea --system1-tiers fast"; }
    # shellcheck disable=SC2086
    run_chunks outputs/s1/eval/off val eval $EVAL_N gemma4-12b-dual --system planner --mode compose --library $LIB/shape_gemma.json $s1
    # shellcheck disable=SC2086
    [[ -n $ea ]] && run_chunks outputs/s1/eval/selected val eval $EVAL_N gemma4-12b-dual --system planner --mode compose --library $LIB/shape_gemma.json $s1 $ea
    [[ $DRY == 1 ]] || touch outputs/s1/eval/.done
    # shellcheck disable=SC2086
    run $PY scripts/system1_report.py report --part eval --selection outputs/s1/fitcal_report/selection.json \
        --off "off=$([[ $DRY == 1 ]] && echo 'outputs/s1/eval/off/c*' || chunk_dirs outputs/s1/eval/off)" \
        $([[ -n $ea ]] && echo "--arm selected=$([[ $DRY == 1 ]] && echo 'outputs/s1/eval/selected/c*' || chunk_dirs outputs/s1/eval/selected)") \
        --offline $S/tuned_s$S1_SEED/report.json --out outputs/s1/eval_report
  fi
  # 8. E15 with the fast-model screen, held-out injection families (planner and react, gate vs gate+screen)
  local set=data/safety/injection_val_diverse.jsonl
  [[ -f $set ]] || run $PY -c "from ftm.safety import injection as inj; inj.write_jsonl(inj.generate_diverse(400, 400, 0, 'val'), '$set')"
  for sysn in "planner|${E15_N_PLANNER:-60}" "react|${E15_N_REACT:-40}"; do
    # shellcheck disable=SC2086
    run_log "$V/e15_s1_${sysn%%|*}.log" $PY scripts/run_injection_tests.py --system "${sysn%%|*}" --alerts val --val-part eval --data hi_medium --n "${sysn#*|}" \
        --llm endpoint --model gemma4-12b-dual --backtest "$BT" --backtest-mode realistic --fusion-params "$FUSION" --set "$set" \
        --screen fastmodel --screen-endpoint "${FTM_FAST_ENDPOINT:-http://127.0.0.1:8001/v1}" --screen-model fast3-s$S1_SEED \
        --arms nogate,gate,gate+screen --machine "$MACHINE" --run-id "e15_s1_${sysn%%|*}" --out outputs/injection_v3 \
      || { say "e15_s1_${sysn%%|*}: exit $? (3 = an arm was NOT TESTED / not exposed, see $V/e15_s1_${sysn%%|*}.log); nothing is recorded as a result"; exit 3; }
  done
  serve_down
}

stage_plan() {
  $TOOLS hours
  echo "see docs/t3_rerun.md for the stage hours and the queue lines"
}

for st in "$@"; do
  case $st in
    plan) stage_plan ;;
    teacher) stage_teacher ;;
    library) stage_library ;;
    distill) stage_distill ;;
    student-train) stage_student_train ;;
    student-e16) stage_student_eval ;;
    student) stage_student_train; stage_student_eval ;;
    val-dspy) stage_val_dspy ;;
    val-fitcal) stage_val_fitcal ;;
    val-calibrate) stage_val_calibrate ;;
    val-eval) stage_val_eval ;;
    val-apply) stage_val_apply ;;
    val-report) stage_val_report ;;
    val) stage_val ;;
    e15) stage_e15 ;;
    e18) stage_e18 ;;
    rag) stage_rag ;;
    system1) stage_system1 ;;
    all) for s in teacher library distill student val e15 e18 rag system1; do "$0" "$s" || exit $?; done ;;
    *) echo "unknown stage: $st (plan teacher library distill student[-train|-e16] val[-dspy|-fitcal|-calibrate|-eval|-apply|-report] e15 e18 rag system1 all)" >&2; exit 2 ;;
  esac
done
