#!/bin/bash
#SBATCH -p u22
#SBATCH -A research
#SBATCH -q medium
#SBATCH -C 2080ti
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 2-00:00:00
#SBATCH -J s1b_fast
#SBATCH -o /home2/garimella.abhishek/tt/logs/s1b_fast_%j.out
# S1BENCH GPU steps on ada (one RTX 2080 Ti, sm_75, fp16): System 1 adapters + the untuned Qwen3-4B-AWQ base, served by vLLM 0.31.0 (cu129 wheel),
# Part A (public benchmarks + S1MB, latency) and Part B (our decisions).  Owner: T3 (Akshat).  Resumable: every runner skips finished tasks.
#   sbatch scripts/ada/s1bench_fast.sbatch            (from ~/tt/tt-s1bench)
set -uo pipefail
cd ~/tt/tt-s1bench
export HF_HOME=/share1/garimella.abhishek/hf HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export S1BENCH_PREPARED=$HOME/tt/s1bench_data/prepared
export VLLM_API_KEY=$(python3 -c "import secrets;print(secrets.token_urlsafe(24))") FTM_LLM_API_KEY=$VLLM_API_KEY
export FTM_LLM_ENDPOINT=http://127.0.0.1:8000/v1 PYTHONPATH=$PWD
PY=$HOME/venv312_vllm/bin/python
export PATH=$HOME/venv312_vllm/bin:$PATH
OUT=outputs/s1bench; mkdir -p $OUT/ada outputs/s1v1
T0=$(date +%s)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee $OUT/ada/gpu_fast.txt

setsid $PY scripts/serve_llm.py --model qwen3-4b-fast --turing --gpu-util 0.85 --lora-only fast3-s0,fast3soft-s0 > $OUT/ada/serve_fast.log 2>&1 &
spid=$!
trap 'kill -- -$spid 2>/dev/null; wait $spid 2>/dev/null' EXIT
t0=$(date +%s)
until curl -sf -H "Authorization: Bearer $VLLM_API_KEY" $FTM_LLM_ENDPOINT/models | grep -q '"qwen3-4b-fast"'; do
  kill -0 $spid 2>/dev/null || { echo "server died"; tail -40 $OUT/ada/serve_fast.log; exit 1; }
  (( $(date +%s) - t0 > 2400 )) && { echo "server did not load"; exit 1; }
  sleep 5
done
echo "$(date -Is) server up after $(( $(date +%s) - t0 ))s"

part_a() {  # part_a <tag> <served model name or adapter>
  local tag=$1 m=$2
  $PY scripts/s1bench_run.py ext   --tag "$tag" --model "$m" --tasks all --perm --workers 32 --resume --out $OUT > $OUT/$tag.ext.log 2>&1
  $PY scripts/s1bench_run.py s1mb  --tag "$tag" --model "$m" --benchmarks all --perm --workers 32 --resume --out $OUT > $OUT/$tag.s1mb.log 2>&1
  $PY scripts/s1bench_run.py latency --tag "$tag" --model "$m" --tasks all --latency-n 200 --out $OUT > $OUT/$tag.lat.log 2>&1
  echo "$(date -Is) part A done: $tag  ($(( $(date +%s) - T0 ))s since start)"
}
part_b() {  # part_b <arm dir> <served name>: our decisions (val rows), then one-request-at-a-time latency
  local arm=$1 m=$2 d=outputs/s1v1/$1; mkdir -p $d
  [ -s $d/fast.jsonl ] || $PY scripts/score_fast_baseline.py score --tier fast --model "$m" --workers 16 --out $d/fast.jsonl > $d/score.log 2>&1
  [ -s $d/latency_rows.jsonl ] || $PY scripts/score_fast_baseline.py score --tier fast --model "$m" --workers 1 --limit 300 --out $d/latency_rows.jsonl > $d/latency.log 2>&1
  echo "$(date -Is) part B done: $arm"
}
# Part B first (small, and the selection rule needs it), then Part A
part_b untuned_4b qwen3-4b-fast
part_b tuned_s0 fast3-s0
part_b tuned_soft_s0 fast3soft-s0
part_a system1 fast3-s0
part_a qwen3-4b qwen3-4b-fast
part_a system1_soft fast3soft-s0
$PY scripts/s1bench_run.py ext --tag system1_listed --model fast3-s0 --variant listed --tasks banking77 clinc150_oos hwu64 --perm --workers 32 --resume --out $OUT > $OUT/system1_listed.ext.log 2>&1
echo "WALL_SECONDS $(( $(date +%s) - T0 )) s1bench_fast on 1x RTX 2080 Ti"
