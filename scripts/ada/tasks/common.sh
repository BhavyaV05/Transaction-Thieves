# sourced by every task script.  Owner: T3 (Akshat).  Everything runs from the shared-home ~/venv312 (torch 2.13+cu129, vLLM 0.31.0, PyG, transformers, peft, bnb).
export PY=${VENV:-$HOME/venv312}/bin/python PATH=${VENV:-$HOME/venv312}/bin:$PATH
export RES=$HOME/tt/results; mkdir -p $RES
GPU0=${CUDA_VISIBLE_DEVICES%%,*}; export PORT=$((8100 + ${GPU0:-0}))
export GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
SC=${SCRATCH_DIR:-/ssd_scratch/$USER}; export HF_HOME=$SC/hf HF_HUB_DISABLE_PROGRESS_BARS=1 TMPDIR=$SC/tmp; mkdir -p $TMPDIR
export VLLM_API_KEY=$($PY -c "import secrets;print(secrets.token_urlsafe(24))"); export FTM_LLM_API_KEY=$VLLM_API_KEY FTM_LLM_ENDPOINT=http://127.0.0.1:$PORT/v1
export S1BENCH_PREPARED=$HOME/tt/s1bench_data/prepared S1BENCH_CACHE=$HOME/tt/s1bench_data S1BENCH_DTYPE=float16
S1=$HOME/tt/tt-s1bench
wait_server() {  # wait_server <served name> <pid>
  local t0=$(date +%s)
  until curl -sf -H "Authorization: Bearer $VLLM_API_KEY" $FTM_LLM_ENDPOINT/models | grep -q "\"$1\""; do
    kill -0 $2 2>/dev/null || { echo "server died"; tail -60 ${SERVELOG:-/dev/null}; return 1; }
    (( $(date +%s) - t0 > 3000 )) && { echo "server did not load in 50 min"; return 1; }
    sleep 5
  done; echo "$(date -Is) server up after $(( $(date +%s) - t0 ))s on $GPU_NAME"
}
