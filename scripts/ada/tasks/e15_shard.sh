#!/bin/bash
# One shard of the E15 injection stage on ada: Gemma 4 12B TP2 on 2 GPUs, the stage's own command (scripts/t3_rerun.sh stage_e15) over a slice of the SAME alert list.  Owner: T3 (Akshat).
#   usage: e15_shard.sh <planner|react> <i> <k>      shard i of k (1-based) of the first N alerts (N = E15_N_PLANNER 60 / E15_N_REACT 40, the stage's defaults)
# Shards write outputs/injection_v3/e15_<system>_s<i>/ ; when all k exist, scripts/ada/merge_injection_shards.py makes outputs/injection_v3/e15_<system>/ (the format the report reads).
source $(dirname $0)/common.sh; SYS=$1; I=$2; K=$3; D=$RES/e15_${SYS}_s$I; mkdir -p $D; cd $S1; export PYTHONPATH=$PWD; SERVELOG=$D/serve.log
exec > >(tee -a $D/task.log) 2>&1
setsid $PY scripts/serve_llm.py --model gemma4-12b --turing --tp 2 --gpu-util 0.88 --port $PORT > $SERVELOG 2>&1 &
pid=$!; trap 'kill -- -$pid 2>/dev/null' EXIT
wait_server gemma4-12b $pid || fail "gemma tp2 did not serve (see $SERVELOG)"
cd $HOME/tt/tt-main-git; export PYTHONPATH=$PWD FTM_P0_VARIANT=v2 FTM_EDGE_SCORER=v2
OUT=outputs/injection_v3; mkdir -p $OUT outputs/t3_val_v3; SET=data/safety/injection_val_diverse.jsonl
[ -f $SET ] || $PY -c "from ftm.safety import injection as inj; inj.write_jsonl(inj.generate_diverse(400, 400, 0, 'val'), '$SET')"
if [ $SYS = planner ]; then N=${E15_N_PLANNER:-60}; ARMS=nogate,gate,gate+camel_flow,gate+camel_full,gate+screen; else N=${E15_N_REACT:-40}; ARMS=nogate,gate,gate+screen; fi
IDS=$D/alert_ids.txt
# the unsharded run's alert list (same call as run_injection_tests.py), split into k contiguous slices
$PY - $N $I $K $IDS <<'PY' || fail "could not build the alert list"
import importlib.util, sys
sys.path.insert(0, "scripts")
import run_injection_tests as rit
n, i, k, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
rla = rit.load_run_llm_agent()
tools, _ = rla.make_toolbox("harness", 0, "hi_medium", "val")
alerts = rla.data_alerts("hi_medium", "val", n, 0, 40, "eval", rla.askable_ids(tools))
ids = [c.alert_id for c, _ in alerts]
lo, hi = (i - 1) * len(ids) // k, i * len(ids) // k
open(out, "w").write("".join(a + "\n" for a in ids[lo:hi]))
print(f"shard {i}/{k}: alerts {lo}..{hi} of {len(ids)}")
PY
NSH=$(wc -l < $IDS)
$PY scripts/run_injection_tests.py --system $SYS --alerts val --val-part eval --data hi_medium --n $NSH --alert-ids $IDS \
    --llm endpoint --model gemma4-12b --backtest configs/fusion/headroom_v1_table.jsonl --backtest-mode realistic --fusion-params configs/fusion/headroom_v1.json \
    --set $SET --screen heuristic --arms $ARMS --machine ada-2080ti-tp2 --run-id e15_${SYS}_s$I --out $OUT > $D/run.log 2>&1
rc=$?; tail -3 $D/run.log
[ $rc = 0 ] || fail "e15 $SYS shard $I exit $rc (3 = an arm NOT TESTED; see $D/run.log)"
echo "SHARD_DONE $SYS $I/$K"
