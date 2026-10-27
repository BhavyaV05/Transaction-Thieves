#!/usr/bin/env bash
# One-at-a-time GPU queue for T3 (the 16 GB card runs one job at a time). Detached, resumable, extendable while it runs.
#
#   setsid nohup scripts/gpu_queue.sh > outputs/t3_queue/runner.log 2>&1 &     # start (from any checkout; it runs the RUN worktree)
#   echo "name | <git sha or HEAD> | command" >> outputs/t3_queue/queue.txt     # add a step at any time
#   touch outputs/t3_queue/STOP                                               # finish the current step, then exit
#
# Each step runs in the run worktree (default ~/projects/tt-run, pinned: the step line names the commit it is checked out at, so
# edits in the main checkout never change a running job) with PYTHONPATH set to it; log: outputs/t3_queue/<name>.log; marker:
# outputs/t3_queue/done/<name> (exit 0) or failed/<name> (non-zero; the queue goes on with the next step). Steps are taken in file
# order; a step with a marker is skipped, so after a reboot start the runner again and it resumes (long steps resume themselves:
# chunked runs skip finished chunks, training scripts have --resume). Every step's start and end time is appended to
# outputs/t3_queue/windows.log for GPU_CALENDAR.md.
set -uo pipefail
MAIN=${MAIN:-$HOME/projects/Transaction-Thieves}
RUN=${RUN:-$HOME/projects/tt-run}
Q=$MAIN/outputs/t3_queue
mkdir -p "$Q/done" "$Q/failed"; touch "$Q/queue.txt"
export PYTHONPATH=$RUN
while [[ ! -f $Q/STOP ]]; do
  name=""
  while IFS='|' read -r n sha cmd; do
    n=$(echo "$n" | xargs); [[ -z $n || $n == \#* ]] && continue
    [[ -f $Q/done/$n || -f $Q/failed/$n ]] && continue
    name=$n; sha=$(echo "$sha" | xargs); cmd=$(echo "$cmd" | sed 's/^ *//'); break
  done < "$Q/queue.txt"
  if [[ -z $name ]]; then sleep 60; continue; fi
  echo "$(date -Is) START $name ($sha)" | tee -a "$Q/windows.log"
  ( cd "$RUN" && git checkout -q --detach "$sha" && cd "$RUN" && set -a && source .env 2>/dev/null; set +a; cd "$RUN" && bash -c "$cmd" ) > "$Q/$name.log" 2>&1
  rc=$?
  echo "$(date -Is) END   $name rc=$rc" | tee -a "$Q/windows.log"
  if (( rc == 0 )); then touch "$Q/done/$name"; else touch "$Q/failed/$name"; fi
  sleep 15
done
echo "$(date -Is) STOP file seen, queue runner exits"
