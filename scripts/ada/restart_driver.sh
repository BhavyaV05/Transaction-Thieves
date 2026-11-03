#!/bin/bash
# usage (inside a step of the job): restart_driver.sh <jobid>: freeze the sbatch (it would rm -rf the SSD scratch when its driver exits), stop the old driver and its tasks, start a new driver
J=$1; SC=/ssd_scratch/$USER/$J
SB=$(pgrep -f "^/bin/bash /var/spool/slurmd/job$J/slurm_script"); echo "sbatch pid $SB"
kill -STOP $SB
for p in $(pgrep -f "[a]da_scripts/driver.py"); do kill $p; done; sleep 1
for p in $(pgrep -f "[a]da_scripts/tasks/|[t]rain_gnn.py"); do kill $p; done; sleep 3
rm -f ~/tt/results/done/gnn_* 
export SCRATCH_DIR=$SC VENV=$SC/venv HF_HOME=$SC/hf
exec $VENV/bin/python $HOME/tt/ada_scripts/driver.py
