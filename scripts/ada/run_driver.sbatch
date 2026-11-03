#!/bin/bash
#SBATCH -p u22
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -N1
#SBATCH -C 2080ti
#SBATCH --gres=gpu:4
#SBATCH -c 24
#SBATCH --mem=100G
#SBATCH -t 2-00:00:00
#SBATCH -J ada_drv
#SBATCH -o /home2/garimella.abhishek/tt/logs/ada_drv_%j.out
# Self-bootstrapping: builds venv + wheels + HF cache on the node SSD (bootstrap_ssd.sh), then runs the task list.  Nothing is installed into the NFS home.
hostname; nvidia-smi -L
for d in /ssd_scratch /tmp; do [ "$(df --output=avail -BG $d | tail -1 | tr -dc 0-9)" -gt 80 ] && { SC=$d/$USER/$SLURM_JOB_ID; break; }; done; [ -n "$SC" ] || { echo "no node-local disk with 80 GB"; exit 1; }   # other users sometimes fill /ssd_scratch
mkdir -p $SC/tmp; export SCRATCH_DIR=$SC VENV=$SC/venv HF_HOME=$SC/hf
T0=$(date +%s); cp $HOME/tt/ada_scripts/bootstrap_ssd.sh $SC/b.sh; bash $SC/b.sh $SC > $HOME/tt/logs/bootstrap_$SLURM_JOB_ID.out 2>&1 || { echo BOOTSTRAP_FAILED; tail -30 $HOME/tt/logs/bootstrap_$SLURM_JOB_ID.out; exit 1; }
echo "bootstrap wall $(( $(date +%s) - T0 ))s"
$VENV/bin/python $HOME/tt/ada_scripts/driver.py
rm -rf $SC
