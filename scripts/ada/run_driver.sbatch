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
# Self-driving: no installs here (the CPU job `venv312` did them).  Copies the pre-downloaded HF weights (home) to the node SSD, then runs the task list.
hostname; nvidia-smi -L
SC=/ssd_scratch/$USER; df --output=avail -BG /ssd_scratch | tail -1 | tr -dc 0-9 | awk '{ if ($1 < 80) exit 1 }' || SC=/tmp/$USER   # other users sometimes fill /ssd_scratch
export SCRATCH_DIR=$SC; mkdir -p $SC/hf $SC/tmp
cp -a $HOME/tt/hf/. $SC/hf/ && echo "HF cache copied to $SC/hf"
HF_HOME=$SC/hf nohup $HOME/venv312/bin/hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 > $HOME/tt/results/prefetch_qwen8b.log 2>&1 &
HF_HOME=$SC/hf $HOME/venv312/bin/python $HOME/tt/ada_scripts/driver.py
