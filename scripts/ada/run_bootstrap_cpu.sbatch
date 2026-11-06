#!/bin/bash
#SBATCH -p u22
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -N1
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 3:00:00
#SBATCH -J boot_cpu
#SBATCH -x gnode012
#SBATCH -o /home2/garimella.abhishek/tt/logs/boot_cpu_%j.out
hostname
for d in /ssd_scratch /tmp; do [ "$(df --output=avail -BG $d | tail -1 | tr -dc 0-9)" -gt 80 ] && { SC=$d/$USER/$SLURM_JOB_ID; break; }; done; echo SC=$SC
mkdir -p $SC
cp $HOME/tt/ada_scripts/bootstrap_ssd.sh $SC/b.sh; bash $SC/b.sh $SC
echo BOOT_RC $?
rm -rf $SC
