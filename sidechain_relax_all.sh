#!/bin/bash
#SBATCH --job-name=sidechain_relax
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH --partition=main-cpu,long-cpu
#SBATCH --array=0-76
#SBATCH --output=logs/sidechain_relax_%A_%a.out
#SBATCH --error=logs/sidechain_relax_%A_%a.err

INDEX=$SLURM_ARRAY_TASK_ID
which python
python -m bioemu.sidechain_relax --sequence-idx $INDEX