#!/bin/bash
#SBATCH --gpus=1
#SBATCH -t 0-12:00:00
#SBATCH -J essflow_materials
#SBATCH -o slurm-%x-%A_%a.out
#SBATCH --array=1-8

# ESS-Flow on a material generation task. Each array job runs 125 chains, so
# the 8 jobs give 1000 samples (one per chain after burn-in).
# Usage: sbatch scripts/slurm/materials/essflow.sh <task>
#   <task>: bulk_modulus | shear_modulus | band_gap | e_hull | space_group
# band_gap needs a longer time limit, e.g. sbatch -t 3-00:00:00 ...

source scripts/slurm/activate.sh essflow-materials
TASK=$1

python -m experiments.materials.essflow \
    --checkpoint pretrained/flowmm_mp20/checkpoints/flowmm_mp20.ckpt \
    --task "$TASK" \
    --num-chains 125 \
    --output "results/materials/$TASK/essflow_${SLURM_ARRAY_TASK_ID}.pt"
