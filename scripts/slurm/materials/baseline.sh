#!/bin/bash
#SBATCH --gpus=1
#SBATCH -t 2-00:00:00
#SBATCH -J baseline_materials
#SBATCH -o slurm-%x-%j.out

# Unconditional samples or a gradient-based baseline on a material generation
# task, with the hyperparameters used in the paper (1000 samples).
# Usage: sbatch scripts/slurm/materials/baseline.sh <method> <task>
#   <method>: unconditional | dflow | pnpflow | daps
#   <task>:   bulk_modulus | shear_modulus | band_gap | e_hull (| space_group for unconditional)

source scripts/slurm/activate.sh essflow-materials
METHOD=$1
TASK=$2

python -m experiments.materials.$METHOD \
    --checkpoint pretrained/flowmm_mp20/checkpoints/flowmm_mp20.ckpt \
    --task "$TASK" \
    --num-samples 1000 \
    --output "results/materials/$TASK/$METHOD.pt"
