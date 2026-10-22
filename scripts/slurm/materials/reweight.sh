#!/bin/bash
#SBATCH --gpus=1
#SBATCH -t 0-02:00:00
#SBATCH -J reweight_materials
#SBATCH -o slurm-%x-%j.out

# Multi-fidelity ESS-Flow: reweights the ESS-Flow samples (50 ODE steps) of a
# task towards the target with 1000 ODE steps and resamples them.
# Usage: sbatch scripts/slurm/materials/reweight.sh <task>
#   <task>: bulk_modulus | shear_modulus | band_gap | e_hull

source scripts/slurm/activate.sh essflow-materials
TASK=$1

python -m experiments.materials.reweight \
    --checkpoint pretrained/flowmm_mp20/checkpoints/flowmm_mp20.ckpt \
    --task "$TASK" \
    --samples results/materials/$TASK/essflow_*.pt \
    --ode-steps-high 1000 \
    --output "results/materials/$TASK/reweighted.pt"
