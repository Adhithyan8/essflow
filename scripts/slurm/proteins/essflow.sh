#!/bin/bash
#SBATCH --gpus=1
#SBATCH -t 0-12:00:00
#SBATCH -J essflow_proteins
#SBATCH -o slurm-%x-%j.out

# ESS-Flow for designable protein backbones (16 chains x 64 samples after 200 burn-in steps).
# Usage: sbatch scripts/slurm/proteins/essflow.sh <reward weight>   (0.5, 1.0 or 2.0 in the paper)

source scripts/slurm/activate.sh essflow-proteins
WEIGHT=$1

python -m experiments.proteins.essflow \
    --weight "$WEIGHT" \
    --output "results/proteins/essflow_w$WEIGHT.pt"
