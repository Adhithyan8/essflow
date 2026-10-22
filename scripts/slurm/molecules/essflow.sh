#!/bin/bash
#SBATCH --gpus=1
#SBATCH -t 0-10:00:00
#SBATCH -J essflow_molecules
#SBATCH -o slurm-%x-%j.out

# ESS-Flow for drug-like molecules (16 chains x 64 samples after 200 burn-in steps).
# Usage: sbatch scripts/slurm/molecules/essflow.sh <reward weight>   (5, 10 or 20 in the paper)

source scripts/slurm/activate.sh essflow-molecules
WEIGHT=$1

python -m experiments.molecules.essflow \
    --model-path pretrained/geoldm_drugs \
    --reward-weight "$WEIGHT" \
    --output "results/molecules/essflow_w$WEIGHT.pt"
