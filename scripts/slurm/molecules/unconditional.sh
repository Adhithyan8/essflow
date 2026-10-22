#!/bin/bash
#SBATCH --gpus=1
#SBATCH -t 0-04:00:00
#SBATCH -J unconditional_molecules
#SBATCH -o slurm-%x-%j.out

# Unconditional GeoLDM samples with the DDPM (1000 steps) or DDIM (50 steps) sampler.
# Usage: sbatch scripts/slurm/molecules/unconditional.sh <ddpm | ddim>

source scripts/slurm/activate.sh essflow-molecules
SAMPLER=$1

python -m experiments.molecules.unconditional \
    --model-path pretrained/geoldm_drugs \
    --sampler "$SAMPLER" \
    --output "results/molecules/unconditional_$SAMPLER.pt"
