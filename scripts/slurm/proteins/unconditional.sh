#!/bin/bash
#SBATCH --gpus=1
#SBATCH -t 0-12:00:00
#SBATCH -J unconditional_proteins
#SBATCH -o slurm-%x-%j.out

# Unconditional Proteina samples (1024 backbones), evaluated for designability.
# Usage: sbatch scripts/slurm/proteins/unconditional.sh

source scripts/slurm/activate.sh essflow-proteins

python -m experiments.proteins.unconditional \
    --output results/proteins/unconditional.pt
