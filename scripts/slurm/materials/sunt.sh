#!/bin/bash
#SBATCH --gpus=1
#SBATCH -t 0-04:00:00
#SBATCH -J sunt_materials
#SBATCH -o slurm-%x-%j.out

# S.U.N.T. rate of the samples of one method on one task.
# Usage: sbatch scripts/slurm/materials/sunt.sh <method> <task>
#   <method>: unconditional | dflow | pnpflow | daps | essflow | reweighted

source scripts/slurm/activate.sh essflow-materials
METHOD=$1
TASK=$2
shopt -s nullglob

python -m experiments.materials.sunt \
    --task "$TASK" \
    --samples results/materials/$TASK/${METHOD}.pt results/materials/$TASK/${METHOD}_[0-9]*.pt \
    --output "results/materials/$TASK/sunt_$METHOD.json"
