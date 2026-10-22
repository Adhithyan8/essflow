# Activates the mamba environment given as the first argument inside a slurm job.
# The module below is specific to the Berzelius cluster. On other clusters,
# replace it with whatever makes `mamba activate` available in a job.
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate "$1"
cd "${SLURM_SUBMIT_DIR:-.}"
