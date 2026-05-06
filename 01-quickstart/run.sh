#!/bin/bash
#SBATCH --account=project_xxxxxxxxx  # project account to bill 
#SBATCH --partition=small-g          # other options are small-g and standard-g
#SBATCH --gpus-per-node=1            # Number of GPUs per node (max of 8)
#SBATCH --ntasks-per-node=1          # Use one task for one GPU
#SBATCH --cpus-per-task=7            # Use 1/8 of all available 56 CPUs on LUMI-G nodes
#SBATCH --mem-per-gpu=60G            # CPU RAM per GPU (GPU memory is always 64GB per GPU)
#SBATCH --time=1:00:00               # time limit

# this module facilitates the use of singularity containers on LUMI
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# set MIOPEN temp folder
MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR/cache
export MIOPEN_USER_DB=$MIOPEN_DIR/config

# Set your TORCH_HOME cache to scratch to avoid saving to home directory
# https://docs.pytorch.org/docs/2.11/hub.html#where-are-my-downloaded-models-saved
export TORCH_HOME="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/torch_home"
mkdir -p "$TORCH_HOME"

# choose container
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif

singularity run $SIF bash -c 'python visiontransformer.py'