#!/bin/bash
#SBATCH --job-name=comp-vit
#SBATCH --output=./run-scripts/training-benchmarks/comp-vision-transformer-%j.out
#SBATCH --account=project_xxxxxxxxx
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:15:00

# this module facilitates the use of singularity containers on LUMI
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# choose container
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif

export MPICH_MPIIO_STATS=1
export MPICH_MEMORY_REPORT=1

# set MIOPEN temp folder
MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR/cache
export MIOPEN_USER_DB=$MIOPEN_DIR/config

# Set your TORCH_HOME cache to scratch to avoid saving to home directory
# https://docs.pytorch.org/docs/2.11/hub.html#where-are-my-downloaded-models-saved
export TORCH_HOME="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/torch_home"
mkdir -p "$TORCH_HOME"


if [[ $1 == 'squashfs' ]]; then
    SQUASH=data-formats/squashfs/train.squashfs
    IMAGES=/
    SQUASHVAL=data-formats/squashfs/val.squashfs
    srun singularity run -B $SQUASH:/train_images:image-src=$IMAGES -B $SQUASHVAL:/val_images:image-src=$IMAGES $SIF bash -c 'python run-scripts/training-benchmarks/compare-dataset-training.py -n 7 -ff "squashfs"'
elif [[ $1 == 'lmdb' ]]; then
    export SINGULARITYENV_PREPEND_PATH=/user-software/bin # gives access to packages inside the container
    srun singularity run -B venv.sqsh:/user-software:image-src=/ $SIF bash -c 'python run-scripts/training-benchmarks/compare-dataset-training.py -n 7 -ff "lmdb"'
elif [[ $1 == 'hdf5' ]]; then
    srun singularity run $SIF bash -c 'python run-scripts/training-benchmarks/compare-dataset-training.py -n 7 -ff "hdf5"'
fi
