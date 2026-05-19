#!/bin/bash
#SBATCH --job-name=comp-tiny-seq
#SBATCH --output=./run-scripts/simple-benchmarks/comp-tiny-seq-%j.out
#SBATCH --account=project_xxxxxxxxx
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:40:00

# this module facilitates the use of singularity containers on LUMI
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# choose container
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif

if [[ $1 == 'squashfs' ]]; then
    SQUASH=data-formats/squashfs/train.squashfs
    IMAGES=/
    srun singularity run -B $SQUASH:/train_images:image-src=$IMAGES $SIF bash -c 'python run-scripts/simple-benchmarks/compare-dataset-tiny.py -n 1 -ff "squashfs" -N 100000'
elif [[ $1 == 'lmdb' ]]; then
    export SINGULARITYENV_PREPEND_PATH=/user-software/bin # gives access to packages inside the container
    srun singularity run -B venv.sqsh:/user-software:image-src=/ $SIF bash -c 'python run-scripts/simple-benchmarks/compare-dataset-tiny.py -n 1 -ff "lmdb" -N 100000'
elif [[ $1 == 'hdf5' ]]; then
    srun singularity run $SIF bash -c 'python run-scripts/simple-benchmarks/compare-dataset-tiny.py -n 1 -ff "hdf5" -N 100000'
fi

