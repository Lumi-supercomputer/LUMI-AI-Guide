#!/bin/bash
#SBATCH --job-name=comp-tiny
#SBATCH --output=./run-scripts/simple-benchmarks/comp-tiny-%j.out
#SBATCH --account=project_462000002
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:10:00

# this module facilitates the use of singularity containers on LUMI
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# choose container
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif

export MPICH_MPIIO_STATS=1
export MPICH_MEMORY_REPORT=1

if [[ $1 == 'squashfs' ]]; then
    SQUASH=data-formats/squashfs/train.squashfs
    IMAGES=/
    srun singularity exec -B $SQUASH:/train_images:image-src=$IMAGES $CONTAINER bash -c 'python run-scripts/simple-benchmarks/compare-dataset-tiny.py -n 7 -ff "squashfs" -N 100000'
elif [[ $1 == 'lmdb' ]]; then
    srun singularity exec  $CONTAINER bash -c 'python run-scripts/simple-benchmarks/compare-dataset-tiny.py -n 7 -ff "lmdb" -N 100000'
elif [[ $1 == 'hdf5' ]]; then
    srun singularity exec $CONTAINER bash -c 'python run-scripts/simple-benchmarks/compare-dataset-tiny.py -n 7 -ff "hdf5" -N 100000'
fi
