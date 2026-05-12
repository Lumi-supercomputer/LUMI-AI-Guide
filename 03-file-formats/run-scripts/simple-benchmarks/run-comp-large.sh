#!/bin/bash
#SBATCH --job-name=comp-large
#SBATCH --output=./run-scripts/simple-benchmarks/comp-large-%j.out
#SBATCH --account=project_462000002
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=04:00:00

# this module facilitates the use of singularity containers on LUMI
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# choose container
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif

echo "Warning: This benchmark requires significant resources"
export MPICH_MPIIO_STATS=1
export MPICH_MEMORY_REPORT=1

if [[ $1 == 'squashfs' ]]; then
    SQUASH=/scratch/project_462000002/joachimsode/file-format-ai-benchmark/imagenet-object-localization-challenge.squashfs
    IMAGES=/Data/CLS-LOC/train/
    srun singularity exec -B $SQUASH:/train_images:image-src=$IMAGES $CONTAINER bash -c 'python run-scripts/simple-benchmarks/compare-dataset-large.py -n 7 -ff "squashfs" -N 200000'
elif [[ $1 == 'lmdb' ]]; then
    srun singularity exec  $CONTAINER bash -c 'python run-scripts/simple-benchmarks/compare-dataset-large.py -n 7 -ff "lmdb" -N 200000'
elif [[ $1 == 'hdf5' ]]; then
    echo "Error: HDF5 is incompatible with large imagenet dataset."
fi
