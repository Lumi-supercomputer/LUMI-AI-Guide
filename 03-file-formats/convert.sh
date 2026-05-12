#!/bin/bash
#SBATCH --account=project_xxxxxxxxx
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1750
#SBATCH --time=0:20:00

# this module facilitates the use of singularity containers on LUMI
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# choose container
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif


if [[ $1 == 'squashfs' ]]; then
    mkdir -p data-formats/squashfs/
    time srun bash -c 'mksquashfs data-formats/raw/tiny-imagenet-200/val/ data-formats/squashfs/val.squashfs -processors 16 -no-progress -no-xattrs -noappend'
    time srun bash -c 'mksquashfs data-formats/raw/tiny-imagenet-200/train/ data-formats/squashfs/train.squashfs -processors 16 -no-progress -no-xattrs -noappend'
elif [[ $1 == 'lmdb' ]]; then
    mkdir -p data-formats/lmdb/
    export SINGULARITYENV_PREPEND_PATH=/user-software/bin # gives access to packages inside the container
    time srun singularity run -B venv.sqsh:/user-software:image-src=/ $SIF bash -c 'python scripts/lmdb/convert_to_lmdb.py'
elif [[ $1 == 'hdf5' ]]; then
    mkdir -p data-formats/hdf5/
    time srun singularity run $SIF bash -c 'python scripts/hdf5/convert_to_hdf5.py'
fi
