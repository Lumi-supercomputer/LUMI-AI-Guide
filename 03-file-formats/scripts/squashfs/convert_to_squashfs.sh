#!/bin/bash
#SBATCH --account=project_462000002
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1750
#SBATCH --time=1:00:00

mkdir -p data-formats/squashfs/

srun bash -c 'mksquashfs data-formats/raw/tiny-imagenet-200/val/ data-formats/squashfs/val.squashfs -processors 1 -no-progress'
srun bash -c 'mksquashfs data-formats/raw/tiny-imagenet-200/train/ data-formats/squashfs/train.squashfs -processors 1 -no-progress'
