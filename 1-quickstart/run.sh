#!/bin/bash
#SBATCH --account=project_462000131  # project account to bill 
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

# choose container
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif

singularity run $SIF bash -c 'source /scratch/project_462000131/marlonto/LUMI-AI-Guide/resources/ai-guide-env/bin/activate && python visiontransformer.py' 
