#!/bin/bash
#SBATCH --account=project_xxxxxxxxx  # project account to bill 
#SBATCH --partition=small-g          # other options are small-g and standard-g
#SBATCH --gpus-per-node=1            # Number of GPUs per node (max of 8)
#SBATCH --ntasks-per-node=1          # Use one task for one GPU
#SBATCH --cpus-per-task=7            # Use 1/8 of all available 56 CPUs on LUMI-G nodes
#SBATCH --mem-per-gpu=60G            # CPU RAM per GPU (GPU memory is always 64GB per GPU)
#SBATCH --time=1:00:00               # time limit

# this module facilitates the use of singularity containers on LUMI
module use  /appl/local/containers/ai-modules
module load singularity-AI-bindings

# choose container that is copied over by set_up_environment.sh
CONTAINER=../resources/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.0.sif

# add path to additional packages in squasfs file
export SINGULARITYENV_PREPEND_PATH=/user-software/bin
# bind squashfs file into container and run python script inside container 
singularity exec -B ../resources/visiontransformer-env.sqsh:/user-software:image-src=/ $CONTAINER python visiontransformer.py
