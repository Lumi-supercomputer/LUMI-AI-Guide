#!/bin/bash
#SBATCH --account=project_xxxxxxxxx
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=2:00:00
#SBATCH --output=single_GPU.out.2h

# this module facilitates the use of singularity containers on LUMI
module use  /appl/local/containers/ai-modules
module load singularity-AI-bindings

# choose container that is copied over by set_up_environment.sh
CONTAINER=../resources/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif

# add path to additional packages in squasfs file
export SINGULARITYENV_PREPEND_PATH=/user-software/bin
# bind squashfs file into container and run python script inside container 
srun singularity exec  -B ../resources/visualtransformer-env.sqsh:/user-software:image-src=/ $CONTAINER bash -c '
  time cp -a ../resources/train_images.hdf5 /tmp/. ;
  time python visualtransformer_ramfs.py  ;
  time /bin/cp -a /tmp/vit_b_16_imagenet.pth ./vit_b_16_imagenet.pth.$$ ;
  time /bin/cp -a /tmp/train_images.hdf5     ../resources/train_images.hdf5.$$'
