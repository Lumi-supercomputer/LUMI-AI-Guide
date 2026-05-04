#!/bin/bash
#SBATCH --account=project_xxxxxxxxx
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=2:00:00

# this module facilitates the use of singularity containers on LUMI
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# set MIOPEN temp folder
MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR/cache
export MIOPEN_USER_DB=$MIOPEN_DIR/config

# choose container
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif

# run python script inside container 
srun singularity run $SIF bash -c '
  echo "Copying training data to /tmp/";
  time cp -a ../resources/train_images.hdf5 /tmp/. ;
  echo "Running training";
  time python visiontransformer_ramfs.py  ;
  echo "Copying checkpoint back from /tmp/";
  time /bin/cp -a /tmp/vit_b_16_imagenet.pth ./vit_b_16_imagenet.pth.$$ ;
  echo "Copying training data from /tmp/ ";
  time /bin/cp -a /tmp/train_images.hdf5     ../resources/train_images.hdf5.$$'
