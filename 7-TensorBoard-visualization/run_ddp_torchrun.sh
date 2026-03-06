#!/bin/bash
#SBATCH --account=project_xxxxxxxxx
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=1:00:00

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G


# this module facilitates the use of LUMI AIF singularity containers on LUMI
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# set MIOPEN temp folder
MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR/cache
export MIOPEN_USER_DB=$MIOPEN_DIR/config

# choose container
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif

# Tell RCCL to use Slingshot interfaces and GPU RDMA
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

export SINGULARITYENV_PREPEND_PATH=/user-software/bin # gives access to packages inside the container

srun singularity run -B ../resources/ai-guide-env.sqsh:/user-software:image-src=/ $SIF bash -c 'python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 tensorboard_ddp_visiontransformer.py'