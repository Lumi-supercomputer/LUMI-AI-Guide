#!/bin/bash
#SBATCH --account=project_xxxxxxxxx
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=1:00:00

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G

# this module facilitates the use of singularity containers on LUMI
module use  /appl/local/containers/ai-modules
module load singularity-AI-bindings

# choose container that is copied over by set_up_environment.sh
CONTAINER=../resources/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif

# Tell RCCL to use Slingshot interfaces and GPU RDMA
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

export SINGULARITYENV_PREPEND_PATH=/user-software/bin
srun singularity exec -B ../resources/visiontransformer-env.sqsh:/user-software:image-src=/ $CONTAINER bash -c 'python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 ddp_visiontransformer.py'
