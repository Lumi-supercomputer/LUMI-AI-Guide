#!/bin/bash
#SBATCH --account=project_xxxxxxxxx
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --time=1:00:00

#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G

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

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT="1${SLURM_JOB_ID:0-4}" # set port based on SLURM_JOB_ID to avoid conflicts
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE

# Set up the CPU bind masks
CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

export SINGULARITYENV_PREPEND_PATH=/user-software/bin # gives access to packages inside the container

srun --cpu-bind=v,mask_cpu=$CPU_BIND_MASKS singularity run -B ../resources/ai-guide-env.sqsh:/user-software:image-src=/ $SIF bash -c "export RANK=\$SLURM_PROCID && export LOCAL_RANK=\$SLURM_LOCALID && python ddp_visiontransformer.py"
