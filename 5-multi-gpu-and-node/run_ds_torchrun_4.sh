#!/bin/bash
#SBATCH --account=project_xxxxxxxxx
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --time=1:00:00

#SBATCH --tasks-per-node=1
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

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT="1${SLURM_JOB_ID:0-4}" # set port based on SLURM_JOB_ID to avoid conflicts

export SINGULARITYENV_PREPEND_PATH=/user-software/bin # gives access to packages inside the container

srun singularity run -B ../resources/ai-guide-env.sqsh:/user-software:image-src=/ \
	$SIF bash -c 'python -m torch.distributed.run --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=8 --node_rank $SLURM_PROCID --rdzv_id=\$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" ds_visiontransformer.py --deepspeed --deepspeed_config ds_config.json'
