#!/bin/bash
#SBATCH --account=project_462000002
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --time=1:00:00

#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G

# this module facilitates the use of singularity containers on LUMI
module use  /appl/local/containers/ai-modules
module load singularity-AI-bindings

# choose container that is copied over by set_up_environment.sh
CONTAINER=../resources/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif

# Tell RCCL to use Slingshot interfaces and GPU RDMA
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE

# Set up the CPU bind masks
CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

export SINGULARITYENV_PREPEND_PATH=/user-software/bin
srun --cpu-bind=v,mask_cpu=$CPU_BIND_MASKS singularity exec \
 	-B ../resources/visualtransformer-env.sqsh:/user-software:image-src=/ \
        -B ../resources/deepspeed_adam:/user-software/lib/python3.12/site-packages/deepspeed/ops/csrc/adam \
        -B ../resources/deepspeed_includes:/user-software/lib/python3.12/site-packages/deepspeed/ops/csrc/includes \
      	$CONTAINER bash -c 'export CXX=g++-12; export RANK=$SLURM_PROCID; export LOCAL_RANK=$SLURM_LOCALID; python ds_visualtransformer.py --deepspeed --deepspeed_config ds_config.json'
