#!/bin/bash
#SBATCH --account=project_xxxxxxxxx
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%x-%j.out
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

# Set your TORCH_HOME cache to scratch to avoid saving to home directory
# https://docs.pytorch.org/docs/2.11/hub.html#where-are-my-downloaded-models-saved
export TORCH_HOME="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/torch_home"
mkdir -p "$TORCH_HOME"

# choose container
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT="1${SLURM_JOB_ID:0-4}" # set port based on SLURM_JOB_ID to avoid conflicts

srun singularity run \
	$SIF bash -c 'python -m torch.distributed.run --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=8 --node_rank $SLURM_PROCID --rdzv_id=\$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" ds_visiontransformer.py --deepspeed --deepspeed_config ds_config.json'
