#!/bin/bash
#SBATCH --account=project_xxxxxxxxx
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=2
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:30:00


# Load the bindings to give LUMI containers access to the file system of the working directory
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# Set MIOPEN temp folder
MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR/cache
export MIOPEN_USER_DB=$MIOPEN_DIR/config

# We use the PyTorch container provided by the LUMI AI Factory Services, which contains vLLM.
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif


# Where to store the huge models. Point this to your project's scratch directory.
export HF_HOME=/scratch/$SLURM_JOB_ACCOUNT/hf-cache/

# Redirect all vLLM cache files from $HOME to scratch.
export VLLM_CACHE_ROOT=/scratch/$SLURM_JOB_ACCOUNT/vllm-cache

# Model selection
MODEL_NAME="Qwen/Qwen3.6-35B-A3B"


# Run offline benchmark 
srun singularity run \
    $SIF \
    vllm bench throughput \
    --model $MODEL_NAME \
    --tensor-parallel-size $SLURM_GPUS_ON_NODE \
    --dataset-name sharegpt \
    --num-prompts 1000 \
    --load-format runai_streamer
