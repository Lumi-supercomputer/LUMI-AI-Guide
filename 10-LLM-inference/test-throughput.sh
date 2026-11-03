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
module load Local-LAIF lumi-aif-singularity-bindings

# set MIOPEN temp folder
export MIOPEN_CUSTOM_CACHE_DIR="/tmp/miopen-cache-${USER}"
export MIOPEN_USER_DB_PATH="/tmp/miopen-config-${USER}"
srun mkdir -p "$MIOPEN_CUSTOM_CACHE_DIR" "$MIOPEN_USER_DB_PATH"

# We use the PyTorch container provided by the LUMI AI Factory Services, which contains vLLM.
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20261028_090706/lumi-multitorch-full-u24r72f21m50t211-20261028_090706.sif

# Where to store the huge models. Point this to your project's scratch directory.
export HF_HOME=/scratch/$SLURM_JOB_ACCOUNT/hf-cache/

# Redirect all vLLM cache files from $HOME to scratch.
export VLLM_CACHE_ROOT=/scratch/$SLURM_JOB_ACCOUNT/vllm-cache

# Model selection
MODEL_NAME="Qwen/Qwen3.8-27B"

# Download the ShareGPT dataset to node-local /tmp, which is cleaned up when the job ends
DATASET=/tmp/sharegpt-${SLURM_JOB_ID}.json
curl -sSfL -o "$DATASET" https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# Run offline benchmark
# TRITON_ATTN is needed for hybrid models such as Qwen3.8: the default ROCm attention
# can't use its fast kernel with them and falls back to a slow implementation.
srun singularity run \
    $SIF \
    vllm bench throughput \
    --model $MODEL_NAME \
    --tensor-parallel-size $SLURM_GPUS_ON_NODE \
    --dataset-name sharegpt \
    --dataset-path $DATASET \
    --num-prompts 1000 \
    --load-format runai_streamer \
    --attention-backend TRITON_ATTN
