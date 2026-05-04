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

# Set your huggingface cache to scratch to avoid saving to home directory
# See https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables
export HF_HUB_CACHE="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hfcache"
mkdir -p "$HF_HUB_CACHE"

# choose container
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif

srun singularity run $SIF bash -c 'python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 mlflow_ddp_visiontransformer.py'