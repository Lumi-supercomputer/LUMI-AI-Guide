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

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

export SINGULARITYENV_PREPEND_PATH=/user-software/bin

srun singularity exec \
	-B ../resources/visiontransformer-env.sqsh:/user-software:image-src=/ \
	$CONTAINER bash -c 'export CXX=g++-12; python -m torch.distributed.run --nproc_per_node 8 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT ds_visiontransformer.py --deepspeed --deepspeed_config ds_config.json'
