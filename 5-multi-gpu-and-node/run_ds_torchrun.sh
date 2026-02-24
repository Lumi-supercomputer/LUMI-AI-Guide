#!/bin/bash
#SBATCH --account=project_462000131
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

# choose container
SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

srun singularity run \
	$SIF bash -c 'export CXX=g++-12; source /scratch/project_462000131/marlonto/LUMI-AI-Guide/resources/ai-guide-env/bin/activate && python -m torch.distributed.run --nproc_per_node 8 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT ds_visiontransformer.py --deepspeed --deepspeed_config ds_config.json'