#!/bin/bash

#SBATCH --job-name=quickstart-vit
#SBATCH --account=project_462000131
#SBATCH --partition=small-g

#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G

#SBATCH --time=01:00:00

set -euo pipefail

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

source ../env.sh

: "${CONTAINER:?Set CONTAINER in env.sh}"
singularity exec -B ../resources/visiontransformer-env.sqsh:/user-software:image-src=/ "$CONTAINER" /user-software/bin/python visiontransformer.py
