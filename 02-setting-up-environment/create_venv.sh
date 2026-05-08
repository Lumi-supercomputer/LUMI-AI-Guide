#!/bin/bash
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
export BUILD_DIR=optuna-env
export SQUASHFS_NAME=optuna-env.sqsh
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif

# creating the venv
echo "Creating venv in: $BUILD_DIR"
mkdir $BUILD_DIR
singularity exec "$SIF" bash -c '
set -euo pipefail
python -m venv $BUILD_DIR --system-site-packages
source $BUILD_DIR/bin/activate
pip install optuna
'