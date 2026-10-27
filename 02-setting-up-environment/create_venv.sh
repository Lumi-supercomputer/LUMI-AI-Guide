#!/bin/bash
module purge
module load Local-LAIF lumi-aif-singularity-bindings
export BUILD_DIR=optuna-env
export SQUASHFS_NAME=optuna-env.sqsh
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20260929_104918/lumi-multitorch-full-u24r72f21m50t211-20260929_104918.sif

# creating the venv
echo "Creating venv in: $BUILD_DIR"
mkdir $BUILD_DIR
singularity exec "$SIF" bash -c '
set -euo pipefail
python -m venv $BUILD_DIR --system-site-packages
source $BUILD_DIR/bin/activate
pip install optuna
'