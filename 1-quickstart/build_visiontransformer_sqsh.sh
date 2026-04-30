#!/bin/bash
set -euo pipefail

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

source ../env.sh

: "${CONTAINER:?Set CONTAINER in env.sh}"

OUT_SQSH="../resources/visiontransformer-env.sqsh"
BUILD_DIR="$SCRATCH_ROOT/visiontransformer-env-build"

mkdir -p ../resources
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo "Building extension environment in: $BUILD_DIR"
singularity exec -B "$BUILD_DIR":/user-software "$CONTAINER" bash -c '
set -euo pipefail
python -m venv /user-software --system-site-packages
/user-software/bin/python -m pip install h5py lmdb msgpack six tqdm mlflow==2.22.0
'

rm -f "$OUT_SQSH"
echo "Creating squashfs image: $OUT_SQSH"
mksquashfs "$BUILD_DIR" "$OUT_SQSH" -noappend -comp gzip -no-xattrs

echo "Done: $OUT_SQSH"
