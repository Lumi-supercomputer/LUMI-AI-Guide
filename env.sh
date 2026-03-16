#!/bin/bash

# Shared configuration for LUMI runs across the repo.
# Override any value by exporting it before sourcing this file.

export REPO_ROOT="/project/project_462000131/anisrahm/LUMI-AI-Guide"

export PROJECT_ACCOUNT="${PROJECT_ACCOUNT:-project_462000131}"
export LUMI_USER="${LUMI_USER:-${USER:-anisrahm}}"
export CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif}"

export PROJECT_ROOT="${PROJECT_ROOT:-/project/${PROJECT_ACCOUNT}/${LUMI_USER}}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${PROJECT_ACCOUNT}/${LUMI_USER}}"

export DATA_PROJECT_DIR="${DATA_PROJECT_DIR:-${SCRATCH_ROOT}/file-format-ai-benchmark}"
export DATA_BENCH_DIR="${DATA_BENCH_DIR:-${SCRATCH_ROOT}/file-format-ai-benchmark}"

export SQUASH_LARGE="${SQUASH_LARGE:-${DATA_BENCH_DIR}/imagenet-object-localization-challenge.squashfs}"
export LMDB_LARGE="${LMDB_LARGE:-${DATA_BENCH_DIR}/imagenet-object-localization-challenge.lmdb}"

export TINY_LMDB_PATH="${TINY_LMDB_PATH:-${DATA_PROJECT_DIR}/data-formats/lmdb/train_images}"
export TINY_HDF5_PATH="${TINY_HDF5_PATH:-${DATA_PROJECT_DIR}/data-formats/hdf5/train_images.hdf5}"

export IMAGENET_ZIP_DIR="${IMAGENET_ZIP_DIR:-${DATA_BENCH_DIR}/}"
