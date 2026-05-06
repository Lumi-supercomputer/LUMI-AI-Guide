#!/bin/bash
module use /appl/local/training/modules/AI-20240529
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif

if [ -d "venv-extension" ]; then echo 'Removing existing venv-extension'; rm -Rf venv-extension; fi

singularity exec $CONTAINER bash -c '$WITH_CONDA && python -m venv venv-extension --system-site-packages && source venv-extension/bin/activate && python -m pip install -r venv-requirements.txt'


