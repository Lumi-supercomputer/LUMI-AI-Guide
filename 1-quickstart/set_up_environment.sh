#!/bin/bash

# this module facilitates the use of singularity containers on LUMI
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

# Please have a look at the terms of access (https://www.image-net.org/download.php) before using the dataset
echo "Copying container, training data and squashfs file to ../resources/ directory."
cp /appl/local/training/LUMI-AI-Guide/tiny-imagenet-dataset.hdf5 ../resources/train_images.hdf5
cp /appl/local/training/LUMI-AI-Guide/visualtransformer-env.sqsh ../resources/visiontransformer-env.sqsh 
cp /appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif ../resources/

# For the deepspeed examples, we need to copy the following two directories to ../resources/ 
cp -r /appl/local/training/LUMI-AI-Guide/deepspeed_adam ../resources/
cp -r /appl/local/training/LUMI-AI-Guide/deepspeed_includes ../resources/
