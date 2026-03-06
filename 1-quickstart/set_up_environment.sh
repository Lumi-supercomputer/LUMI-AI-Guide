#!/bin/bash

# Please have a look at the terms of access (https://www.image-net.org/download.php) before using the dataset
echo "Copying training data to ../resources/ directory."
cp /appl/local/training/LUMI-AI-Guide/tiny-imagenet-dataset.hdf5 ../resources/train_images.hdf5
echo "Copying sqsh virtual environment to ../resources/ directory."
cp /appl/local/training/LUMI-AI-Guide/ai-guide-env.sqsh ../resources/ai-guide-env.sqsh