#!/bin/bash

mkdir -p data-formats/zip
cd data-formats/zip
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
cd ../..

mkdir -p data-formats/raw
unzip -q data-formats/zip/tiny-imagenet-200.zip -d data-formats/raw

