import os, sys
import time
from PIL import Image
import h5py
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from pathlib import Path


def create_hdf5(image_folder, output_file):
    dataset = ImageFolder(image_folder, transform=transforms.ToTensor())
    num_images = len(dataset)

    with h5py.File(output_file, "w") as h5f:
        # Create datasets for images and labels
        images = h5f.create_dataset("images", (num_images, 3, 64, 64), dtype="f")
        labels = h5f.create_dataset("labels", (num_images,), dtype="i")

        for i, (img, label) in enumerate(dataset):
            images[i] = img.numpy()
            labels[i] = label
            if i % 100 == 0:
                print(f"Processed {i} images")


def main():
    folder_in = "data-formats/raw/tiny-imagenet-200/"
    folder_out = "data-formats/hdf5/"

    create_hdf5(folder_in + "train", folder_out + "train_images.hdf5")
    create_hdf5(folder_in + "val", folder_out + "val_images.hdf5")


if __name__ == "__main__":
    main()
