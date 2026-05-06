import os
import sys
import six
import string
import argparse

import lmdb
import pickle
import msgpack
import tqdm
from PIL import Image

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets

# Code adopted from https://github.com/rmccorm4/PyTorch-LMDB which is adopted from https://github.com/Lyken17/Efficient-PyTorch


def raw_reader(path):
    with open(path, "rb") as f:
        bin_data = f.read()
    return bin_data


def dumps(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pickle.dumps(obj, protocol=5)


def folder2lmdb(image_folder, output_file, write_frequency=100):
    directory = os.path.expanduser(image_folder)
    print("Loading dataset from %s" % directory)
    dataset = ImageFolder(directory, loader=raw_reader)
    data_loader = DataLoader(dataset, num_workers=16, collate_fn=lambda x: x)

    lmdb_path = os.path.expanduser(output_file)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(
        lmdb_path,
        subdir=isdir,
        map_size=1099511627776 * 2,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    txn = db.begin(write=True)
    for idx, data in enumerate(data_loader):
        image, label = data[0]
        txn.put("{}".format(idx).encode("ascii"), dumps((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = ["{}".format(k).encode("ascii") for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b"__keys__", dumps(keys))
        txn.put(b"__len__", dumps(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


def main():
    folder_in = "data-formats/raw/tiny-imagenet-200/"
    folder_out = "data-formats/lmdb/"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        help="Path to original image dataset folder",
        default=folder_in,
    )
    parser.add_argument(
        "-o", "--output_folder", help="Path to output LMDB file", default=folder_out
    )
    args = parser.parse_args()

    folder2lmdb(args.input_folder + "val", args.output_folder + "val_images")
    folder2lmdb(args.input_folder + "train", args.output_folder + "train_images")


if __name__ == "__main__":
    main()
