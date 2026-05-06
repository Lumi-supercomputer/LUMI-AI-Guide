import sys, argparse
from time import time

sys.path.append("scripts/lmdb")
from lmdb_dataset import LMDBDataset

sys.path.append("scripts/hdf5")
from hdf5_dataset import HDF5Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np

# Define transformations
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def DataLoaderDifference(
    data, name, batch_size=32, shuffle=True, num_workers=7, N_sample=10000
):
    print("Executing DataLoaderDifference...")

    loader = DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    print(
        f"Dataloader with input: batch_size={batch_size}, shuffle={shuffle}, num_workers={num_workers}"
    )
    unique = set()
    print(f"Looping over N_sample={N_sample} iterations")

    t2 = time()
    i = 0
    for data in loader:
        batch_images, batch_labels = data
        for label in batch_labels:
            unique.add(int(label))
        i += 1
        if i == N_sample // batch_size:
            break

    print(f"{name} dataloader time: {time()-t2}")
    print(f"Number of unique labels: {len(unique)}")
    print(f"The first 10 labels are: {list(unique)[:10]}")


def loop_timing(data, name):
    t2 = time()
    for i in range(10000):
        image, label = data[i]
    print(f"{name} looping time: {time()-t2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_workers", help="Number of workers", default=1)
    parser.add_argument(
        "-N", "--N_sample", help="Number of samples to extract", default=10000
    )
    parser.add_argument(
        "-ff",
        "--file_format",
        help="Which file format to benchmark",
        default="squashfs",
    )

    args = parser.parse_args()
    N_sample = int(args.N_sample)
    num_workers = int(args.num_workers)
    if args.file_format == "squashfs":
        t1 = time()
        sqsh_data = ImageFolder("/train_images", transform=transform)
        print(f"Squash loading time: {time()-t1}")
        # loop_timing(sqsh_data, 'SquashFS')
        DataLoaderDifference(
            sqsh_data, "SquashFS", num_workers=num_workers, N_sample=N_sample
        )
    elif args.file_format == "lmdb":
        lmdb = "/project/project_462000002/joachimsode/file-format-ai-benchmark/LUMI-AI-example/data-formats/lmdb-test/data.mdb"
        t3 = time()
        with LMDBDataset(lmdb, transform=transform) as lmdb_data:
            print(f"LMDB loading time: {time()-t3}")
            # loop_timing(lmdb_data, 'LMDB')
            DataLoaderDifference(
                lmdb_data, "LMDB", num_workers=num_workers, N_sample=N_sample
            )
    elif args.file_format == "hdf5":
        hdf5 = "/project/project_462000002/joachimsode/file-format-ai-benchmark/LUMI-AI-example/data-formats/hdf5/train_images.hdf5"
        t3 = time()
        with HDF5Dataset(hdf5, transform=transform) as hdf5_data:
            print(f"HDF5 loading time: {time()-t3}")
            # loop_timing(lmdb_data, 'LMDB')
            DataLoaderDifference(
                hdf5_data, "HDF5", num_workers=num_workers, N_sample=N_sample
            )
