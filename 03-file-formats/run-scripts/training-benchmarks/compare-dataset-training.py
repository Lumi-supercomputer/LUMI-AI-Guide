import sys
import argparse
from time import time

sys.path.append("scripts/lmdb")

sys.path.append("scripts/hdf5")

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from hdf5_dataset import HDF5Dataset

import torch
from torchvision.models import vit_b_16



# Define transformations
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]
)


def load_data(file_format: str, train: bool):
    if train:
        split = "train"
    else:
        split = "val"

    start = time()
    if file_format == "squashfs":
        sqsh_data = ImageFolder(f"/{split}_images", transform=transform)
        return sqsh_data
    elif file_format == "lmdb":
        # do this only here to not crash with other options
        from lmdb_dataset import LMDBDataset
        lmdb = f"data-formats/lmdb/{split}_images"
        return LMDBDataset(lmdb, transform=transform)
    elif file_format == "hdf5":
        hdf5 = f"data-formats/hdf5/{split}_images.hdf5"
        return HDF5Dataset(hdf5, transform=transform)


def train_model(model, criterion, optimizer, train_loader, val_loader, epochs, fileformat):
    # note that "cuda" is used as a general reference to GPUs,
    # even when running on AMD GPUs that use ROCm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    start = time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader)}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validation Accuracy: {100 * correct / total}%")

    print(f"{fileformat} time: {time()-start}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_workers",
                        help="Number of workers", default=1)
    parser.add_argument(
        "-ff",
        "--file_format",
        help="Which file format to benchmark",
        default="squashfs",
    )

    args = parser.parse_args()
    num_workers = int(args.num_workers)

    BATCH_SIZE = 32
    SHUFFLE = True
    EPOCHS = 4

    # squashfs does not use context manager
    if args.file_format == "squashfs":
        start_data_loading = time()
        train_data = load_data(file_format=args.file_format, train=True)
        val_data = load_data(file_format=args.file_format, train=False)
        print(f"Data loading took {time()-start_data_loading}.")
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=num_workers)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
        model = vit_b_16(weights="DEFAULT")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        print(f"Training for {EPOCHS} epochs in total.")
        train_model(model, criterion, optimizer, train_loader, val_loader, epochs=EPOCHS, fileformat=args.file_format)
    else:
        start_data_loading = time()
        with load_data(file_format=args.file_format, train=True) as train_data, load_data(file_format=args.file_format, train=False) as val_data:
            print(f"Data loading took {time()-start_data_loading}.")
            train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=num_workers)
            val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
            model = vit_b_16(weights="DEFAULT")
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            print(f"Training for {EPOCHS} epochs in total.")
            train_model(model, criterion, optimizer, train_loader, val_loader, epochs=EPOCHS, fileformat=args.file_format)


if __name__ == "__main__":
    main()