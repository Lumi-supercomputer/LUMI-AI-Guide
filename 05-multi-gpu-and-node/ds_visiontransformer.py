import torch
import os
import time
import argparse
import deepspeed
import psutil
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from resources.hdf5_dataset import HDF5Dataset


def train_model(args, model, criterion, optimizer, train_loader, val_loader, epochs, rank):
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=model.parameters()
    )

    if rank == 0:
        start = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(model_engine.local_rank), labels.to(
                model_engine.local_rank
            )
            optimizer.zero_grad()

            outputs = model_engine(images)
            loss = criterion(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()
            running_loss += loss.item()

        if rank == 0:
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

        # Validation step, note that only results from rank 0 are used here.
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(model_engine.local_rank), labels.to(
                    model_engine.local_rank
                )
                outputs = model_engine(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if rank == 0:
            print(f"Accuracy: {100 * correct / total}%")

    if rank == 0:
        print(f"Time elapsed (s): {time.time()-start}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = int(os.environ["RANK"])

    # Define transformations
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model = vit_b_16(weights="DEFAULT")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    deepspeed.init_distributed()

    with HDF5Dataset(
        "../resources/train_images.hdf5", transform=transform
    ) as full_train_dataset:

        # Splitting the dataset into train and validation sets
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_train_dataset, [train_size, val_size]
        )

        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=32, num_workers=7
        )

        val_sampler = DistributedSampler(val_dataset)
        val_loader = DataLoader(
            val_dataset, sampler=val_sampler, batch_size=32, num_workers=7
        )

        train_model(args, model, criterion, optimizer, train_loader, val_loader, epochs=10, rank=rank)

    torch.save(model.state_dict(), "vit_b_16_imagenet.pth")
