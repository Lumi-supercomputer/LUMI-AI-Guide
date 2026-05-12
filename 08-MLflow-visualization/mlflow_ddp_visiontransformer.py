import torch
import os
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import psutil
import mlflow
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from resources.hdf5_dataset import HDF5Dataset

def train_model(model, criterion, optimizer, train_loader, val_loader, epochs, rank):
    # note that "cuda" is used as a general reference to GPUs,
    # even when running on AMD GPUs that use ROCm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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

        if rank == 0:
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
            mlflow.log_metric("loss", running_loss /
                              len(train_loader), step=epoch)

        # Validation step
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

        if rank == 0:
            print(f"Accuracy: {100 * correct / total}%")
            mlflow.log_metric("accuracy", correct / total, step=epoch)


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = int(os.environ["RANK"])

    if rank == 0:
        #    mlflow.set_tracking_uri(os.environ['PWD'] + "/mlflow")
        mlflow.set_tracking_uri(
            "sqlite:///" + os.environ["PWD"] + "/mlruns.db")
        mlflow.start_run(run_name="vision_transformer_test")

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

    model = vit_b_16(weights="DEFAULT").to(local_rank)
    model = DistributedDataParallel(model, device_ids=[local_rank])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

        train_model(model, criterion, optimizer, train_loader,
                    val_loader, epochs=10, rank=rank)

        dist.destroy_process_group()

    torch.save(model.state_dict(), "vit_b_16_imagenet.pth")
