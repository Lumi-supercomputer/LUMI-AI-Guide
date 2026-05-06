from torch.utils.data import DataLoader, random_split
from hdf5_dataset import HDF5Dataset
import torch
import torchvision.transforms as transforms
from torch.profiler import profile, record_function, ProfilerActivity
from torchvision.models import vit_b_16
import sys

sys.path.append("scripts/")
from generics import time

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


def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prof = None
    for epoch in range(epochs):
        if epoch == 3:
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            prof.start()

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
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
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

        print(f"HDF5 Accuracy: {100 * correct / total}%")
        if epoch == 3:
            prof.stop()
            prof.export_chrome_trace("hdf5-trace.json")


@time("visualtransformer-hdf5")
def main():
    folder = "data-formats/hdf5/"

    # folder = 'data-formats/'
    with HDF5Dataset(folder + "images.hdf5", transform=transform) as full_train_dataset:
        # Splitting the dataset into train and validation sets
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_train_dataset, [train_size, val_size]
        )
        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=7
        )
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=7)
        train_model(model, criterion, optimizer, train_loader, val_loader)
