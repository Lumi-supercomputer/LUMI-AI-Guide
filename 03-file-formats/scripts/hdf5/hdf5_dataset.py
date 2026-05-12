import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image
import h5py


class HDF5Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform

    def __enter__(self):
        self.file = h5py.File(self.file_path, "r")
        self.images = self.file["images"]
        self.labels = self.file["labels"]
        return self

    def __exit__(self, type, value, traceback):
        self.file.close()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx])
        image = to_pil_image(image)  # Convert tensor to PIL Image
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label
