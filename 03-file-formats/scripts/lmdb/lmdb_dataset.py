import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_pil_image
import lmdb
import pickle
import os, six
from PIL import Image


class LMDBDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform

    def __enter__(self):
        self.file = lmdb.open(
            self.file_path,
            subdir=os.path.isdir(self.file_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with self.file.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b"__len__"))
            self.keys = pickle.loads(txn.get(b"__keys__"))
        return self

    def __exit__(self, type, value, traceback):
        self.file.close()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.file.begin(write=False) as txn:
            byteflow = txn.get(self.keys[idx])
        byte_image, label = pickle.loads(byteflow)

        image = Image.open(byte_image).convert("RGB")
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label
