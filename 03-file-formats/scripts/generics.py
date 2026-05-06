import io
from torchvision.datasets.folder import has_file_allowed_extension, IMG_EXTENSIONS
from torchvision import get_image_backend
import zipfile
import os
from typing import Any, cast
from collections.abc import Callable
from pathlib import Path
from torchvision.datasets import DatasetFolder
from PIL import Image


# https://github.com/ain-soph/trojanzoo/blob/9cbd31c99d674c1f3e401321a23e78438fb8d222/trojanvision/utils/dataset.py#L54
# https://github.com/koenvandesande/vision/blob/read_zipped_data/torchvision/datasets/zippedfolder.py
class ZipFolder(DatasetFolder):
    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        is_valid_file: Callable[[str], bool] | None = None,
        memory: bool = True,
    ) -> None:
        if not root.lower().endswith(".zip"):
            raise TypeError("Need zip file for data source: ", root)
        if memory:
            with open(root, "rb") as zf:
                data = zf.read()
            self.root_data = zipfile.ZipFile(io.BytesIO(data), "r")
        else:
            self.root_data = zipfile.ZipFile(root, "r")
        super().__init__(
            root=root,
            loader=self.raw_loader,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples

    @staticmethod
    def initialize_from_folder(root: str, zip_path: str = None):
        root = os.path.normpath(root)
        folder_dir, folder_base = os.path.split(root)
        if zip_path is None:
            zip_path = os.path.join(folder_dir, f"{folder_base}_store.zip")
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_STORED) as zf:
            for walk_root, walk_dirs, walk_files in os.walk(root):
                zip_root = walk_root.removeprefix(folder_dir)
                for _file in walk_files:
                    org_path = os.path.join(walk_root, _file)
                    dst_path = os.path.join(zip_root, _file)
                    zf.write(org_path, dst_path)

    def make_dataset(
        self,
        directory: str,
        class_to_idx: dict[str, int],
        extensions: None | tuple[str, ...] = None,
        is_valid_file: None | Callable[[str], bool] = None,
        allow_empty: bool = False,
    ) -> list[tuple[str, int]]:
        instances = []
        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError(
                "Both extensions and is_valid_file cannot be None or not None at the same time"
            )
        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                if len(Path(x).parts) > 3:
                    # Assume Path.parts = ('ILSVRC', 'Data', 'CLS-LOC', 'train', 'n02493793', 'n02493793_2317.JPEG')
                    # train_data = Path(x).parts[1] == 'Data' and Path(x).parts[3] == 'train'
                    # Assume Path.parts = ('tiny-imagenet-200', 'train', 'n03584254', 'images', 'n03584254_229.JPEG')
                    train_data = Path(x).parts[1] == "train"
                else:
                    train_data = False
                return (
                    has_file_allowed_extension(x, cast(tuple[str, ...], extensions))
                    and train_data
                )

        is_valid_file = cast(Callable[[str], bool], is_valid_file)
        for filepath in self.root_data.namelist():
            if is_valid_file(filepath):
                # target_class = Path(filepath).parts[4]
                target_class = Path(filepath).parts[2]
                instances.append((filepath, class_to_idx[target_class]))
        return instances

    def zip_loader(self, path: str) -> Image.Image:
        f = self.root_data.open(path, "r")
        if get_image_backend() == "accimage":
            try:
                import accimage  # type: ignore

                return accimage.Image(f)
            except IOError:
                pass  # fall through to PIL
        return Image.open(f).convert("RGB")

    def raw_loader(self, path: str) -> bytes:
        data = self.root_data.read(path)
        bin_data = io.BytesIO(data)
        return bin_data

    def find_classes(self, *args, **kwargs) -> tuple[list[str], dict[str, int]]:
        r"""Finds the class folders in a dataset.

        Args:
            dir (str): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        class_set = set()
        for filepath in self.root_data.namelist():
            root, target_class = os.path.split(os.path.dirname(filepath))
            if root:
                class_set.add(target_class)
        classes = list(class_set)
        classes.sort()  # TODO: Pylance issue
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
