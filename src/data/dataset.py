
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CUB_ROOT = PROJECT_ROOT / "data" / "raw" / "CUB_200_2011"


def read_mapping(file_path):
    """Read CUB annotation files containing: ID value."""
    mapping = {}

    with open(file_path, "r") as file:
        for line in file:
            image_id, value = line.strip().split(maxsplit=1)
            mapping[int(image_id)] = value

    return mapping


class CUBDataset(Dataset):
    """PyTorch Dataset for the CUB-200-2011 bird dataset."""

    def __init__(self, split="train", transform=None):
        if split not in ("train", "test"):
            raise ValueError("split must be either 'train' or 'test'")

        self.transform = transform

        self.image_paths = read_mapping(CUB_ROOT / "images.txt")
        self.labels = read_mapping(CUB_ROOT / "image_class_labels.txt")
        self.splits = read_mapping(CUB_ROOT / "train_test_split.txt")
        self.classes = read_mapping(CUB_ROOT / "classes.txt")

        # Preserve the official CUB train/test split.
        split_value = "1" if split == "train" else "0"

        self.image_ids = [
            image_id
            for image_id in sorted(self.image_paths)
            if self.splits[image_id] == split_value
        ]

        self.class_names = {
            class_id - 1: name
            for class_id, name in self.classes.items()
        }

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        relative_path = self.image_paths[image_id]
        image_path = CUB_ROOT / "images" / relative_path

        # Load one image at a time and convert to RGB.
        image = Image.open(image_path).convert("RGB")

        # CUB labels start at 1; PyTorch classification labels start at 0.
        label = int(self.labels[image_id]) - 1

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "species": self.class_names[label],
            "image_id": image_id,
        }
