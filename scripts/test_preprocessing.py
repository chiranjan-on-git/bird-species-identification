
import torch

from src.data.dataset import CUBDataset
from src.data.preprocessing import (
    get_train_transforms,
    get_eval_transforms,
)

train_dataset = CUBDataset(
    split="train",
    transform=get_train_transforms(),
)

test_dataset = CUBDataset(
    split="test",
    transform=get_eval_transforms(),
)

train_sample = train_dataset[0]
test_sample = test_dataset[0]

print("TRAIN IMAGE")
print("Tensor shape:", train_sample["image"].shape)
print("Tensor dtype:", train_sample["image"].dtype)
print("Species:", train_sample["species"])

print("\nTEST IMAGE")
print("Tensor shape:", test_sample["image"].shape)
print("Tensor dtype:", test_sample["image"].dtype)
print("Species:", test_sample["species"])

assert train_sample["image"].shape == (3, 224, 224)
assert test_sample["image"].shape == (3, 224, 224)
assert train_sample["image"].dtype == torch.float32
assert test_sample["image"].dtype == torch.float32

print("\nPREPROCESSING TEST PASSED")
