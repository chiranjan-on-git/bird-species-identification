
from src.data.dataset import CUBDataset

train_dataset = CUBDataset(split="train")
test_dataset = CUBDataset(split="test")

print("TRAIN DATASET")
print("Number of images:", len(train_dataset))

print("\nTEST DATASET")
print("Number of images:", len(test_dataset))

# Load one training example.
sample = train_dataset[0]

print("\nSAMPLE IMAGE")
print("Image type:", type(sample["image"]))
print("Image dimensions:", sample["image"].size)
print("Species:", sample["species"])
print("Numerical label:", sample["label"])
print("Image ID:", sample["image_id"])

# Verify label and image dimensions.
assert len(train_dataset) == 5994
assert len(test_dataset) == 5794
assert sample["image"].mode == "RGB"
assert 0 <= sample["label"] < 200

print("\nDATASET LOADER TEST PASSED")
