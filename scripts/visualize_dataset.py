
import matplotlib.pyplot as plt
import random

from src.data.dataset import CUBDataset


dataset = CUBDataset(split="train")

# Select nine random samples.
random.seed(42)
indices = random.sample(range(len(dataset)), 9)

fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for ax, index in zip(axes.flat, indices):
    sample = dataset[index]

    ax.imshow(sample["image"])
    ax.set_title(
        sample["species"].replace("_", " "),
        fontsize=9
    )
    ax.axis("off")

plt.tight_layout()
plt.savefig("data/processed/dataset_samples.png", dpi=150)
plt.show()

print("Saved visualization to data/processed/dataset_samples.png")
