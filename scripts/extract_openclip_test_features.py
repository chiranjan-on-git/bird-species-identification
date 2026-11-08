
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import CUBDataset
from src.models.openclip_encoder import load_openclip_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "openclip_features"
    / "cub_test_features.npz"
)

BATCH_SIZE = 4


def main():

    print("Loading OpenCLIP encoder...")

    model, preprocess, device = load_openclip_model()

    print(f"Device: {device}")

    print("\nLoading official CUB test dataset...")

    dataset = CUBDataset(
        split="test",
        transform=preprocess
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    all_features = []
    all_labels = []
    all_image_ids = []

    print(f"Test images: {len(dataset)}")
    print("\nExtracting image features...")

    model.eval()

    with torch.inference_mode():

        for batch_index, batch in enumerate(loader):

            images = batch["image"].to(device)

            features = model.encode_image(images)

            features = features / features.norm(
                dim=-1,
                keepdim=True
            )

            all_features.append(
                features.cpu().numpy()
            )

            all_labels.append(
                batch["label"].numpy()
            )

            all_image_ids.append(
                batch["image_id"].numpy()
            )

            processed = min(
                (batch_index + 1) * BATCH_SIZE,
                len(dataset)
            )

            if processed % 400 == 0 or processed == len(dataset):
                print(
                    f"Processed {processed}/{len(dataset)} images"
                )

    features = np.concatenate(all_features)
    labels = np.concatenate(all_labels)
    image_ids = np.concatenate(all_image_ids)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_PATH,
        features=features,
        labels=labels,
        image_ids=image_ids
    )

    print("\nTEST FEATURE EXTRACTION COMPLETE")
    print("-" * 45)
    print(f"Feature shape: {features.shape}")
    print(f"Label shape: {labels.shape}")
    print(f"Image ID shape: {image_ids.shape}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
