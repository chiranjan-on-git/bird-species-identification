
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import DataLoader

from src.data.dataset import CUBDataset
from src.models.openclip_encoder import load_openclip_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "openclip_features"
)

BATCH_SIZE = 4


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading OpenCLIP ViT-B/32...")

    model, preprocess, device = load_openclip_model()

    print("Device:", device)
    print("Batch size:", BATCH_SIZE)

    # Load only the official training split.
    dataset = CUBDataset(
        split="train",
        transform=preprocess,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    all_features = []
    all_labels = []
    all_image_ids = []

    model.eval()

    print("\nExtracting image features...\n")

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            images = batch["image"].to(device)

            features = model.encode_image(images)

            # Normalize each feature vector.
            features = features / features.norm(
                dim=-1,
                keepdim=True,
            )

            all_features.append(
                features.float().cpu().numpy()
            )

            all_labels.extend(
                batch["label"].numpy().tolist()
            )

            all_image_ids.extend(
                batch["image_id"].numpy().tolist()
            )

            if (batch_index + 1) % 100 == 0:
                processed = min(
                    (batch_index + 1) * BATCH_SIZE,
                    len(dataset),
                )

                print(
                    f"Processed {processed}/{len(dataset)} "
                    f"training images"
                )

    # Combine the extracted batches.
    features_array = np.concatenate(
        all_features,
        axis=0,
    )

    labels_array = np.asarray(
        all_labels,
        dtype=np.int64,
    )

    image_ids_array = np.asarray(
        all_image_ids,
        dtype=np.int64,
    )

    output_path = OUTPUT_DIR / "cub_train_features.npz"

    np.savez_compressed(
        output_path,
        features=features_array,
        labels=labels_array,
        image_ids=image_ids_array,
    )

    print("\nFEATURE EXTRACTION COMPLETE")
    print("---------------------------")
    print("Feature shape:", features_array.shape)
    print("Labels shape:", labels_array.shape)
    print("Image IDs shape:", image_ids_array.shape)
    print("Feature data type:", features_array.dtype)
    print("Saved to:", output_path)

    # Basic integrity checks.
    assert features_array.shape == (5994, 512)
    assert labels_array.shape == (5994,)
    assert image_ids_array.shape == (5994,)
    assert np.isfinite(features_array).all()

    feature_norms = np.linalg.norm(
        features_array,
        axis=1,
    )

    print(
        "Average feature norm:",
        round(float(feature_norms.mean()), 4),
    )

    print("\nOPENCLIP FEATURE EXTRACTION PASSED")


if __name__ == "__main__":
    main()
