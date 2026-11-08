
import numpy as np

from pathlib import Path

from src.data.dataset import CUBDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "openclip_features"
    / "cub_train_features.npz"
)


def main():
    print("Loading saved OpenCLIP features...")

    saved = np.load(FEATURE_PATH)

    features = saved["features"]
    labels = saved["labels"]
    image_ids = saved["image_ids"]

    # Reconstruct the official training metadata.
    dataset = CUBDataset(split="train")

    expected_ids = np.array(dataset.image_ids)

    expected_labels = np.array([
        int(dataset.labels[image_id]) - 1
        for image_id in dataset.image_ids
    ])

    print("\nFEATURE FILE VALIDATION")
    print("-----------------------")
    print("Feature shape:", features.shape)
    print("Labels shape:", labels.shape)
    print("Image IDs shape:", image_ids.shape)
    print("Data type:", features.dtype)

    # Validate dimensions.
    assert features.shape == (5994, 512)
    assert labels.shape == (5994,)
    assert image_ids.shape == (5994,)

    # Validate numerical values.
    assert np.isfinite(features).all()
    assert np.isfinite(labels).all()

    # Validate unique image IDs.
    assert len(np.unique(image_ids)) == 5994

    # Validate class range.
    assert labels.min() == 0
    assert labels.max() == 199

    # Validate official image IDs and label alignment.
    assert np.array_equal(image_ids, expected_ids)
    assert np.array_equal(labels, expected_labels)

    # Validate normalized feature vectors.
    norms = np.linalg.norm(features, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-4)

    print("\nINTEGRITY CHECKS")
    print("----------------")
    print("PASS: Correct feature dimensions")
    print("PASS: No non-finite feature values")
    print("PASS: No duplicate image IDs")
    print("PASS: Labels range from 0 to 199")
    print("PASS: Original training image IDs preserved")
    print("PASS: Original labels correctly aligned")
    print("PASS: Feature vectors normalized")

    print("\nOPENCLIP FEATURE VALIDATION SUCCESSFUL")


if __name__ == "__main__":
    main()
