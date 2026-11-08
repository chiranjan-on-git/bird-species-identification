
import numpy as np
from pathlib import Path

from src.data.dataset import CUBDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "openclip_features"
    / "cub_test_features.npz"
)


def main():

    print("Loading saved test features...")

    saved = np.load(FEATURE_PATH)

    features = saved["features"]
    labels = saved["labels"]
    image_ids = saved["image_ids"]

    dataset = CUBDataset(split="test")

    expected_ids = np.array(dataset.image_ids)

    expected_labels = np.array([
        int(dataset.labels[image_id]) - 1
        for image_id in dataset.image_ids
    ])

    print("\nTEST FEATURE VALIDATION")
    print("-" * 45)

    print(f"Feature shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image IDs shape: {image_ids.shape}")
    print(f"Data type: {features.dtype}")

    assert features.shape == (5794, 512)
    assert labels.shape == (5794,)
    assert image_ids.shape == (5794,)

    assert np.isfinite(features).all()
    assert np.isfinite(labels).all()

    assert len(np.unique(image_ids)) == 5794

    assert labels.min() == 0
    assert labels.max() == 199

    assert np.array_equal(image_ids, expected_ids)
    assert np.array_equal(labels, expected_labels)

    norms = np.linalg.norm(features, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-4)

    print("\nINTEGRITY CHECKS")
    print("-" * 45)
    print("PASS: Correct feature dimensions")
    print("PASS: No non-finite values")
    print("PASS: No duplicate image IDs")
    print("PASS: Labels range from 0 to 199")
    print("PASS: Official test image IDs preserved")
    print("PASS: Original labels correctly aligned")
    print("PASS: Feature vectors normalized")

    print("\nOFFICIAL TEST FEATURE VALIDATION SUCCESSFUL")


if __name__ == "__main__":
    main()
