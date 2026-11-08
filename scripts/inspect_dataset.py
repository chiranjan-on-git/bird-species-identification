
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def find_dataset_root():
    """Find the directory containing the CUB annotation files."""

    candidates = [
        DATA_DIR / "CUB_200_2011",
        *DATA_DIR.glob("**/CUB_200_2011"),
    ]

    for candidate in candidates:
        if (candidate / "images.txt").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find images.txt inside data/raw. "
        "Check the extracted dataset directory."
    )


def read_id_mapping(file_path):
    """Read files containing: ID VALUE."""

    mapping = {}

    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(maxsplit=1)

            if len(parts) == 2:
                mapping[int(parts[0])] = parts[1]

    return mapping


def main():
    root = find_dataset_root()

    print("=" * 55)
    print("CUB-200-2011 DATASET INSPECTION")
    print("=" * 55)

    print(f"\nDataset root: {root}")

    # Read annotation files
    images = read_id_mapping(root / "images.txt")
    labels = read_id_mapping(root / "image_class_labels.txt")
    classes = read_id_mapping(root / "classes.txt")
    splits = read_id_mapping(root / "train_test_split.txt")

    # Basic statistics
    print("\n--- Dataset Statistics ---")
    print(f"Total image records: {len(images)}")
    print(f"Total species: {len(classes)}")
    print(f"Label records: {len(labels)}")
    print(f"Split records: {len(splits)}")

    # Split counts
    train_ids = {
        image_id for image_id, split in splits.items()
        if split == "1"
    }

    test_ids = {
        image_id for image_id, split in splits.items()
        if split == "0"
    }

    print("\n--- Official Split ---")
    print(f"Training images: {len(train_ids)}")
    print(f"Testing images: {len(test_ids)}")

    # Class distribution
    class_counts = Counter(
        int(labels[image_id])
        for image_id in images
        if image_id in labels
    )

    print("\n--- Class Distribution ---")
    print(f"Classes represented: {len(class_counts)}")
    print(f"Minimum images per class: {min(class_counts.values())}")
    print(f"Maximum images per class: {max(class_counts.values())}")

    # Validate annotation IDs
    image_ids = set(images)
    label_ids = set(labels)
    split_ids = set(splits)

    print("\n--- Annotation Validation ---")
    print("Image/label IDs match:", image_ids == label_ids)
    print("Image/split IDs match:", image_ids == split_ids)

    # Validate image files
    missing_images = []

    for image_id, relative_path in images.items():
        image_path = root / "images" / relative_path

        if not image_path.is_file():
            missing_images.append(relative_path)

    print(f"Missing image files: {len(missing_images)}")

    if missing_images:
        print("First missing files:")
        for path in missing_images[:10]:
            print("  ", path)

    # Species examples
    print("\n--- First 10 Species ---")

    for class_id in sorted(classes, key=int)[:10]:
        print(f"{class_id}: {classes[class_id]}")

    # Final validation
    print("\n--- Expected Benchmark Checks ---")

    checks = {
        "11,788 images": len(images) == 11788,
        "200 species": len(classes) == 200,
        "5,994 training images": len(train_ids) == 5994,
        "5,794 testing images": len(test_ids) == 5794,
        "No missing images": len(missing_images) == 0,
        "Matching annotation IDs": (
            image_ids == label_ids == split_ids
        ),
    }

    for description, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {description}")

    if all(checks.values()):
        print("\nDATASET VALIDATION SUCCESSFUL")
    else:
        print("\nDATASET VALIDATION NEEDS INVESTIGATION")


if __name__ == "__main__":
    main()
