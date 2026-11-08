
import csv
from pathlib import Path

from src.data.dataset import (
    CUB_ROOT,
    read_mapping,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cub_manifest.csv"


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    image_paths = read_mapping(CUB_ROOT / "images.txt")
    labels = read_mapping(CUB_ROOT / "image_class_labels.txt")
    splits = read_mapping(CUB_ROOT / "train_test_split.txt")
    classes = read_mapping(CUB_ROOT / "classes.txt")

    rows = []

    for image_id in sorted(image_paths):
        class_id = int(labels[image_id]) - 1
        split = "train" if splits[image_id] == "1" else "test"

        rows.append({
            "image_id": image_id,
            "relative_path": image_paths[image_id],
            "class_id": class_id,
            "species": classes[class_id + 1],
            "split": split,
        })

    fieldnames = [
        "image_id",
        "relative_path",
        "class_id",
        "species",
        "split",
    ]

    with open(OUTPUT_PATH, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    train_count = sum(row["split"] == "train" for row in rows)
    test_count = sum(row["split"] == "test" for row in rows)

    print("DATASET MANIFEST CREATED")
    print("------------------------")
    print("Output:", OUTPUT_PATH)
    print("Total records:", len(rows))
    print("Training records:", train_count)
    print("Testing records:", test_count)
    print("\nFirst 5 records:")

    for row in rows[:5]:
        print(row)


if __name__ == "__main__":
    main()
