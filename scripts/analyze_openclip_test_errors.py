
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch

from src.models.openclip_encoder import get_device
from scripts.train_openclip_mlp import OpenCLIPMLP


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "openclip_features" / "cub_test_features.npz"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints"
    / "openclip_mlp_best.pth"
)

CLASSES_PATH = (
    PROJECT_ROOT / "data" / "raw"
    / "CUB_200_2011" / "classes.txt"
)

OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_species_names():
    names = []

    with open(CLASSES_PATH, "r") as file:
        for line in file:
            _, name = line.strip().split(" ", 1)
            names.append(name.split(".", 1)[1])

    return names


def main():

    device = get_device()

    print(f"Device: {device}")
    print("Loading official test features...")

    saved = np.load(FEATURE_PATH)

    features = torch.from_numpy(
        saved["features"]
    ).to(device)

    labels = saved["labels"]

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model = OpenCLIPMLP(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_classes=checkpoint["num_classes"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Generating predictions...")

    with torch.inference_mode():
        predictions = []

        for start in range(0, len(features), 256):
            batch = features[start:start + 256]
            outputs = model(batch)
            predictions.extend(
                outputs.argmax(dim=1).cpu().numpy()
            )

    predictions = np.array(predictions)

    species_names = load_species_names()

    # Build the confusion matrix.
    confusion_matrix = np.zeros((200, 200), dtype=np.int64)

    np.add.at(
        confusion_matrix,
        (labels, predictions),
        1
    )

    # Calculate per-species accuracy.
    per_class_results = []

    for class_id, species in enumerate(species_names):

        total = int((labels == class_id).sum())
        correct = int(confusion_matrix[class_id, class_id])

        accuracy = (
            100.0 * correct / total if total > 0 else 0.0
        )

        per_class_results.append({
            "class_id": class_id,
            "species": species,
            "correct": correct,
            "total": total,
            "accuracy_percent": accuracy
        })

    df = pd.DataFrame(per_class_results)

    df = df.sort_values(
        "accuracy_percent",
        ascending=True
    )

    csv_path = OUTPUT_DIR / "openclip_mlp_per_class_test.csv"
    df.to_csv(csv_path, index=False)

    # Save the confusion matrix.
    confusion_df = pd.DataFrame(
        confusion_matrix,
        index=species_names,
        columns=species_names
    )

    confusion_path = OUTPUT_DIR / "openclip_mlp_test_confusion_matrix.csv"
    confusion_df.to_csv(confusion_path)

    # Identify the most frequent misclassification pairs.
    confusion_pairs = Counter()

    for true_id, predicted_id in zip(labels, predictions):
        if true_id != predicted_id:
            confusion_pairs[(int(true_id), int(predicted_id))] += 1

    print("\n20 MOST FREQUENT MISCLASSIFICATIONS")
    print("-" * 65)

    for (true_id, predicted_id), count in confusion_pairs.most_common(20):
        print(
            f"{species_names[true_id]} -> "
            f"{species_names[predicted_id]}: {count}"
        )

    print("\n10 LOWEST-ACCURACY SPECIES")
    print("-" * 65)

    for _, row in df.head(10).iterrows():
        print(
            f"{row['species']:40s} "
            f"{row['correct']}/{row['total']} "
            f"({row['accuracy_percent']:.2f}%)"
        )

    print("\n10 HIGHEST-ACCURACY SPECIES")
    print("-" * 65)

    for _, row in df.tail(10).iloc[::-1].iterrows():
        print(
            f"{row['species']:40s} "
            f"{row['correct']}/{row['total']} "
            f"({row['accuracy_percent']:.2f}%)"
        )

    print("\nFILES SAVED")
    print(f"Per-class results: {csv_path}")
    print(f"Confusion matrix: {confusion_path}")

    print("\nOFFICIAL TEST ERROR ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
