
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch

from src.data.dataset import CUBDataset
from scripts.train_baseline import create_stratified_split
from scripts.train_openclip_mlp import OpenCLIPMLP

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "openclip_features" / "cub_train_features.npz"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints"
    / "openclip_mlp_best.pth"
)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = get_device()

    print(f"Device: {device}")
    print("Loading validation data...")

    saved = np.load(FEATURE_PATH)

    features = torch.from_numpy(saved["features"])
    labels = torch.from_numpy(saved["labels"]).long()

    dataset = CUBDataset(split="train")
    _, val_indices = create_stratified_split(dataset)

    val_features = features[val_indices].to(device)
    val_labels = labels[val_indices].to(device)

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

    print(f"Validation images: {len(val_indices)}")

    with torch.inference_mode():
        logits = model(val_features)

        predictions = logits.argmax(dim=1)
        top5_predictions = logits.topk(5, dim=1).indices

        top1_accuracy = (
            (predictions == val_labels)
            .float().mean().item() * 100
        )

        top5_accuracy = (
            (top5_predictions == val_labels.unsqueeze(1))
            .any(dim=1).float().mean().item() * 100
        )

    print("\nOVERALL VALIDATION RESULTS")
    print("-" * 45)
    print(f"Top-1 accuracy: {top1_accuracy:.2f}%")
    print(f"Top-5 accuracy: {top5_accuracy:.2f}%")

    # Read official species names.
    classes_path = (
        PROJECT_ROOT / "data" / "raw"
        / "CUB_200_2011" / "classes.txt"
    )

    class_names = []

    with open(classes_path, "r") as file:
        for line in file:
            _, species_name = line.strip().split(" ", 1)
            class_names.append(species_name.split(".", 1)[1])

    true_labels = val_labels.cpu().numpy()
    predicted_labels = predictions.cpu().numpy()

    correct_by_class = Counter()
    total_by_class = Counter()
    confusion_pairs = Counter()

    for true_label, predicted_label in zip(
        true_labels, predicted_labels
    ):
        total_by_class[int(true_label)] += 1

        if true_label == predicted_label:
            correct_by_class[int(true_label)] += 1
        else:
            confusion_pairs[
                (int(true_label), int(predicted_label))
            ] += 1

    class_accuracies = []

    for class_id in sorted(total_by_class):
        total = total_by_class[class_id]
        correct = correct_by_class[class_id]
        accuracy = 100.0 * correct / total

        class_accuracies.append(
            (accuracy, class_names[class_id], correct, total)
        )

    class_accuracies.sort()

    print("\n10 SPECIES WITH LOWEST VALIDATION ACCURACY")
    print("-" * 65)

    for accuracy, name, correct, total in class_accuracies[:10]:
        print(f"{name:40s} {correct}/{total} ({accuracy:.1f}%)")

    print("\n10 SPECIES WITH HIGHEST VALIDATION ACCURACY")
    print("-" * 65)

    for accuracy, name, correct, total in class_accuracies[-10:][::-1]:
        print(f"{name:40s} {correct}/{total} ({accuracy:.1f}%)")

    print("\nTOP 15 MISCLASSIFICATION PAIRS")
    print("-" * 65)

    for (true_label, predicted_label), count in (
        confusion_pairs.most_common(15)
    ):
        print(
            f"{class_names[true_label]} -> "
            f"{class_names[predicted_label]}: {count}"
        )

    print("\nOPENCLIP MLP VALIDATION ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
