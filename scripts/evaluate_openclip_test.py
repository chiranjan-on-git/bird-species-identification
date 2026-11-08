
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import open_clip

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

from src.models.openclip_encoder import (
    MODEL_NAME,
    PRETRAINED,
    get_device
)
from scripts.train_openclip_mlp import OpenCLIPMLP


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "openclip_features" / "cub_test_features.npz"
)

LINEAR_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints"
    / "openclip_linear_classifier_best.pth"
)

MLP_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints"
    / "openclip_mlp_best.pth"
)

CLASSES_PATH = (
    PROJECT_ROOT / "data" / "raw"
    / "CUB_200_2011" / "classes.txt"
)


def load_species_names():
    names = []

    with open(CLASSES_PATH, "r") as file:
        for line in file:
            _, name = line.strip().split(" ", 1)
            names.append(name.split(".", 1)[1].replace("_", " "))

    return names


def calculate_metrics(logits, labels):
    predictions = logits.argmax(dim=1)

    top5_predictions = logits.topk(5, dim=1).indices

    top1 = accuracy_score(
        labels.cpu().numpy(),
        predictions.cpu().numpy()
    ) * 100

    top5 = (
        (top5_predictions == labels.unsqueeze(1))
        .any(dim=1)
        .float()
        .mean()
        .item() * 100
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels.cpu().numpy(),
        predictions.cpu().numpy(),
        labels=np.arange(200),
        average="macro",
        zero_division=0
    )

    return {
        "top1": top1,
        "top5": top5,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100
    }


def print_metrics(name, metrics):
    print(f"\n{name}")
    print("-" * 55)
    print(f"Top-1 accuracy:       {metrics['top1']:.2f}%")
    print(f"Top-5 accuracy:       {metrics['top5']:.2f}%")
    print(f"Macro precision:      {metrics['precision']:.2f}%")
    print(f"Macro recall:         {metrics['recall']:.2f}%")
    print(f"Macro F1-score:       {metrics['f1']:.2f}%")


def main():
    device = get_device()

    print(f"Device: {device}")
    print("Loading official test features...")

    saved = np.load(FEATURE_PATH)

    features = torch.from_numpy(saved["features"]).to(device)
    labels = torch.from_numpy(saved["labels"]).long().to(device)

    print(f"Test images: {len(labels)}")
    print(f"Feature dimensions: {features.shape}")

    species_names = load_species_names()

    results = {}

    # ------------------------------------------------
    # 1. Zero-shot OpenCLIP
    # ------------------------------------------------

    print("\nLoading OpenCLIP for zero-shot evaluation...")

    clip_model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED
    )

    clip_model = clip_model.to(device)
    clip_model.eval()

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    prompts = [
        f"a photo of a {name}, a bird."
        for name in species_names
    ]

    text_tokens = tokenizer(prompts).to(device)

    with torch.inference_mode():
        text_features = clip_model.encode_text(text_tokens)

        text_features = text_features / text_features.norm(
            dim=-1,
            keepdim=True
        )

        zero_shot_logits = features @ text_features.T

    results["Zero-shot"] = calculate_metrics(
        zero_shot_logits,
        labels
    )

    print_metrics("ZERO-SHOT OPENCLIP", results["Zero-shot"])

    del clip_model, text_features, text_tokens

    if device.type == "mps":
        torch.mps.empty_cache()

    # ------------------------------------------------
    # 2. Linear classifier
    # ------------------------------------------------

    print("\nLoading linear classifier...")

    linear_checkpoint = torch.load(
        LINEAR_CHECKPOINT,
        map_location=device,
        weights_only=True
    )

    linear_model = nn.Linear(512, 200).to(device)

    linear_model.load_state_dict(
        linear_checkpoint["model_state_dict"]
    )

    linear_model.eval()

    with torch.inference_mode():
        linear_logits = linear_model(features)

    results["Linear"] = calculate_metrics(
        linear_logits,
        labels
    )

    print_metrics(
        "OPENCLIP LINEAR CLASSIFIER",
        results["Linear"]
    )

    # ------------------------------------------------
    # 3. MLP classifier
    # ------------------------------------------------

    print("\nLoading MLP classifier...")

    mlp_checkpoint = torch.load(
        MLP_CHECKPOINT,
        map_location=device,
        weights_only=True
    )

    mlp_model = OpenCLIPMLP(
        input_dim=mlp_checkpoint["input_dim"],
        hidden_dim=mlp_checkpoint["hidden_dim"],
        num_classes=mlp_checkpoint["num_classes"]
    ).to(device)

    mlp_model.load_state_dict(
        mlp_checkpoint["model_state_dict"]
    )

    mlp_model.eval()

    with torch.inference_mode():
        mlp_logits = mlp_model(features)

    results["MLP"] = calculate_metrics(
        mlp_logits,
        labels
    )

    print_metrics(
        "OPENCLIP MLP CLASSIFIER",
        results["MLP"]
    )

    # ------------------------------------------------
    # Final comparison
    # ------------------------------------------------

    print("\n" + "=" * 65)
    print("OFFICIAL CUB TEST SET: FINAL COMPARISON")
    print("=" * 65)

    print(
        f"{'Model':<20}"
        f"{'Top-1':>10}"
        f"{'Top-5':>10}"
        f"{'Macro F1':>12}"
    )

    for name, metrics in results.items():
        print(
            f"{name:<20}"
            f"{metrics['top1']:>9.2f}%"
            f"{metrics['top5']:>9.2f}%"
            f"{metrics['f1']:>11.2f}%"
        )

    print("\nOFFICIAL TEST EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
