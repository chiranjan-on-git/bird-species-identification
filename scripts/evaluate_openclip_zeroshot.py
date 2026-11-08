
import numpy as np
import torch
import open_clip
from pathlib import Path

from src.data.dataset import CUBDataset
from scripts.train_baseline import create_stratified_split
from src.models.openclip_encoder import (
    MODEL_NAME,
    PRETRAINED,
    get_device
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "openclip_features"
    / "cub_train_features.npz"
)

CLASSES_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CUB_200_2011"
    / "classes.txt"
)


def load_species_names():
    species_names = []

    with open(CLASSES_PATH, "r") as file:
        for line in file:
            _, name = line.strip().split(" ", 1)
            name = name.split(".", 1)[1]
            name = name.replace("_", " ")
            species_names.append(name)

    return species_names


def main():

    device = get_device()

    print(f"Device: {device}")
    print("Loading OpenCLIP model...")

    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED
    )

    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    for parameter in model.parameters():
        parameter.requires_grad = False

    print("Loading species names...")
    species_names = load_species_names()

    assert len(species_names) == 200

    # One fixed prompt for every species.
    prompts = [
        f"a photo of a {name}, a bird."
        for name in species_names
    ]

    text_tokens = tokenizer(prompts).to(device)

    print("Encoding species descriptions...")

    with torch.inference_mode():

        text_features = model.encode_text(text_tokens)

        text_features = text_features / text_features.norm(
            dim=-1,
            keepdim=True
        )

    print(f"Text embeddings: {text_features.shape}")

    # Load cached image embeddings.
    print("\nLoading cached image features...")

    saved = np.load(FEATURE_PATH)

    image_features = torch.from_numpy(
        saved["features"]
    ).to(device)

    labels = torch.from_numpy(
        saved["labels"]
    ).long().to(device)

    # Recreate the exact validation split.
    dataset = CUBDataset(split="train")

    _, val_indices = create_stratified_split(dataset)

    val_features = image_features[val_indices]
    val_labels = labels[val_indices]

    print(f"Validation images: {len(val_indices)}")

    # Calculate image-text cosine similarities.
    print("Calculating image-text similarities...")

    with torch.inference_mode():

        similarities = val_features @ text_features.T

        predictions = similarities.argmax(dim=1)

        top5_predictions = similarities.topk(
            5,
            dim=1
        ).indices

        top1_accuracy = (
            (predictions == val_labels)
            .float()
            .mean()
            .item() * 100
        )

        top5_accuracy = (
            (top5_predictions == val_labels.unsqueeze(1))
            .any(dim=1)
            .float()
            .mean()
            .item() * 100
        )

    print("\nOPENCLIP ZERO-SHOT RESULTS")
    print("-" * 50)

    print(f"Top-1 accuracy: {top1_accuracy:.2f}%")
    print(f"Top-5 accuracy: {top5_accuracy:.2f}%")

    print("\nEXAMPLE PREDICTIONS")
    print("-" * 50)

    for index in range(min(10, len(val_indices))):

        actual_id = val_labels[index].item()
        predicted_id = predictions[index].item()

        top5_ids = top5_predictions[index].tolist()

        print(f"\nImage {index + 1}")
        print(f"Actual: {species_names[actual_id]}")
        print(f"Predicted: {species_names[predicted_id]}")

        print("Top 5 candidates:")

        for rank, class_id in enumerate(top5_ids, start=1):
            print(f"  {rank}. {species_names[class_id]}")

    print("\nOPENCLIP ZERO-SHOT EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
