
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data.dataset import CUBDataset
from src.data.preprocessing import (
    get_train_transforms,
    get_eval_transforms,
)
from src.models.baseline import build_model, get_device


# Experiment configuration
SEED = 42
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.001
VALIDATION_FRACTION = 0.10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def create_stratified_split(dataset):
    """Create reproducible train/validation indices by species."""
    random_generator = random.Random(SEED)
    class_indices = {}

    for index, image_id in enumerate(dataset.image_ids):
        label = int(dataset.labels[image_id]) - 1
        class_indices.setdefault(label, []).append(index)

    train_indices = []
    val_indices = []

    for label in sorted(class_indices):
        indices = class_indices[label].copy()
        random_generator.shuffle(indices)

        val_count = max(
            1,
            round(len(indices) * VALIDATION_FRACTION)
        )

        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    random_generator.shuffle(train_indices)
    random_generator.shuffle(val_indices)

    return train_indices, val_indices


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = get_device()
    print("Training device:", device)

    # Create two views of the same official training split.
    train_base = CUBDataset(
        split="train",
        transform=get_train_transforms(),
    )

    val_base = CUBDataset(
        split="train",
        transform=get_eval_transforms(),
    )

    train_indices, val_indices = create_stratified_split(train_base)

    train_dataset = Subset(train_base, train_indices)
    val_dataset = Subset(val_base, val_indices)

    print("Training subset:", len(train_dataset))
    print("Validation subset:", len(val_dataset))
    print("Official test set: reserved for final evaluation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(
        num_classes=200,
        pretrained=True,
        freeze_backbone=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
    )

    # Keep the frozen feature extractor in evaluation mode.
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_accuracy = 0.0

    print("\nStarting baseline training...\n")

    for epoch in range(EPOCHS):
        model.train()
        model.features.eval()
        model.avgpool.eval()

        total_loss = 0.0
        total = 0
        correct = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_accuracy = 100.0 * correct / total

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            checkpoint_path = (
                CHECKPOINT_DIR / "mobilenet_v3_small_best.pth"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_accuracy,
                    "class_names": train_base.class_names,
                },
                checkpoint_path,
            )

            print("Saved best checkpoint:", checkpoint_path)

    print("\nTRAINING COMPLETE")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")


if __name__ == "__main__":
    main()
