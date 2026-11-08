
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

SEED = 42
BATCH_SIZE = 8
EPOCHS = 5

BACKBONE_LR = 0.00005
HEAD_LR = 0.001

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def create_stratified_split(dataset):
    rng = random.Random(SEED)
    class_indices = {}

    for index, image_id in enumerate(dataset.image_ids):
        label = int(dataset.labels[image_id]) - 1
        class_indices.setdefault(label, []).append(index)

    train_indices = []
    val_indices = []

    for indices in class_indices.values():
        indices = indices.copy()
        rng.shuffle(indices)

        val_count = max(1, round(len(indices) * 0.10))

        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

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
            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = get_device()
    print("Training device:", device)

    train_base = CUBDataset(
        split="train",
        transform=get_train_transforms(),
    )

    val_base = CUBDataset(
        split="train",
        transform=get_eval_transforms(),
    )

    train_indices, val_indices = create_stratified_split(
        train_base
    )

    train_loader = DataLoader(
        Subset(train_base, train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        Subset(val_base, val_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Training subset:", len(train_loader.dataset))
    print("Validation subset:", len(val_loader.dataset))
    print("Official test set: reserved")

    model = build_model(
        num_classes=200,
        pretrained=True,
        unfreeze_last_n_blocks=3,
    ).to(device)

    backbone_parameters = [
        p for p in model.features.parameters()
        if p.requires_grad
    ]

    head_parameters = [
        p for p in model.classifier.parameters()
        if p.requires_grad
    ]

    print(
        "Trainable backbone parameters:",
        sum(p.numel() for p in backbone_parameters),
    )

    print(
        "Trainable classifier parameters:",
        sum(p.numel() for p in head_parameters),
    )

    optimizer = torch.optim.AdamW([
        {
            "params": backbone_parameters,
            "lr": BACKBONE_LR,
        },
        {
            "params": head_parameters,
            "lr": HEAD_LR,
        },
    ])

    criterion = nn.CrossEntropyLoss()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_accuracy = 0.0

    print("\nStarting partial fine-tuning...\n")

    for epoch in range(EPOCHS):
        model.train()

        # Keep frozen blocks in evaluation mode.
        for block in model.features[:-3]:
            block.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

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
                CHECKPOINT_DIR
                / "mobilenet_v3_small_finetuned_best.pth"
            )

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_accuracy,
                "class_names": train_base.class_names,
            }, checkpoint_path)

            print("Saved best checkpoint:", checkpoint_path)

    print("\nFINE-TUNING COMPLETE")
    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.2f}%"
    )


if __name__ == "__main__":
    main()
