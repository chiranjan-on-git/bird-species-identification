
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import CUBDataset
from scripts.train_baseline import create_stratified_split

# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
PATIENCE = 5

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "openclip_features"
    / "cub_train_features.npz"
)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "openclip_linear_classifier_best.pth"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def main():
    set_seed(SEED)

    device = get_device()

    print(f"Device: {device}")
    print("Loading cached OpenCLIP features...")

    saved = np.load(FEATURE_PATH)

    features = torch.from_numpy(saved["features"])
    labels = torch.from_numpy(saved["labels"]).long()

    print(f"Features: {features.shape}")
    print(f"Labels: {labels.shape}")

    # Reproduce the exact MobileNet train/validation split.
    dataset = CUBDataset(split="train")
    train_indices, val_indices = create_stratified_split(dataset)

    print(f"\nTraining samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")

    train_dataset = TensorDataset(
        features[train_indices],
        labels[train_indices]
    )

    val_dataset = TensorDataset(
        features[val_indices],
        labels[val_indices]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Linear classification head: 512 OpenCLIP features -> 200 species.
    model = nn.Linear(512, 200).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_val_accuracy = 0.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print("\nStarting OpenCLIP classifier training...")
    print("-" * 65)

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_labels.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)

        train_loss = total_loss / total
        train_accuracy = 100.0 * correct / total

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}%"
        )

        # Save the model with the best validation accuracy.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "validation_accuracy": best_val_accuracy,
                    "validation_loss": best_val_loss,
                    "seed": SEED,
                },
                CHECKPOINT_PATH
            )

            print("  -> Best checkpoint saved.")

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping: validation accuracy did not improve.")
            break

    print("\n" + "=" * 65)
    print("OPENCLIP CLASSIFIER TRAINING COMPLETE")
    print("=" * 65)
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
