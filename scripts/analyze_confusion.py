
import torch
from torch.utils.data import DataLoader, Subset
from collections import Counter

from src.data.dataset import CUBDataset
from src.data.preprocessing import get_eval_transforms
from src.models.baseline import build_model, get_device
from scripts.train_baseline import create_stratified_split


def main():
    device = get_device()

    # Recreate the original validation split.
    split_dataset = CUBDataset(split="train")
    _, val_indices = create_stratified_split(split_dataset)

    dataset = CUBDataset(
        split="train",
        transform=get_eval_transforms(),
    )

    val_dataset = Subset(dataset, val_indices)

    loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    # Load the saved best checkpoint.
    checkpoint = torch.load(
        "checkpoints/mobilenet_v3_small_best.pth",
        map_location=device,
        weights_only=False,
    )

    model = build_model(
        num_classes=200,
        pretrained=False,
        freeze_backbone=True,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    confusion = Counter()
    total = 0
    correct = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            for actual, predicted in zip(
                labels.cpu().tolist(),
                predictions.cpu().tolist(),
            ):
                total += 1

                if actual == predicted:
                    correct += 1
                else:
                    confusion[(actual, predicted)] += 1

    print("VALIDATION RESULTS")
    print("------------------")
    print("Total images:", total)
    print("Correct predictions:", correct)
    print("Incorrect predictions:", total - correct)
    print(f"Accuracy: {100 * correct / total:.2f}%")

    print("\nTOP 20 MOST FREQUENT MISCLASSIFICATIONS")
    print("---------------------------------------")

    for (actual, predicted), count in confusion.most_common(20):
        actual_name = dataset.class_names[actual]
        predicted_name = dataset.class_names[predicted]

        print(
            f"{actual_name} -> {predicted_name}: "
            f"{count} image(s)"
        )


if __name__ == "__main__":
    main()
