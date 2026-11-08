
import torch
from torch.utils.data import DataLoader

from src.data.dataset import CUBDataset
from src.data.preprocessing import get_eval_transforms
from src.models.baseline import build_model, get_device


def main():
    device = get_device()

    dataset = CUBDataset(
        split="train",
        transform=get_eval_transforms(),
    )

    # Recreate the same stratified validation split.
    from scripts.train_baseline import create_stratified_split

    _, val_indices = create_stratified_split(
        CUBDataset(split="train")
    )

    val_dataset = torch.utils.data.Subset(
        dataset,
        val_indices,
    )

    loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

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

    correct_by_class = [0] * 200
    total_by_class = [0] * 200

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

            for label, prediction in zip(
                labels.cpu().tolist(),
                predictions.cpu().tolist(),
            ):
                total_by_class[label] += 1

                if label == prediction:
                    correct_by_class[label] += 1

    results = []

    for class_id in range(200):
        accuracy = (
            100.0 * correct_by_class[class_id]
            / total_by_class[class_id]
        )

        results.append((
            accuracy,
            dataset.class_names[class_id],
            correct_by_class[class_id],
            total_by_class[class_id],
        ))

    print("VALIDATION ACCURACY")
    print(
        f"{100.0 * total_correct / total_samples:.2f}%"
    )

    print("\n10 LOWEST-ACCURACY SPECIES")

    for accuracy, name, correct, total in sorted(results)[:10]:
        print(
            f"{name}: {accuracy:.1f}% "
            f"({correct}/{total})"
        )

    print("\n10 HIGHEST-ACCURACY SPECIES")

    for accuracy, name, correct, total in sorted(
        results,
        reverse=True,
    )[:10]:
        print(
            f"{name}: {accuracy:.1f}% "
            f"({correct}/{total})"
        )


if __name__ == "__main__":
    main()
