
import torch

from src.models.baseline import build_model, get_device


def main():
    device = get_device()

    print("Selected device:", device)

    print("\nLoading pretrained MobileNetV3-Small...")
    model = build_model(num_classes=200, pretrained=True)
    model = model.to(device)
    model.eval()

    # Count trainable and total parameters.
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Total parameters:", total_parameters)
    print("Trainable parameters:", trainable_parameters)

    # Test a small batch of 224x224 images.
    dummy_images = torch.randn(
        2, 3, 224, 224
    ).to(device)

    with torch.no_grad():
        predictions = model(dummy_images)

    print("Output shape:", predictions.shape)
    print("Output device:", predictions.device)

    assert predictions.shape == (2, 200)
    assert trainable_parameters > 0
    assert trainable_parameters < total_parameters

    print("\nMODEL SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
