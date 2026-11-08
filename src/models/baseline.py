
import torch
import torch.nn as nn

from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def build_model(
    num_classes=200,
    pretrained=True,
    freeze_backbone=True,
    unfreeze_last_n_blocks=0,
):
    weights = (
        MobileNet_V3_Small_Weights.DEFAULT
        if pretrained
        else None
    )

    model = mobilenet_v3_small(weights=weights)

    # Freeze the entire backbone initially.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Optionally unfreeze selected deeper feature blocks.
    if unfreeze_last_n_blocks > 0:
        for block in model.features[-unfreeze_last_n_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    # Replace the original ImageNet classification layer.
    input_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        input_features,
        num_classes,
    )

    return model
