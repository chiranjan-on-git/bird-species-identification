
import torch

from src.data.dataset import CUBDataset
from src.models.openclip_encoder import load_openclip_model


def main():
    print("Loading OpenCLIP ViT-B/32...")

    model, preprocess, device = load_openclip_model()

    print("Selected device:", device)

    # Load one bird image.
    dataset = CUBDataset(split="train")
    sample = dataset[0]

    image = preprocess(sample["image"]).unsqueeze(0)
    image = image.to(device)

    print("Input shape:", image.shape)
    print("Species:", sample["species"])

    # Extract image features.
    with torch.no_grad():
        features = model.encode_image(image)

    # Normalize feature vectors.
    features = features / features.norm(
        dim=-1,
        keepdim=True,
    )

    print("Feature shape:", features.shape)
    print("Feature device:", features.device)
    print("Feature dtype:", features.dtype)
    print("Feature norm:", features.norm(dim=-1).item())

    assert features.shape == (1, 512)
    assert torch.isfinite(features).all()

    print("\nOPENCLIP SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
