
import torch
import open_clip


MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_openclip_model():
    """
    Load pretrained OpenCLIP ViT-B/32.
    The encoder is frozen for feature extraction.
    """
    device = get_device()

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
    )

    model = model.to(device)
    model.eval()

    # Freeze all model parameters.
    for parameter in model.parameters():
        parameter.requires_grad = False

    return model, preprocess, device
