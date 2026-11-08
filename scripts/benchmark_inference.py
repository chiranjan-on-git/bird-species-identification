
import time
import statistics
from pathlib import Path

import torch
import open_clip

from src.data.dataset import CUBDataset
from src.models.openclip_encoder import (
    MODEL_NAME,
    PRETRAINED,
    get_device
)
from scripts.train_openclip_mlp import OpenCLIPMLP


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints"
    / "openclip_mlp_best.pth"
)

CLASSES_PATH = (
    PROJECT_ROOT / "data" / "raw"
    / "CUB_200_2011" / "classes.txt"
)

WARMUP_RUNS = 10
BENCHMARK_RUNS = 50


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def load_species_names():
    names = []

    with open(CLASSES_PATH, "r") as file:
        for line in file:
            _, name = line.strip().split(" ", 1)
            names.append(name.split(".", 1)[1].replace("_", " "))

    return names


def main():

    device = get_device()

    print(f"Device: {device}")
    print("Loading OpenCLIP encoder...")

    clip_model, _, preprocess = (
        open_clip.create_model_and_transforms(
            MODEL_NAME,
            pretrained=PRETRAINED
        )
    )

    clip_model = clip_model.to(device)
    clip_model.eval()

    for parameter in clip_model.parameters():
        parameter.requires_grad = False

    print("Loading MLP classifier...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    classifier = OpenCLIPMLP(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_classes=checkpoint["num_classes"]
    ).to(device)

    classifier.load_state_dict(
        checkpoint["model_state_dict"]
    )

    classifier.eval()

    # Use one image from the official test dataset.
    dataset = CUBDataset(split="test")
    sample = dataset[0]
    image = sample["image"]

    print(f"Sample species: {sample['species']}")
    print(f"Warm-up runs: {WARMUP_RUNS}")
    print(f"Benchmark runs: {BENCHMARK_RUNS}")

    # -----------------------------------------------
    # Preprocessing benchmark
    # -----------------------------------------------

    print("\nBenchmarking preprocessing...")

    for _ in range(WARMUP_RUNS):
        preprocess(image)

    preprocessing_times = []

    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()
        preprocess(image)
        preprocessing_times.append(
            (time.perf_counter() - start) * 1000
        )

    # -----------------------------------------------
    # Prepare model input
    # -----------------------------------------------

    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # Warm up the model.
    print("Warming up model...")

    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            features = clip_model.encode_image(image_tensor)

            features = features / features.norm(
                dim=-1,
                keepdim=True
            )

            outputs = classifier(features)

    synchronize(device)

    # -----------------------------------------------
    # Encoder + classifier benchmark
    # -----------------------------------------------

    print("Benchmarking encoder and classifier...")

    inference_times = []

    with torch.inference_mode():

        for _ in range(BENCHMARK_RUNS):

            synchronize(device)
            start = time.perf_counter()

            features = clip_model.encode_image(image_tensor)

            features = features / features.norm(
                dim=-1,
                keepdim=True
            )

            outputs = classifier(features)

            synchronize(device)

            inference_times.append(
                (time.perf_counter() - start) * 1000
            )

    predicted_id = outputs.argmax(dim=1).item()
    confidence = torch.softmax(outputs, dim=1)[0, predicted_id].item()

    # -----------------------------------------------
    # End-to-end benchmark
    # -----------------------------------------------

    print("Benchmarking end-to-end prediction...")

    end_to_end_times = []

    with torch.inference_mode():

        for _ in range(BENCHMARK_RUNS):

            synchronize(device)
            start = time.perf_counter()

            processed_image = preprocess(image).unsqueeze(0).to(device)

            features = clip_model.encode_image(processed_image)

            features = features / features.norm(
                dim=-1,
                keepdim=True
            )

            predictions = classifier(features)

            synchronize(device)

            end_to_end_times.append(
                (time.perf_counter() - start) * 1000
            )

    # -----------------------------------------------
    # Report results
    # -----------------------------------------------

    species_names = load_species_names()

    print("\n" + "=" * 60)
    print("INFERENCE PERFORMANCE BENCHMARK")
    print("=" * 60)

    print(f"\nHardware: {device}")
    print(f"Image size: {image.size}")

    print("\nPREPROCESSING")
    print("-" * 40)
    print(
        f"Average: {statistics.mean(preprocessing_times):.2f} ms"
    )

    print("\nMODEL INFERENCE")
    print("-" * 40)
    print(
        f"Average: {statistics.mean(inference_times):.2f} ms"
    )
    print(
        f"Median: {statistics.median(inference_times):.2f} ms"
    )

    print("\nEND-TO-END PREDICTION")
    print("-" * 40)
    print(
        f"Average: {statistics.mean(end_to_end_times):.2f} ms"
    )
    print(
        f"Median: {statistics.median(end_to_end_times):.2f} ms"
    )

    average_latency = statistics.mean(end_to_end_times)

    print(
        f"Approximate throughput: "
        f"{1000 / average_latency:.2f} images/second"
    )

    print("\nSAMPLE PREDICTION")
    print("-" * 40)
    print(f"Actual: {sample['species']}")
    print(f"Predicted: {species_names[predicted_id]}")
    print(f"Classifier softmax score: {confidence:.4f}")

    print("\nINFERENCE BENCHMARK COMPLETE")


if __name__ == "__main__":
    main()
