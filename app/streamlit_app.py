import streamlit as st
import torch
import torch.nn as nn
import open_clip

from PIL import Image
from pathlib import Path

import io
import json
import zipfile
from datetime import datetime


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints" / "openclip_mlp_best.pth"
)

CLASSES_PATH = (
    PROJECT_ROOT / "data" / "raw" /
    "CUB_200_2011" / "classes.txt"
)

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

NUM_CLASSES = 200


# ============================================================
# 2. MLP CLASSIFIER ARCHITECTURE
# ============================================================

class OpenCLIPMLP(nn.Module):

    def __init__(
        self,
        input_dim=512,
        hidden_dim=256,
        num_classes=200
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# 3. DEVICE CONFIGURATION
# ============================================================

def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# 4. LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    device = get_device()

    # Load pretrained OpenCLIP encoder
    encoder, _, preprocess = (
        open_clip.create_model_and_transforms(
            MODEL_NAME,
            pretrained=PRETRAINED
        )
    )

    encoder = encoder.to(device).eval()

    # Freeze encoder parameters
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    # Load trained MLP classifier
    classifier = OpenCLIPMLP()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    classifier.load_state_dict(
        checkpoint["model_state_dict"]
    )

    classifier = classifier.to(device).eval()

    return encoder, classifier, preprocess, device


# ============================================================
# 5. LOAD SPECIES NAMES
# ============================================================

@st.cache_data
def load_class_names():

    with open(CLASSES_PATH, "r") as file:

        names = [
            line.strip().split(".", 1)[1].replace("_", " ")
            for line in file
        ]

    if len(names) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} species, found {len(names)}."
        )

    return names


# ============================================================
# 6. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bird Species Identifier",
    page_icon="🐦",
    layout="centered"
)


# ============================================================
# 7. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    .block-container {
        max-width: 1050px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-size: 2.5rem !important;
        font-weight: 750 !important;
        letter-spacing: -1px;
    }

    h2, h3 {
        font-weight: 650 !important;
        letter-spacing: -0.3px;
    }

    .app-subtitle {
        color: #9CA3AF;
        font-size: 1.05rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }

    .prediction-name {
        font-size: 2rem;
        font-weight: 750;
        margin-top: 5px;
        margin-bottom: 5px;
    }

    div.stButton > button {
        border-radius: 9px;
        font-weight: 650;
        min-height: 44px;
        padding-left: 24px;
        padding-right: 24px;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.07);
        padding: 16px;
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 12px;
    }

    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# 8. APPLICATION HEADER
# ============================================================

st.title("Bird Species Identifier")

st.markdown(
    '<p class="app-subtitle">'
    'An AI-powered bird classification system using OpenCLIP '
    'and a trained neural network.'
    '</p>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# 9. IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a bird image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# 10. IMAGE PREDICTION
# ============================================================

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

    except Exception:
        st.error("Unable to read this image. Please upload a valid image.")
        st.stop()

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )

    if st.button("Identify Bird", type="primary"):

        if not CHECKPOINT_PATH.exists():
            st.error(
                f"Model checkpoint not found: {CHECKPOINT_PATH}"
            )
            st.stop()

        if not CLASSES_PATH.exists():
            st.error(
                f"Species names file not found: {CLASSES_PATH}"
            )
            st.stop()

        try:

            with st.spinner("Analyzing image..."):

                # Load models and preprocessing
                encoder, classifier, preprocess, device = (
                    load_models()
                )

                class_names = load_class_names()

                # Preprocess uploaded image
                image_tensor = (
                    preprocess(image)
                    .unsqueeze(0)
                    .to(device)
                )

                # Generate prediction
                with torch.inference_mode():

                    features = encoder.encode_image(
                        image_tensor
                    )

                    # L2 normalization
                    features = features / features.norm(
                        dim=-1,
                        keepdim=True
                    )

                    logits = classifier(features)

                    probabilities = torch.softmax(
                        logits,
                        dim=1
                    )

                    # Extract top 5 predictions
                    top_probabilities, top_indices = torch.topk(
                        probabilities,
                        k=5,
                        dim=1
                    )

            # Main prediction
            predicted_index = top_indices[0][0].item()

            predicted_species = class_names[
                predicted_index
            ]

            predicted_score = (
                top_probabilities[0][0].item()
            )

            # =================================================
            # 11. IDENTIFICATION RESULT
            # =================================================

            st.success("Identification complete")

            st.divider()

            st.subheader("Identification Result")

            st.markdown(
                f'<div class="prediction-name">'
                f'{predicted_species}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.caption(
                "Most likely species identified by the model"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Top Prediction Score",
                    f"{predicted_score * 100:.2f}%"
                )

            with col2:

                st.metric(
                    "Species Candidates",
                    NUM_CLASSES
                )

            st.caption(
                "The model score is a softmax output, not a "
                "calibrated probability or a guarantee of "
                "identification accuracy."
            )

            # =================================================
            # 12. TOP 5 PREDICTIONS
            # =================================================

            st.divider()

            st.subheader("Top 5 Species Predictions")

            st.write(
                "The five species receiving the highest "
                "scores for this image."
            )

            for rank, (probability, index) in enumerate(
                zip(
                    top_probabilities[0],
                    top_indices[0]
                ),
                start=1
            ):

                species = class_names[index.item()]

                score = probability.item()

                col1, col2 = st.columns([5, 1])

                with col1:

                    st.write(
                        f"**{rank}. {species}**"
                    )

                with col2:

                    st.write(
                        f"**{score * 100:.2f}%**"
                    )

                st.progress(score)

            st.caption(
                "Scores are normalized across the 200 "
                "candidate species using the softmax function."
            )

            #add from here 

            # =================================================
            # DOWNLOADABLE PREDICTION REPORT
            # =================================================

            st.divider()

            st.subheader("Download Prediction Report")

            # Prepare Top 5 prediction data
            top5_results = []

            for rank, (probability, index) in enumerate(
                zip(
                    top_probabilities[0],
                    top_indices[0]
                ),
                start=1
            ):
                top5_results.append({
                    "rank": rank,
                    "species": class_names[index.item()],
                    "score": round(probability.item() * 100, 2)
                })

            # Build report metadata
            report = {
                "project": "Bird Species Identifier",
                "model": "OpenCLIP ViT-B/32 + MLP",
                "dataset": "CUB-200-2011",
                "prediction_timestamp": datetime.now().astimezone().isoformat(),
                "uploaded_image": uploaded_file.name,
                "predicted_species": predicted_species,
                "top_prediction_score": round(
                    predicted_score * 100, 2
                ),
                "top_5_predictions": top5_results
            }

            # Create ZIP archive in memory
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(
                zip_buffer,
                mode="w",
                compression=zipfile.ZIP_DEFLATED
            ) as archive:

                archive.writestr(
                    "prediction_report.json",
                    json.dumps(report, indent=4)
                )

                archive.writestr(
                    f"uploaded_image_{Path(uploaded_file.name).name}",
                    uploaded_file.getvalue()
                )

            zip_buffer.seek(0)

            st.download_button(
                label="Download Prediction Report",
                data=zip_buffer.getvalue(),
                file_name="bird_identification_report.zip",
                mime="application/zip",
                on_click="ignore",
                type="primary"
            )

            st.caption(
                "The ZIP file contains the original image and "
                "a JSON report with the prediction results."
            )


        except Exception as error:

            st.error(
                f"An error occurred during identification: {error}"
            )


# ============================================================
# 13. ABOUT THE PROJECT
# ============================================================

st.divider()

st.header("About This Project")

st.write(
    """
    This project explores fine-grained bird species identification
    using computer vision and vision-language models.

    The system uses a pretrained OpenCLIP image encoder to extract
    visual representations from bird photographs. A lightweight
    Multi-Layer Perceptron (MLP) classifier then predicts one of
    200 bird species.
    """
)


# ============================================================
# 14. MODEL ARCHITECTURE
# ============================================================

st.subheader("Model Architecture")

with st.container(border=True):

    st.markdown("**Vision Encoder**")

    st.write(
        "OpenCLIP ViT-B/32 — pretrained on LAION-2B."
    )

    st.markdown("**Classification Head**")

    st.write(
        "MLP neural network: 512 → 256 → 200."
    )

    st.markdown("**Dataset**")

    st.write(
        "CUB-200-2011 — 200 bird species."
    )

    st.markdown("**Feature Extraction**")

    st.write(
        "512-dimensional, L2-normalized image embeddings."
    )


# ============================================================
# 15. MODEL EVALUATION
# ============================================================

st.subheader("Model Evaluation")

metric1, metric2 = st.columns(2)

with metric1:

    st.metric(
        "Test Top-1 Accuracy",
        "73.02%"
    )

with metric2:

    st.metric(
        "Test Top-5 Accuracy",
        "94.10%"
    )

st.caption(
    "Evaluation performed on the official CUB-200-2011 "
    "test split of 5,794 images. These are offline benchmark "
    "results, not live measurements."
)


# ============================================================
# 16. INFERENCE PERFORMANCE
# ============================================================

st.subheader("Inference Performance")

metric1, metric2 = st.columns(2)

with metric1:

    st.metric(
        "Average End-to-End Latency",
        "22.62 ms"
    )

with metric2:

    st.metric(
        "Approximate Throughput",
        "44.21 images/sec"
    )

st.caption(
    "Measured on an Apple M1 using MPS. Benchmark results "
    "are hardware- and workload-dependent."
)


# ============================================================
# 17. FOOTER
# ============================================================

st.divider()

st.caption(
    "Bird Species Identifier | Capstone Research Project | "
    "CUB-200-2011"
)
