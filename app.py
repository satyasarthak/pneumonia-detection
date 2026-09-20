"""Streamlit app: upload a chest X-ray, get a predicted class + probabilities.

Wraps the shared inference service (src/inference.py) behind a simple upload
UI. The served model is the serialized best model from the registry
(models/best_model.keras). The app validates the upload, decodes and
preprocesses it, runs prediction, and displays the predicted class alongside a
per-class probability bar chart.

Run locally (from the project venv):
    C:\\venvs\\pneu\\Scripts\\streamlit.exe run app.py

Requirements: 6.1, 6.2, 6.3
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

from src.constants import CLASSES
from src.inference import (
    UnsupportedImageError,
    load_image_bytes,
    predict,
    validate_upload,
)
from src.registry import load_model

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("models", "best_model.keras"))
GEOMETRY_FILE = os.path.join("models", "best_model_geometry.txt")


def _resolve_geometry() -> tuple[int, tuple[int, int]]:
    """Determine the served model's input geometry (channels, (H, W)).

    Priority: explicit env vars > models/best_model_geometry.txt (written by the
    training scripts) > sensible default (RGB 160x160). This lets the app adapt
    automatically when a differently-sized best model is deployed.
    """
    default_channels, default_h, default_w = 3, 160, 160
    if os.path.exists(GEOMETRY_FILE):
        try:
            with open(GEOMETRY_FILE) as fh:
                c, h, w = (int(x) for x in fh.read().strip().split(","))
            default_channels, default_h, default_w = c, h, w
        except (ValueError, OSError):
            pass
    channels = int(os.environ.get("TARGET_CHANNELS", str(default_channels)))
    height = int(os.environ.get("IMAGE_HEIGHT", str(default_h)))
    width = int(os.environ.get("IMAGE_WIDTH", str(default_w)))
    return channels, (height, width)


TARGET_CHANNELS, IMAGE_SIZE = _resolve_geometry()


@st.cache_resource
def get_model():
    """Load and cache the served model once per session."""
    return load_model(MODEL_PATH)


def _display_image(img: np.ndarray) -> None:
    disp = img
    if disp.ndim == 3 and disp.shape[-1] == 1:
        disp = disp[..., 0]
    # Scale to 8-bit for display.
    disp = disp.astype(np.float32)
    if disp.max() > disp.min():
        disp = (disp - disp.min()) / (disp.max() - disp.min())
    st.image(disp, caption="Uploaded chest X-ray", clamp=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁")
    st.title("Chest X-ray Pneumonia Detection")
    st.write(
        "Upload a chest X-ray (DICOM or image). The model classifies it as "
        "**Normal**, **Lung Opacity** (pneumonia), or "
        "**No Lung Opacity / Not Normal**, and shows the probability for each class."
    )
    st.caption(
        "Decision-support tool for demonstration only - not a substitute for "
        "professional medical diagnosis."
    )

    if not os.path.exists(MODEL_PATH):
        st.error(
            f"No model found at '{MODEL_PATH}'. Train and serialize a model first "
            "(see run_registry_demo.py)."
        )
        return

    uploaded = st.file_uploader(
        "Choose a chest X-ray file",
        type=["dcm", "dicom", "png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    )
    if uploaded is None:
        return

    file_bytes = uploaded.getvalue()
    try:
        validate_upload(file_bytes, uploaded.name)  # Req 6.3
        img = load_image_bytes(file_bytes, uploaded.name)
    except UnsupportedImageError as exc:
        st.error(f"Upload rejected: {exc}")  # Req 6.3
        return
    except Exception as exc:  # noqa: BLE001 - surface decode issues to the user
        st.error(f"Could not read the image: {exc}")
        return

    _display_image(img)

    with st.spinner("Running inference..."):
        model = get_model()
        label, probs = predict(
            model, img, target_channels=TARGET_CHANNELS, image_size=IMAGE_SIZE
        )  # Req 6.1, 6.2

    st.subheader(f"Prediction: {label}")
    top_p = probs[label]
    st.metric("Confidence", f"{top_p:.1%}")

    # Per-class probability bar chart (Req 6.2).
    chart_df = pd.DataFrame(
        {"probability": [probs[c] for c in CLASSES]}, index=CLASSES
    )
    st.bar_chart(chart_df)

    st.write("Per-class probabilities:")
    for cls in CLASSES:
        st.write(f"- {cls}: {probs[cls]:.3f}")


if __name__ == "__main__":
    main()
