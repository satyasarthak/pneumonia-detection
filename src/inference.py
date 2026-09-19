"""Shared inference service.

The prediction path used by both the notebooks/demos and the Streamlit app:
validate an uploaded file, decode + preprocess it into a model-ready tensor,
and produce a predicted class label with per-class probabilities.

Keeping this logic in one place guarantees the app and the offline evaluation
use identical preprocessing and label mapping.

Requirements: 5.7, 6.1, 6.2, 6.3
"""

from __future__ import annotations

import io
import os

import numpy as np

from .constants import CLASSES, INDEX_TO_CLASS
from .preprocess import GRAYSCALE, RGB, normalize, resize, to_channels

# Accepted upload extensions: DICOM plus common raster image formats.
SUPPORTED_EXTENSIONS = {".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_DICOM_MAGIC = b"DICM"  # present at byte offset 128 in a DICOM preamble


class UnsupportedImageError(Exception):
    """Raised when an uploaded file is not a supported chest X-ray image."""


def validate_upload(file_bytes: bytes, filename: str) -> None:
    """Reject inputs that are not a supported chest X-ray image.

    Validation is by extension and, for DICOM, a light content check on the
    ``DICM`` magic bytes. Raster images are accepted by extension and verified
    when decoded. Raises :class:`UnsupportedImageError` on failure so the
    caller can reject the upload without running the model.

    Requirements: 6.3
    """
    if not filename:
        raise UnsupportedImageError("No filename provided for the upload.")

    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedImageError(
            f"Unsupported file type '{ext}'. Supported: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    if not file_bytes:
        raise UnsupportedImageError("Uploaded file is empty.")

    if ext in {".dcm", ".dicom"}:
        # DICOM files carry a 128-byte preamble followed by 'DICM'.
        if len(file_bytes) < 132 or file_bytes[128:132] != _DICOM_MAGIC:
            raise UnsupportedImageError(
                "File has a DICOM extension but is not a valid DICOM file."
            )


def load_image_bytes(file_bytes: bytes, filename: str) -> np.ndarray:
    """Decode uploaded bytes into a 2D (or channel-last) pixel array.

    DICOM is decoded via pydicom; other formats via Pillow. Assumes
    :func:`validate_upload` has already run.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext in {".dcm", ".dicom"}:
        import pydicom

        ds = pydicom.dcmread(io.BytesIO(file_bytes))
        return np.asarray(ds.pixel_array)

    from PIL import Image

    img = Image.open(io.BytesIO(file_bytes))
    return np.asarray(img)


def preprocess_for_inference(
    img: np.ndarray, target_channels: int, image_size: tuple[int, int] = (224, 224)
) -> np.ndarray:
    """Resize, normalize, and channel-convert an image for the served model.

    Returns a batch of one: shape ``(1, H, W, target_channels)``.
    """
    arr = resize(img, image_size)
    arr = normalize(arr)
    target = GRAYSCALE if target_channels == 1 else RGB
    arr = to_channels(arr, target).astype(np.float32)
    return arr[np.newaxis, ...]


def predict(model, img: np.ndarray, target_channels: int | None = None,
            image_size: tuple[int, int] = (224, 224)) -> tuple[str, dict[str, float]]:
    """Predict the class of a single image.

    ``img`` may be a raw 2D/channel-last array (it is preprocessed here) or an
    already-batched, model-ready tensor of shape ``(1, H, W, C)``. When
    ``target_channels`` is given, ``img`` is treated as raw and preprocessed.

    Returns ``(predicted_label, {class_name: probability})`` where the label is
    ``CLASSES[argmax]`` and probabilities cover all three classes.

    Requirements: 5.7, 6.1, 6.2
    """
    if target_channels is not None:
        x = preprocess_for_inference(img, target_channels, image_size)
    else:
        x = img if img.ndim == 4 else img[np.newaxis, ...]

    probs = np.asarray(model.predict(x, verbose=0))[0]
    idx = int(np.argmax(probs))
    label = INDEX_TO_CLASS[idx]
    probabilities = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    return label, probabilities
