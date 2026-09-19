"""Tests for the shared inference service.

Feature: pneumonia-detection
Covers Property 16 (Req 5.7, 6.1), Property 17 (Req 6.2), and Property 18
(Req 6.3).

A lightweight stub model (returns a fixed softmax vector) stands in for a real
Keras model so the prediction properties run without TensorFlow.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from src.constants import CLASSES, NUM_CLASSES
from src.inference import (
    UnsupportedImageError,
    predict,
    preprocess_for_inference,
    validate_upload,
)


class StubModel:
    """A model whose predict() returns a fixed (batch, 3) softmax matrix."""

    def __init__(self, probs: np.ndarray):
        self._probs = np.asarray(probs, dtype=np.float32)

    def predict(self, x, verbose=0):
        return np.repeat(self._probs[np.newaxis, :], x.shape[0], axis=0)


def _softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max())
    return e / e.sum()


_prob_vectors = st.lists(
    st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False),
    min_size=NUM_CLASSES,
    max_size=NUM_CLASSES,
).map(lambda v: _softmax(np.array(v, dtype=np.float32)))


# --- Property 16: Predicted label maps to a valid category ---
@settings(max_examples=100)
@given(probs=_prob_vectors)
def test_property_16_label_maps_to_category(probs):
    """Feature: pneumonia-detection, Property 16: Predicted label maps to a
    valid category. Validates Requirements 5.7, 6.1."""
    model = StubModel(probs)
    x = np.zeros((1, 8, 8, 1), dtype=np.float32)

    label, _ = predict(model, x)

    assert label in CLASSES
    assert label == CLASSES[int(np.argmax(probs))]


# --- Property 17: Prediction probabilities are a valid distribution ---
@settings(max_examples=100)
@given(probs=_prob_vectors)
def test_property_17_probabilities_valid_distribution(probs):
    """Feature: pneumonia-detection, Property 17: Prediction probabilities are a
    valid distribution over three classes. Validates Requirements 6.2."""
    model = StubModel(probs)
    x = np.zeros((1, 8, 8, 1), dtype=np.float32)

    label, probabilities = predict(model, x)

    assert set(probabilities.keys()) == set(CLASSES)
    assert all(0.0 <= p <= 1.0 for p in probabilities.values())
    assert abs(sum(probabilities.values()) - 1.0) < 1e-5
    # Displayed label corresponds to the max-probability class.
    assert label == max(probabilities, key=probabilities.get)


# --- Property 18: Unsupported uploads are rejected ---
@settings(max_examples=100)
@given(
    name=st.text(min_size=1, max_size=10),
    ext=st.sampled_from([".txt", ".pdf", ".exe", ".csv", ".zip", ".docx", ""]),
    body=st.binary(min_size=0, max_size=32),
)
def test_property_18_unsupported_rejected(name, ext, body):
    """Feature: pneumonia-detection, Property 18: Unsupported uploads are
    rejected. Validates Requirements 6.3."""
    filename = f"{name}{ext}"
    try:
        validate_upload(body, filename)
        # If it did not raise, the extension must be a supported one.
        import os

        assert os.path.splitext(filename)[1].lower() in {
            ".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
        }
    except UnsupportedImageError:
        pass  # expected for unsupported inputs


# --- Examples ---
def test_validate_rejects_fake_dicom():
    import pytest

    with pytest.raises(UnsupportedImageError):
        validate_upload(b"not a dicom", "scan.dcm")


def test_validate_accepts_png_extension():
    # A PNG with a supported extension and non-empty body passes extension check.
    validate_upload(b"\x89PNG\r\n\x1a\n" + b"0" * 20, "scan.png")


def test_preprocess_shapes():
    img = np.arange(32 * 32).reshape(32, 32).astype(np.uint16)
    gray = preprocess_for_inference(img, target_channels=1, image_size=(64, 64))
    rgb = preprocess_for_inference(img, target_channels=3, image_size=(64, 64))
    assert gray.shape == (1, 64, 64, 1)
    assert rgb.shape == (1, 64, 64, 3)
    assert gray.min() >= 0.0 and gray.max() <= 1.0
