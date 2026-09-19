"""Smoke test for the Streamlit app wiring.

Feature: pneumonia-detection
Exercises the upload -> validate -> preprocess -> predict path using the same
functions the app calls, with a stub model, without launching a server
(Req 6.1, 6.2, 6.3).
"""

from __future__ import annotations

import importlib
import io

import numpy as np
import pytest

from src.constants import CLASSES
from src.inference import UnsupportedImageError, predict, validate_upload


class StubModel:
    def __init__(self, probs):
        self._probs = np.asarray(probs, dtype=np.float32)

    def predict(self, x, verbose=0):
        return np.repeat(self._probs[np.newaxis, :], x.shape[0], axis=0)


def test_app_module_imports():
    """The app module imports cleanly (Streamlit installed, wiring valid)."""
    pytest.importorskip("streamlit")
    app = importlib.import_module("app")
    assert hasattr(app, "main")
    assert hasattr(app, "get_model")


def test_upload_predict_path_png():
    """A valid PNG flows through validate -> predict to a labeled result."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.uint8(np.random.rand(32, 32) * 255)).save(buf, format="PNG")
    data = buf.getvalue()

    validate_upload(data, "scan.png")  # should not raise

    from src.inference import load_image_bytes

    img = load_image_bytes(data, "scan.png")
    model = StubModel([0.1, 0.7, 0.2])
    label, probs = predict(model, img, target_channels=1, image_size=(32, 32))

    assert label == CLASSES[1]  # argmax of [0.1, 0.7, 0.2]
    assert set(probs.keys()) == set(CLASSES)
    assert abs(sum(probs.values()) - 1.0) < 1e-5


def test_upload_rejected_bad_type():
    with pytest.raises(UnsupportedImageError):
        validate_upload(b"whatever", "notes.txt")
