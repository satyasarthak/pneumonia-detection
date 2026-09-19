"""Tests for the modeling layer.

Feature: pneumonia-detection
Covers Property 11 (Req 4.1, 5.1, 5.2) and the distinct-backbones example
(Req 5.1).

These tests require TensorFlow. They are skipped automatically when TF is not
installed and run as-is once it is available. To keep them fast, models use a
tiny input shape rather than the full 224x224.
"""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.constants import NUM_CLASSES
from src.models import SUPPORTED_BACKBONES, build_baseline_cnn, build_transfer_model


# --- Property 11: models produce a valid 3-class softmax output ---
def test_property_11_baseline_softmax_output():
    """Feature: pneumonia-detection, Property 11: Classification models produce
    a valid 3-class softmax output. Validates Requirements 4.1, 5.1, 5.2."""
    model = build_baseline_cnn(input_shape=(32, 32, 1))
    batch = np.random.default_rng(0).random((4, 32, 32, 1)).astype("float32")

    out = model.predict(batch, verbose=0)

    assert out.shape == (4, NUM_CLASSES)
    assert np.all(out >= 0.0)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("backbone", ["ResNet50", "MobileNetV2"])
def test_property_11_transfer_softmax_output(backbone):
    """Feature: pneumonia-detection, Property 11: Classification models produce
    a valid 3-class softmax output. Validates Requirements 4.1, 5.1, 5.2."""
    # Backbones need a minimum spatial size; 32x32x3 is safe for these two.
    model = build_transfer_model(backbone, input_shape=(32, 32, 3))
    batch = np.random.default_rng(1).random((2, 32, 32, 3)).astype("float32")

    out = model.predict(batch, verbose=0)

    assert out.shape == (2, NUM_CLASSES)
    assert np.all(out >= 0.0)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)


# --- Example: at least two distinct backbones, all distinct from baseline ---
def test_at_least_two_distinct_backbones():
    """Validates Requirements 5.1: >= 2 distinct pretrained backbones built."""
    baseline = build_baseline_cnn(input_shape=(32, 32, 1))
    m1 = build_transfer_model("ResNet50", input_shape=(32, 32, 3))
    m2 = build_transfer_model("MobileNetV2", input_shape=(32, 32, 3))

    names = {baseline.name, m1.name, m2.name}
    assert len(names) == 3  # all three architectures are distinct
    assert len({m1.name, m2.name}) == 2  # two distinct transfer backbones


def test_unsupported_backbone_rejected():
    with pytest.raises(ValueError):
        build_transfer_model("NotARealNet", input_shape=(32, 32, 3))


def test_supported_backbones_constant():
    assert set(SUPPORTED_BACKBONES) == {"VGG16", "ResNet50", "MobileNetV2"}
