"""Tests for the model registry.

Feature: pneumonia-detection
Covers Property 15 (Req 5.6): a save/reload round trip preserves predictions.

Requires TensorFlow; skipped automatically when unavailable. Uses a tiny model
and generated inputs to stay fast and deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.models import build_baseline_cnn
from src.registry import load_model, save_model


# --- Property 15: Serialization round-trip preserves predictions ---
@pytest.mark.parametrize("seed", list(range(5)))
def test_property_15_roundtrip_preserves_predictions(seed, tmp_path):
    """Feature: pneumonia-detection, Property 15: Serialization round-trip
    preserves predictions. Validates Requirements 5.6.

    Parameterized over several random inputs to approximate the "for any fixed
    input" quantifier while keeping model training out of the loop.
    """
    model = build_baseline_cnn(input_shape=(16, 16, 1))
    rng = np.random.default_rng(seed)
    x = rng.random((3, 16, 16, 1)).astype("float32")

    before = model.predict(x, verbose=0)

    path = save_model(model, str(tmp_path / "m.keras"))
    reloaded = load_model(path)
    after = reloaded.predict(x, verbose=0)

    assert np.allclose(before, after, atol=1e-6)


def test_save_appends_keras_extension(tmp_path):
    model = build_baseline_cnn(input_shape=(16, 16, 1))
    path = save_model(model, str(tmp_path / "no_ext"))
    assert path.endswith(".keras")


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(str(tmp_path / "does_not_exist.keras"))
