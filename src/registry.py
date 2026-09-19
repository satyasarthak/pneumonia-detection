"""Model registry: serialize and reload trained models.

Persists the selected best model to a single artifact and reloads it for
inference. Uses the native Keras v3 format (``.keras``), which bundles the
architecture, weights, and compilation state into one file and round-trips
predictions exactly.

Keras/TensorFlow is imported lazily so this module can be imported without TF.

Requirements: 5.6, 5.7
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tensorflow import keras

DEFAULT_MODEL_PATH = os.path.join("models", "best_model.keras")


def save_model(model: "keras.Model", path: str = DEFAULT_MODEL_PATH) -> str:
    """Serialize ``model`` to a persistent ``.keras`` artifact.

    The parent directory is created if needed. Returns the path written.

    Requirements: 5.6
    """
    # Normalize to the native Keras format so a single file holds everything.
    if not path.endswith(".keras"):
        path = path + ".keras"

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    model.save(path)
    return path


def load_model(path: str = DEFAULT_MODEL_PATH) -> "keras.Model":
    """Reload a serialized model from the registry.

    Requirements: 5.7
    """
    from tensorflow import keras

    if not os.path.exists(path):
        raise FileNotFoundError(f"No model artifact at {path}")
    return keras.models.load_model(path)
