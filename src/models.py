"""Model architectures: baseline CNN and transfer-learning models.

Two builders:

* :func:`build_baseline_cnn` - a CNN trained from scratch (grayscale input),
  serving as the performance baseline.
* :func:`build_transfer_model` - a pretrained ImageNet backbone (VGG16,
  ResNet50, or MobileNetV2) with a custom 3-class softmax head (RGB input).

Keras/TensorFlow is imported lazily inside the builders so that importing this
module (e.g. for documentation or partial test collection) does not require a
TensorFlow installation. All models compile with categorical cross-entropy and
accept ``class_weight`` at ``fit`` time for imbalance handling.

Requirements: 4.1, 5.1, 5.2
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .constants import NUM_CLASSES

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tensorflow import keras

# Supported pretrained backbones -> (keras.applications module attr).
SUPPORTED_BACKBONES = ("VGG16", "ResNet50", "MobileNetV2")


def build_baseline_cnn(
    input_shape: tuple[int, int, int] = (224, 224, 1),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = 1e-3,
) -> "keras.Model":
    """Build and compile a CNN from scratch with a 3-node softmax output.

    Architecture (per design):
        Input(H, W, 1)
          -> [Conv2D(32) -> BN -> ReLU -> MaxPool]
          -> [Conv2D(64) -> BN -> ReLU -> MaxPool]
          -> [Conv2D(128) -> BN -> ReLU -> MaxPool]
          -> GlobalAveragePooling2D -> Dropout(0.5)
          -> Dense(128, ReLU) -> Dense(num_classes, softmax)

    Requirements: 4.1
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=input_shape)
    x = inputs
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="baseline_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _get_backbone(backbone: str, input_shape: tuple[int, int, int]):
    """Instantiate a pretrained backbone with an ImageNet-weighted, headless top."""
    from tensorflow.keras import applications

    name = backbone
    if name not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone {backbone!r}; choose from {SUPPORTED_BACKBONES}"
        )

    factory = {
        "VGG16": applications.VGG16,
        "ResNet50": applications.ResNet50,
        "MobileNetV2": applications.MobileNetV2,
    }[name]

    return factory(weights="imagenet", include_top=False, input_shape=input_shape)


def build_transfer_model(
    backbone: str,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = NUM_CLASSES,
    freeze_base: bool = True,
    learning_rate: float = 1e-3,
) -> "keras.Model":
    """Build and compile a transfer-learning model.

    A pretrained ``backbone`` (VGG16 | ResNet50 | MobileNetV2) with
    ``include_top=False`` is topped with a custom classification head:

        base -> GlobalAveragePooling2D -> Dropout(0.4)
             -> Dense(128, ReLU) -> Dense(num_classes, softmax)

    When ``freeze_base`` is True the backbone weights are frozen so only the
    head trains (feature extraction); set it False to fine-tune.

    Requirements: 5.1, 5.2
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    base = _get_backbone(backbone, input_shape)
    base.trainable = not freeze_base

    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name=f"transfer_{backbone.lower()}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
