"""Training entry points.

Fit the baseline CNN and transfer-learning models using the memory-efficient
``XrayBatchGenerator``. Class weights are computed from the training-set
frequencies so the ~2x class imbalance (No Lung Opacity vs. Lung Opacity)
does not bias the model toward the majority class.

Keras/TensorFlow is imported lazily so the module can be imported without TF.

Requirements: 4.2, 5.1
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .constants import CLASS_TO_INDEX, NUM_CLASSES
from .generator import XrayBatchGenerator
from .models import build_baseline_cnn, build_transfer_model

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tensorflow import keras


def compute_class_weights(
    df: pd.DataFrame, label_col: str = "class_label"
) -> dict[int, float]:
    """Inverse-frequency class weights keyed by class index.

    weight[c] = total / (num_classes * count[c]). Balanced classes yield
    weights near 1; rarer classes get a higher weight. Classes absent from
    ``df`` receive weight 1.0.

    Requirements: 4.2, 5.1
    """
    counts = df[label_col].value_counts().to_dict()
    total = len(df)
    weights: dict[int, float] = {}
    for cls, idx in CLASS_TO_INDEX.items():
        c = counts.get(cls, 0)
        weights[idx] = (total / (NUM_CLASSES * c)) if c > 0 else 1.0
    return weights


def make_generators(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_source,
    target_channels: int,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    augment: bool = True,
) -> tuple[XrayBatchGenerator, XrayBatchGenerator]:
    """Build (train, val) generators. Train shuffles and may augment; val does not."""
    train_gen = XrayBatchGenerator(
        train_df,
        image_source,
        batch_size=batch_size,
        target_channels=target_channels,
        image_size=image_size,
        augment=augment,
        shuffle=True,
    )
    val_gen = XrayBatchGenerator(
        val_df,
        image_source,
        batch_size=batch_size,
        target_channels=target_channels,
        image_size=image_size,
        augment=False,
        shuffle=False,
    )
    return train_gen, val_gen


def _default_callbacks(patience: int = 3):
    from tensorflow.keras import callbacks

    return [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]


def train_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_source,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    epochs: int = 15,
    use_class_weight: bool = True,
    model: "keras.Model | None" = None,
) -> tuple["keras.Model", "keras.callbacks.History"]:
    """Train the from-scratch baseline CNN (grayscale, 1 channel).

    Returns the trained model and its History. The train/val subsets are
    supplied through generators (Req 4.2).

    Requirements: 4.2
    """
    train_gen, val_gen = make_generators(
        train_df, val_df, image_source,
        target_channels=1, image_size=image_size, batch_size=batch_size,
    )
    if model is None:
        model = build_baseline_cnn(input_shape=(*image_size, 1))

    class_weight = compute_class_weights(train_df) if use_class_weight else None
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=_default_callbacks(),
        verbose=2,
    )
    return model, history


def train_transfer(
    backbone: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_source,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    epochs: int = 15,
    use_class_weight: bool = True,
    freeze_base: bool = True,
    model: "keras.Model | None" = None,
) -> tuple["keras.Model", "keras.callbacks.History"]:
    """Train a transfer-learning model (RGB, 3 channels) for ``backbone``.

    Requirements: 5.1
    """
    train_gen, val_gen = make_generators(
        train_df, val_df, image_source,
        target_channels=3, image_size=image_size, batch_size=batch_size,
    )
    if model is None:
        model = build_transfer_model(
            backbone, input_shape=(*image_size, 3), freeze_base=freeze_base
        )

    class_weight = compute_class_weights(train_df) if use_class_weight else None
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=_default_callbacks(),
        verbose=2,
    )
    return model, history
