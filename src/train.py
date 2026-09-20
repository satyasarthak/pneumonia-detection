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


def train_transfer_finetune(
    backbone: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_source,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    head_epochs: int = 5,
    finetune_epochs: int = 5,
    finetune_lr: float = 1e-5,
    unfreeze_from: float = 0.7,
    use_class_weight: bool = True,
) -> tuple["keras.Model", dict]:
    """Two-phase transfer learning: train the head, then fine-tune the backbone.

    Phase 1 trains only the new classification head with the pretrained base
    frozen (fast, stable). Phase 2 unfreezes the top fraction of the backbone
    (``unfreeze_from`` = 0.7 keeps the bottom 70% frozen and unfreezes the top
    30%) and continues at a much lower learning rate so the pretrained features
    adapt to chest X-rays without being destroyed.

    Fine-tuning is typically the single biggest accuracy lever for transfer
    learning: a frozen-base model can only recombine generic features, whereas
    fine-tuning lets them specialize to lung imagery. Training uses augmentation
    (see XrayBatchGenerator) and class weights for the imbalance.

    Returns the model and a merged history dict (phase1 + phase2).

    Requirements: 5.1
    """
    from tensorflow import keras

    train_gen = XrayBatchGenerator(
        train_df, image_source, batch_size=batch_size,
        target_channels=3, image_size=image_size, augment=True, shuffle=True,
    )
    val_gen = XrayBatchGenerator(
        val_df, image_source, batch_size=batch_size,
        target_channels=3, image_size=image_size, augment=False, shuffle=False,
    )
    class_weight = compute_class_weights(train_df) if use_class_weight else None

    # Phase 1: frozen base, train the head only.
    model = build_transfer_model(
        backbone, input_shape=(*image_size, 3), freeze_base=True
    )
    h1 = model.fit(
        train_gen, validation_data=val_gen, epochs=head_epochs,
        class_weight=class_weight, callbacks=_default_callbacks(), verbose=2,
    )

    # Phase 2: unfreeze the top layers of the backbone and fine-tune.
    base = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):  # the backbone sub-model
            base = layer
            break
    if base is not None:
        base.trainable = True
        cutoff = int(len(base.layers) * unfreeze_from)
        for layer in base.layers[:cutoff]:
            layer.trainable = False
        # Keep BatchNorm layers frozen during fine-tuning for stability.
        for layer in base.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(finetune_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    h2 = model.fit(
        train_gen, validation_data=val_gen, epochs=finetune_epochs,
        class_weight=class_weight, callbacks=_default_callbacks(), verbose=2,
    )

    merged = {k: list(h1.history.get(k, [])) + list(h2.history.get(k, []))
              for k in set(h1.history) | set(h2.history)}
    return model, merged
