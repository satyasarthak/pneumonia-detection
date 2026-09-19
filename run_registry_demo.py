"""Serialize the best model, reload it, and run inference (Req 5.6, 5.7).

Takes the currently best-performing trained model, saves it to the registry as
``models/best_model.keras``, reloads it from disk, verifies the reload
reproduces the original predictions, and prints predicted classes for a few
test images.

Usage:
    C:\\venvs\\pneu\\Scripts\\python.exe run_registry_demo.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.constants import INDEX_TO_CLASS
from src.generator import XrayBatchGenerator
from src.registry import load_model, save_model

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_ZIP = os.path.join(HERE, "stage_2_train_images.zip")
MODEL_DIR = os.path.join(HERE, "models")

# The best model (MobileNetV2, selected by macro-F1 in run_train_transfer.py).
SOURCE_MODEL = os.path.join(MODEL_DIR, "transfer_mobilenetv2.keras")
SOURCE_CHANNELS = 3
SOURCE_SIZE = (160, 160)


def main() -> None:
    from tensorflow import keras

    # Load the source (best) model and serialize it to the registry path.
    source = keras.models.load_model(SOURCE_MODEL)
    best_path = save_model(source, os.path.join(MODEL_DIR, "best_model.keras"))
    print(f"Serialized best model -> {best_path}")

    # Reload from the registry (Req 5.7).
    reloaded = load_model(best_path)
    print("Reloaded model from registry.")

    # Build a small inference batch from the persisted test split.
    test_df = pd.read_csv(os.path.join(MODEL_DIR, "test_split.csv"), dtype=str)
    sample = test_df.head(6).reset_index(drop=True)
    gen = XrayBatchGenerator(
        sample, TRAIN_ZIP, batch_size=6,
        target_channels=SOURCE_CHANNELS, image_size=SOURCE_SIZE, shuffle=False,
    )
    x, _ = gen[0]

    # Round-trip check: original vs reloaded predictions match (Req 5.6).
    before = source.predict(x, verbose=0)
    after = reloaded.predict(x, verbose=0)
    match = np.allclose(before, after, atol=1e-6)
    print(f"Round-trip predictions identical: {match}")

    # Inference output: predicted class + probability per image (Req 5.7).
    print("\nInference on sample test images:")
    for i in range(x.shape[0]):
        probs = after[i]
        idx = int(np.argmax(probs))
        pid = sample.iloc[i]["patientId"]
        true = sample.iloc[i]["class_label"]
        print(f"  {pid[:12]}...  true={true:<28} "
              f"pred={INDEX_TO_CLASS[idx]:<28} p={probs[idx]:.3f}")


if __name__ == "__main__":
    main()
