"""Tests for training entry points.

Feature: pneumonia-detection
Covers the short training-run integration test (Req 4.2) and class-weight
computation used for imbalance handling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

tf = pytest.importorskip("tensorflow")
pydicom = pytest.importorskip("pydicom")

from src.constants import CLASS_TO_INDEX
from src.models import build_baseline_cnn
from src.train import compute_class_weights, train_baseline


def _write_dummy_dcm(path: str, rows: int = 16, cols: int = 16) -> None:
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    pixels = (np.arange(rows * cols) % 4096).astype(np.uint16).reshape(rows, cols)
    ds.PixelData = pixels.tobytes()
    ds.save_as(path, write_like_original=False)


def test_compute_class_weights_inverse_frequency():
    df = pd.DataFrame(
        {
            "patientId": [f"p{i}" for i in range(10)],
            "class_label": ["Normal"] * 6
            + ["Lung Opacity"] * 3
            + ["No Lung Opacity / Not Normal"] * 1,
        }
    )
    weights = compute_class_weights(df)

    # Keyed by class index, one per class.
    assert set(weights.keys()) == set(CLASS_TO_INDEX.values())
    # Rarer class gets a strictly larger weight than the majority class.
    assert weights[CLASS_TO_INDEX["No Lung Opacity / Not Normal"]] > weights[
        CLASS_TO_INDEX["Normal"]
    ]


# --- Integration: a short fit consuming train/val generators (Req 4.2) ---
def test_short_training_run_wiring(tmp_path):
    n = 9
    for i in range(n):
        _write_dummy_dcm(str(tmp_path / f"p{i}.dcm"))
    df = pd.DataFrame(
        {
            "patientId": [f"p{i}" for i in range(n)],
            "class_label": ["Normal", "Lung Opacity", "No Lung Opacity / Not Normal"] * 3,
        }
    )
    train_df = df.iloc[:6].reset_index(drop=True)
    val_df = df.iloc[6:].reset_index(drop=True)

    model = build_baseline_cnn(input_shape=(16, 16, 1))
    model, history = train_baseline(
        train_df,
        val_df,
        str(tmp_path),
        image_size=(16, 16),
        batch_size=3,
        epochs=1,
        model=model,
    )

    # Training ran and produced loss/val_loss history (train + val wired).
    assert "loss" in history.history
    assert "val_loss" in history.history
    assert len(history.history["loss"]) >= 1
