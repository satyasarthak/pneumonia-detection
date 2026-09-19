"""Tests for the batched image generator.

Feature: pneumonia-detection
Covers Property 10 (Req 3.7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.generator import XrayBatchGenerator

pydicom = pytest.importorskip("pydicom")


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


@pytest.fixture
def dataset(tmp_path):
    n = 10
    for i in range(n):
        _write_dummy_dcm(str(tmp_path / f"p{i}.dcm"))
    df = pd.DataFrame(
        {
            "patientId": [f"p{i}" for i in range(n)],
            "class_label": ["Normal", "Lung Opacity", "No Lung Opacity / Not Normal"]
            * 3
            + ["Normal"],
        }
    )
    return df, str(tmp_path)


# --- Property 10: Batched generator covers the dataset exactly once per epoch --
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(batch_size=st.integers(min_value=1, max_value=12), channels=st.sampled_from([1, 3]))
def test_property_10_epoch_coverage(batch_size, channels, dataset):
    """Feature: pneumonia-detection, Property 10: Batched generator covers the
    dataset exactly once per epoch. Validates Requirements 3.7."""
    df, image_dir = dataset
    gen = XrayBatchGenerator(
        df, image_dir, batch_size=batch_size, target_channels=channels, image_size=(16, 16)
    )

    total = 0
    for i in range(len(gen)):
        x, y = gen[i]
        assert x.shape[0] <= batch_size
        assert x.shape[0] == y.shape[0]
        if x.shape[0] > 0:
            assert x.shape[1:] == (16, 16, channels)
            assert y.shape[1] == 3
        total += x.shape[0]

    # Every record surfaces exactly once across the epoch.
    assert total == len(df)
