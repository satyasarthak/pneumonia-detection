"""Tests for image resolution and DICOM decode.

Feature: pneumonia-detection
Covers Property 4 (Req 1.5) and a DICOM decode integration test (Req 3.1).
"""

from __future__ import annotations

import os
import zipfile

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.images import ImageSource, load_dicom, resolve_images

pydicom = pytest.importorskip("pydicom")


def _write_dummy_dcm(path: str, rows: int = 16, cols: int = 16) -> None:
    """Create a minimal but valid DICOM file with a pixel array."""
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
def dicom_dir(tmp_path):
    """A directory containing three dummy .dcm files (p0, p1, p2)."""
    for pid in ["p0", "p1", "p2"]:
        _write_dummy_dcm(str(tmp_path / f"{pid}.dcm"))
    return str(tmp_path)


# --- Property 4: Missing images partition the records ---
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    present=st.lists(st.sampled_from(["p0", "p1", "p2"]), max_size=6),
    absent=st.lists(st.sampled_from(["x0", "x1", "x2"]), max_size=6),
)
def test_property_4_missing_images_partition(present, absent, dicom_dir):
    """Feature: pneumonia-detection, Property 4: Missing images partition the
    records. Validates Requirements 1.5."""
    pids = present + absent
    df = pd.DataFrame({"patientId": pids, "class_label": ["Normal"] * len(pids)})

    present_df, missing_ids = resolve_images(df, dicom_dir)

    # Disjoint.
    assert set(present_df["patientId"]).isdisjoint(set(missing_ids))
    # Covering: union equals input (as multiset counts).
    assert len(present_df) + len(missing_ids) == len(df)
    # Every present record references an existing file.
    with ImageSource(dicom_dir) as src:
        assert all(src.exists(pid) for pid in present_df["patientId"])


# --- Integration: DICOM decode returns a 2D numeric array (Req 3.1) ---
def test_load_dicom_returns_2d_array(dicom_dir):
    arr = load_dicom(os.path.join(dicom_dir, "p0.dcm"))
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2
    assert arr.shape == (16, 16)


def test_image_source_reads_from_zip(tmp_path):
    # Build a zip mirroring the RSNA nesting.
    dcm_path = tmp_path / "p0.dcm"
    _write_dummy_dcm(str(dcm_path))
    zip_path = tmp_path / "imgs.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(dcm_path, "stage_2_train_images/p0.dcm")

    with ImageSource(str(zip_path)) as src:
        assert src.exists("p0")
        arr = load_dicom(src, "p0")
    assert arr.shape == (16, 16)
