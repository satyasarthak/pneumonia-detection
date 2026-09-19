"""Tests for the EDA module.

Feature: pneumonia-detection
Covers Property 5 (Req 2.3), sample display/annotation (Req 2.1, 2.2),
imbalance flag (Req 2.4), and before/after preprocessing plot (Req 3.3).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for tests

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.constants import CLASSES
from src.eda import (
    class_distribution,
    imbalance_report,
    plot_before_after,
    plot_class_distribution,
    plot_samples,
    sample_images_per_class,
)

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


@st.composite
def label_frames(draw):
    size = draw(st.integers(min_value=0, max_value=40))
    labels = draw(st.lists(st.sampled_from(CLASSES), min_size=size, max_size=size))
    return pd.DataFrame(
        {"patientId": [f"p{i}" for i in range(size)], "class_label": labels}
    )


# --- Property 5: Class distribution aggregation is exact ---
@settings(max_examples=100)
@given(df=label_frames())
def test_property_5_class_distribution_exact(df):
    """Feature: pneumonia-detection, Property 5: Class distribution aggregation
    is exact. Validates Requirements 2.3."""
    dist = class_distribution(df)

    assert sum(dist.values()) == len(df)
    for cls in CLASSES:
        assert dist[cls] == int((df["class_label"] == cls).sum())


# --- Example: imbalance flag on a skewed fixture (Req 2.4) ---
def test_imbalance_flagged_on_skew():
    dist = {"Normal": 100, "Lung Opacity": 100, "No Lung Opacity / Not Normal": 10}
    report = imbalance_report(dist, threshold=1.5)
    assert report["imbalanced"] is True
    assert report["minority_class"] == "No Lung Opacity / Not Normal"
    assert "imbalance" in report["observation"].lower()


def test_imbalance_not_flagged_when_balanced():
    dist = {"Normal": 100, "Lung Opacity": 100, "No Lung Opacity / Not Normal": 90}
    report = imbalance_report(dist, threshold=1.5)
    assert report["imbalanced"] is False


# --- Example: sample grid returns n images per class annotated (Req 2.1, 2.2) ---
def test_sample_images_and_plot(tmp_path):
    df_rows = []
    for i, cls in enumerate(CLASSES):
        for j in range(3):
            pid = f"{cls[:3]}_{j}"
            _write_dummy_dcm(str(tmp_path / f"{pid}.dcm"))
            df_rows.append({"patientId": pid, "class_label": cls})
    df = pd.DataFrame(df_rows)

    samples = sample_images_per_class(df, str(tmp_path), n=2)
    assert set(samples.keys()) == set(CLASSES)
    for cls in CLASSES:
        assert len(samples[cls]) == 2

    fig = plot_samples(samples, n=2)
    # Each row's axes are titled with the class label (annotation, Req 2.2).
    titles = {ax.get_title() for ax in fig.axes if ax.get_title()}
    assert set(CLASSES).issubset(titles)


def test_plot_class_distribution_runs():
    fig = plot_class_distribution({"Normal": 5, "Lung Opacity": 3, "No Lung Opacity / Not Normal": 2})
    assert fig is not None


# --- Example: before/after preprocessing plot (Req 3.3) ---
def test_plot_before_after_runs():
    raw = np.arange(256).reshape(16, 16).astype(np.uint16)
    processed = (raw / 255.0)[..., np.newaxis]
    fig = plot_before_after(raw, processed)
    assert len(fig.axes) == 2
