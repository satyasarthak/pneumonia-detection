"""Shared pytest fixtures for the pneumonia-detection test suite.

Provides tiny synthetic DataFrames and in-memory pixel arrays so tests run
fast and deterministically without touching the real ~53k-file archives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants import CLASSES


@pytest.fixture
def raw_label_df() -> pd.DataFrame:
    """A small label frame mimicking stage_2_detailed_class_info.csv.

    Includes a duplicated patientId (p1 appears twice) so de-duplication can
    be exercised, and one out-of-vocabulary label ("Bogus") to exercise
    label restriction.
    """
    return pd.DataFrame(
        {
            "patientId": ["p0", "p1", "p1", "p2", "p3", "p4"],
            "class_label": [
                "Normal",
                "Lung Opacity",
                "Lung Opacity",
                "No Lung Opacity / Not Normal",
                "Normal",
                "Bogus",
            ],
        }
    )


@pytest.fixture
def clean_label_df() -> pd.DataFrame:
    """A de-duplicated, label-restricted frame with a known distribution."""
    return pd.DataFrame(
        {
            "patientId": [f"p{i}" for i in range(9)],
            "class_label": [
                "Normal",
                "Normal",
                "Normal",
                "Lung Opacity",
                "Lung Opacity",
                "Lung Opacity",
                "No Lung Opacity / Not Normal",
                "No Lung Opacity / Not Normal",
                "No Lung Opacity / Not Normal",
            ],
        }
    )


@pytest.fixture
def gray_image() -> np.ndarray:
    """A 2D uint16 grayscale pixel array, like a decoded DICOM."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 4096, size=(64, 64), dtype=np.uint16)


@pytest.fixture
def classes() -> list[str]:
    return list(CLASSES)
