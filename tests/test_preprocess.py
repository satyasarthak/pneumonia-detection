"""Tests for preprocessing transforms.

Feature: pneumonia-detection
Covers Property 6 (Req 3.2) and Property 7 (Req 3.6).
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from src.preprocess import GRAYSCALE, RGB, normalize, to_channels

_arrays_2d = hnp.arrays(
    dtype=st.sampled_from([np.uint16, np.int32, np.float32]),
    shape=st.tuples(
        st.integers(min_value=1, max_value=12),
        st.integers(min_value=1, max_value=12),
    ),
    elements=st.integers(min_value=0, max_value=4095),
)


# --- Property 6: Channel conversion produces the target format ---
@settings(max_examples=100)
@given(img=_arrays_2d)
def test_property_6_channel_conversion(img):
    """Feature: pneumonia-detection, Property 6: Channel conversion produces the
    target format. Validates Requirements 3.2."""
    gray = to_channels(img, GRAYSCALE)
    rgb = to_channels(img, RGB)

    assert gray.shape == (*img.shape, 1)
    assert rgb.shape == (*img.shape, 3)

    # Round trip grayscale -> rgb -> grayscale preserves spatial dims.
    back = to_channels(to_channels(img, RGB), GRAYSCALE)
    assert back.shape[:2] == img.shape


# --- Property 7: Normalization bounds pixel values ---
@settings(max_examples=100)
@given(img=_arrays_2d)
def test_property_7_normalization_bounds(img):
    """Feature: pneumonia-detection, Property 7: Normalization bounds pixel
    values. Validates Requirements 3.6."""
    out = normalize(img)

    assert out.min() >= 0.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6

    # Relative ordering preserved: min-max scaling is monotonic, so for any
    # pair of pixels the ordering (<, ==, >) is unchanged. Compare the sorted
    # inputs against the outputs reordered by the SAME permutation.
    flat_in = np.asarray(img, dtype=np.float64).ravel()
    flat_out = out.ravel()
    if flat_in.max() > flat_in.min():
        order = np.argsort(flat_in, kind="stable")
        sorted_out = flat_out[order]
        # Outputs must be non-decreasing along the input ordering.
        assert np.all(np.diff(sorted_out) >= -1e-6)
