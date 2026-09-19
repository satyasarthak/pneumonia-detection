"""Tests for stratified splitting.

Feature: pneumonia-detection
Covers Property 8 (Req 3.4) and Property 9 (Req 3.5).
"""

from __future__ import annotations

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.constants import CLASSES
from src.split import stratified_split


@st.composite
def balanced_frames(draw):
    """Generate frames with enough samples per class for a 3-way split."""
    # Enough samples per class that the smallest subset (test ~= 15%) still
    # holds several samples, so proportions are meaningful within tolerance.
    per_class = draw(st.integers(min_value=40, max_value=120))
    pids = []
    labels = []
    idx = 0
    for cls in CLASSES:
        for _ in range(per_class):
            pids.append(f"p{idx}")
            labels.append(cls)
            idx += 1
    return pd.DataFrame({"patientId": pids, "class_label": labels})


# --- Property 8: Dataset partition is disjoint and covering ---
@settings(max_examples=100)
@given(df=balanced_frames())
def test_property_8_disjoint_and_covering(df):
    """Feature: pneumonia-detection, Property 8: Dataset partition is disjoint
    and covering. Validates Requirements 3.4."""
    train, val, test = stratified_split(df)

    s_train = set(train["patientId"])
    s_val = set(val["patientId"])
    s_test = set(test["patientId"])

    # Pairwise disjoint.
    assert s_train.isdisjoint(s_val)
    assert s_train.isdisjoint(s_test)
    assert s_val.isdisjoint(s_test)
    # Covering.
    assert s_train | s_val | s_test == set(df["patientId"])
    assert len(train) + len(val) + len(test) == len(df)


# --- Property 9: Stratified split preserves class proportions ---
@settings(max_examples=100)
@given(df=balanced_frames())
def test_property_9_preserves_proportions(df):
    """Feature: pneumonia-detection, Property 9: Stratified split preserves
    class proportions. Validates Requirements 3.5."""
    train, val, test = stratified_split(df)
    overall = df["class_label"].value_counts(normalize=True)

    for subset in (train, val, test):
        props = subset["class_label"].value_counts(normalize=True)
        for cls in CLASSES:
            assert abs(props.get(cls, 0.0) - overall.get(cls, 0.0)) < 0.1
