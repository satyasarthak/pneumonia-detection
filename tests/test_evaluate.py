"""Tests for evaluation, comparison, and selection.

Feature: pneumonia-detection
Covers Property 12 (Req 4.3, 5.3), Property 13 (Req 4.4, 5.4), and
Property 14 (Req 5.5). These use pure metric logic over generated label
vectors and result dicts, so no TensorFlow is required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.constants import NUM_CLASSES
from src.evaluate import compare_models, compute_metrics, select_best

_label_lists = st.lists(
    st.integers(min_value=0, max_value=NUM_CLASSES - 1), min_size=1, max_size=60
)


# --- Property 12: Multi-class metrics are valid ---
@settings(max_examples=100)
@given(data=st.data())
def test_property_12_metrics_valid(data):
    """Feature: pneumonia-detection, Property 12: Multi-class metrics are valid.
    Validates Requirements 4.3, 5.3."""
    n = data.draw(st.integers(min_value=1, max_value=60))
    y_true = data.draw(
        st.lists(st.integers(0, NUM_CLASSES - 1), min_size=n, max_size=n)
    )
    y_pred = data.draw(
        st.lists(st.integers(0, NUM_CLASSES - 1), min_size=n, max_size=n)
    )

    m = compute_metrics(y_true, y_pred)

    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        assert 0.0 <= m[key] <= 1.0

    # Each confusion-matrix row sums to that class's support (count in y_true).
    cm = m["confusion_matrix"]
    assert cm.shape == (NUM_CLASSES, NUM_CLASSES)
    y_true_arr = np.asarray(y_true)
    for cls in range(NUM_CLASSES):
        assert cm[cls].sum() == int((y_true_arr == cls).sum())


# --- Property 13: Model comparison table is complete and consistent ---
@st.composite
def result_dicts(draw):
    n_models = draw(st.integers(min_value=1, max_value=5))
    results = {}
    for i in range(n_models):
        results[f"model_{i}"] = {
            "accuracy": draw(st.floats(0, 1)),
            "macro_precision": draw(st.floats(0, 1)),
            "macro_recall": draw(st.floats(0, 1)),
            "macro_f1": draw(st.floats(0, 1)),
        }
    return results


@settings(max_examples=100)
@given(results=result_dicts())
def test_property_13_comparison_consistent(results):
    """Feature: pneumonia-detection, Property 13: Model comparison table is
    complete and consistent. Validates Requirements 4.4, 5.4."""
    table = compare_models(results)

    assert len(table) == len(results)
    assert set(table["model"]) == set(results.keys())
    expected_cols = {"model", "accuracy", "macro_precision", "macro_recall", "macro_f1"}
    assert set(table.columns) == expected_cols


# --- Property 14: Best-model selection is the metric optimum ---
@settings(max_examples=100)
@given(results=result_dicts())
def test_property_14_best_is_optimum(results):
    """Feature: pneumonia-detection, Property 14: Best-model selection is the
    metric optimum. Validates Requirements 5.5."""
    table = compare_models(results)
    best_name, rationale = select_best(table, metric="macro_f1")

    best_val = table.loc[table["model"] == best_name, "macro_f1"].iloc[0]
    assert best_val >= table["macro_f1"].max() - 1e-9
    assert isinstance(rationale, str) and len(rationale) > 0


# --- Example: metrics on a known confusion pattern ---
def test_compute_metrics_perfect_prediction():
    y = [0, 1, 2, 0, 1, 2]
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    cm = m["confusion_matrix"]
    assert np.array_equal(cm, np.diag([2, 2, 2]))
